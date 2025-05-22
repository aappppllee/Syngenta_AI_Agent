import logging
import uuid
from typing import List, Dict, Any, Optional

from langchain.docstore.document import Document as LangchainDocument

from agent.agent_config import AgentCoreConfig
from agent.nlu import NLUIntentParser, NLUOutput, ExtractedEntity
from agent.retrievers.document_retriever import DocumentRetriever
from agent.retrievers.database_interactor import DatabaseInteractor
from agent.orchestrators import HybridQueryOrchestrator
from agent.synthesis import AnswerSynthesizer
from agent.access_control import AccessController
from agent.memory import ConversationMemoryManager
from app.db.database import AsyncSessionLocal
from app.db.crud import audit_log_crud
from app.schemas.audit_log_schemas import AuditLogCreate
from app.models.db_models import ActionTypeEnum
from app.config import settings # For SENSITIVE_DATA_KEYWORDS

logger = logging.getLogger(__name__)

class AgentCore:
    def __init__(self, config: AgentCoreConfig):
        logger.info("Initializing AgentCore with refined NLU/Orchestration logic...")
        self.config = config
        try:
            self.nlu_parser = NLUIntentParser(config=self.config.nlu_config)
            self.document_retriever = DocumentRetriever(config=self.config.doc_retriever_config)
            llm_for_memory = self.config.llm_config if self.config.memory_config.type == "sql_chat_history" and self.config.memory_config.buffer_type == "summary_buffer" else None
            self.memory_manager = ConversationMemoryManager(memory_config=self.config.memory_config, llm_config=llm_for_memory)
            self.answer_synthesizer = AnswerSynthesizer(config=self.config.llm_config)
            self.db_interactor = DatabaseInteractor(llm_config=self.config.sql_llm_config, db_interactor_config=self.config.db_interactor_config)
            self.hybrid_orchestrator = HybridQueryOrchestrator(
                doc_retriever=self.document_retriever,
                db_interactor=self.db_interactor,
                answer_synthesizer=self.answer_synthesizer,
                criteria_extractor_llm_config=self.config.criteria_extractor_llm_config,
                config=self.config.hybrid_orchestrator_config
            )
            self.access_controller = AccessController(config=self.config.access_control_config)
            logger.info("AgentCore initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR during AgentCore initialization: {e}", exc_info=True)
            raise

    async def _log_access_denied(self, user_context: dict, intent: str, reason: str, query_text: str):
        db_session = AsyncSessionLocal()
        try:
            log_entry = AuditLogCreate(user_id=user_context.get("user_id"), action_type=ActionTypeEnum.PERMISSION_DENIED, details=f"Denied for intent '{intent}'. Reason: {reason}. Query: '{query_text}'", access_granted=False, user_role_context=user_context.get("role_name"), user_region_context=user_context.get("assigned_region"))
            await audit_log_crud.create_audit_log(db_session, log_entry=log_entry)
            await db_session.commit()
        except Exception as e:
            await db_session.rollback(); logger.error(f"Failed to log access denied event: {e}", exc_info=True)
        finally: await db_session.close()

    async def _determine_resource_details_for_access_check(self, intent: str, nlu_output: NLUOutput, query_text: str) -> Dict[str, Any]:
        resource_details: Dict[str, Any] = {"resource_type": "unknown", "resource_name": "unknown", "target_region": None, "query_text_hint": query_text[:100]}
        entities = nlu_output.entities

        region_entity = next((e for e in entities if e.entity_type == "region"), None)
        if region_entity and isinstance(region_entity.value, str):
            resource_details["target_region"] = region_entity.value.lower().strip()

        if intent == "document_search":
            resource_details["resource_type"] = "document"
            name_source = nlu_output.policy_query_component # NLU now provides this
            if not name_source: # Fallback if NLU didn't populate it
                policy_name_entity = next((e for e in entities if e.entity_type == "policy_name"), None)
                if policy_name_entity and isinstance(policy_name_entity.value, str): name_source = policy_name_entity.value
                else: name_source = "general_document_topic"
            resource_details["resource_name"] = name_source.strip().lower().replace(" ", "_")
        
        elif intent == "database_query" or intent == "hybrid_query":
            resource_details["resource_type"] = "database_query"
            name_parts = []
            target_tables = [e.value for e in entities if e.entity_type == "table_name" and isinstance(e.value, str)]
            target_metrics = [e.value for e in entities if e.entity_type == "metric" and isinstance(e.value, str)]
            if target_tables: name_parts.append(f"tables:{','.join(target_tables)}")
            if target_metrics: name_parts.append(f"metrics:{','.join(target_metrics)}")
            
            text_to_check = query_text.lower() + " " + " ".join(str(e.value).lower() for e in entities)
            is_financial = any(keyword in text_to_check for keyword in settings.SENSITIVE_DATA_KEYWORDS)

            if is_financial:
                resource_details["resource_type"] = "financial_data"
                resource_details["resource_name"] = "_".join(name_parts) if name_parts else "general_financial"
            elif name_parts:
                resource_details["resource_type"] = "database_resource"
                resource_details["resource_name"] = "_".join(name_parts)
            else:
                resource_details["resource_name"] = "general_database_access"
        
        elif intent == "greeting" or intent == "chit_chat":
            resource_details["resource_type"] = "interaction"; resource_details["resource_name"] = intent
        else:
            resource_details["resource_type"] = "general_intent"; resource_details["resource_name"] = intent
            
        logger.debug(f"AgentCore: Determined resource_details for AC: {resource_details}")
        return resource_details

    async def handle_query(self, query_text: str, user_context: dict, session_id: Optional[str] = None) -> Dict[str, Any]:
        effective_session_id = session_id if session_id else str(uuid.uuid4())
        if not session_id: logger.info(f"AgentCore: New session started with ID: {effective_session_id}")
        
        final_ai_response_text_for_memory = "Error: Could not generate a response."
        # Always include session_id in the base of the dict returned by handle_query,
        # as QueryService will build QueryResponse model from this.
        answer_detail_dict: Dict[str, Any] = {
            "type": "error", 
            "content": "An unexpected error occurred while processing your request.",
            "session_id": effective_session_id # For QueryResponse model
        }

        try:
            session_memory_wrapper = self.memory_manager.get_session_memory(effective_session_id)
            memory_vars = await session_memory_wrapper.aload_memory_variables({}) 
            chat_history_str = memory_vars.get("chat_history", "")
            logger.info(f"AgentCore: Handling query for User: '{user_context.get('username', 'Unknown')}', Session: '{effective_session_id}', Query: '{query_text}'")
            logger.debug(f"AgentCore: Loaded chat history (len {len(chat_history_str)}) for session '{effective_session_id}'.")

            nlu_result: NLUOutput = await self.nlu_parser.parse_query(query_text, chat_history=chat_history_str)
            intent = nlu_result.intent; entities = nlu_result.entities # entities is PyList[ExtractedEntity]
            logger.info(f"AgentCore: NLU result - Intent='{intent}', Entities Cnt={len(entities)}, PolicyComp='{nlu_result.policy_query_component}', DataComp='{nlu_result.data_query_component}'")

            if nlu_result.requires_clarification or intent.startswith("error_"):
                reason = entities[0].value if entities and intent.startswith("error_") and isinstance(entities[0].value, str) else "Query is vague or NLU failed."
                if nlu_result.requires_clarification: 
                    reason_entity = next((e for e in entities if e.entity_type == "clarification_reason"), None)
                    if reason_entity and isinstance(reason_entity.value, str): reason = reason_entity.value
                ai_resp_text = f"I need a bit more clarity. {reason}" if nlu_result.requires_clarification else f"Sorry, I had trouble understanding that. {reason}"
                answer_detail_dict = {"type": "clarification_needed" if nlu_result.requires_clarification else "error_nlu", "content": ai_resp_text, "summary":"Query understanding issue."}
            else:
                resource_details = await self._determine_resource_details_for_access_check(intent, nlu_result, query_text)
                is_auth, reason_denial = await self.access_controller.check_permission(user_context, intent, entities, resource_details)
                
                if not is_auth:
                    logger.warning(f"AgentCore: Access DENIED. User: '{user_context.get('username')}', Reason: {reason_denial}")
                    await self._log_access_denied(user_context, intent, reason_denial or "Not specified", query_text)
                    ai_resp_text = reason_denial or "Access Denied."
                    answer_detail_dict = {"type": "access_denied", "content": ai_resp_text, "summary": "Access Denied.", "explanation": "Permission check failed."}
                else:
                    logger.info(f"AgentCore: Access GRANTED for User: '{user_context.get('username')}'")
                    
                    # Main Intent Handling Logic
                    if intent == "document_search":
                        doc_query = nlu_result.policy_query_component or query_text
                        logger.debug(f"AgentCore: Document search using effective query: '{doc_query}'")
                        chunks = await self.document_retriever.retrieve_relevant_chunks(doc_query, k=3)
                        if chunks: 
                            synth_resp = await self.answer_synthesizer.synthesize_answer(query_text, chunks, chat_history_str)
                            answer_detail_dict.update(synth_resp)
                            answer_detail_dict["sources"] = [{"document_title":c.metadata.get("document_title"), "document_id": c.metadata.get("policy_document_id"), "page_number": c.metadata.get("page")} for c in chunks]
                        else: answer_detail_dict.update({"type": "no_info_found_docs", "content": "No relevant documents found for your specific query.", "summary": "No documents found."})
                    
                    elif intent == "database_query":
                        db_query_nl = nlu_result.data_query_component or query_text
                        logger.debug(f"AgentCore: Database query using effective NL: '{db_query_nl}'")
                        db_res = await self.db_interactor.query_database(db_query_nl)
                        if db_res.get("error"): 
                            answer_detail_dict.update({"type": "error_db", "content": f"Error querying database: {db_res['error']}", "explanation": f"SQL attempt: {db_res.get('generated_sql')}"})
                        else:
                            db_content = db_res.get("natural_summary") or str(db_res.get("query_result"))
                            db_doc = LangchainDocument(page_content=f"Database result: {db_content}. SQL: {db_res.get('generated_sql')}", metadata={"source":"Database Query"})
                            synth_db_answer = await self.answer_synthesizer.synthesize_answer(query_text, [db_doc], chat_history_str)
                            answer_detail_dict.update(synth_db_answer)
                            answer_detail_dict["type"] = "db_result_synthesized"
                            answer_detail_dict["explanation"] = f"Generated SQL: {db_res.get('generated_sql')}. {answer_detail_dict.get('explanation', '')}"
                            answer_detail_dict["sources"] = [{"document_title":"Supply Chain Database", "document_id": "DATABASE"}]
                    
                    elif intent == "hybrid_query":
                        logger.debug(f"AgentCore: Hybrid query processing using full NLU output.")
                        hybrid_result = await self.hybrid_orchestrator.process_query(original_query=query_text, nlu_output=nlu_result, chat_history=chat_history_str)
                        answer_detail_dict.update(hybrid_result)
                    
                    elif intent == "greeting": 
                        answer_detail_dict.update({"type": "greeting", "content": "Hello! How can I assist with your supply chain questions today?", "summary": "Greeting."})
                    elif intent == "chit_chat": 
                        answer_detail_dict.update({"type": "chit_chat", "content": "I am a supply chain assistant. Do you have a question about policies or data?", "summary": "Chit-chat."})
                    else: # Handles unsupported_query or any other intent not explicitly routed
                        answer_detail_dict.update({
                            "type": "unsupported_intent", 
                            "content": f"I'm not fully equipped to handle requests of type '{intent}' yet. Please try rephrasing or ask a different question about supply chain policies or data.",
                            "summary": f"Intent '{intent}' not fully supported."
                        })
            
            final_ai_response_text_for_memory = answer_detail_dict.get("content", "Processed request.")

        except Exception as e:
            logger.error(f"AgentCore: Unhandled exception in handle_query for query '{query_text}', session '{effective_session_id}': {e}", exc_info=True)
            answer_detail_dict = {
                "type": "error_agent_core",
                "content": "I encountered an internal problem while processing your request. Please try again later.",
                "summary": "Internal processing error.",
                "explanation": f"Details: {str(e)}" if settings.DEBUG_MODE else "Internal error."
            }
            final_ai_response_text_for_memory = answer_detail_dict["content"]
        
        try:
            await self.memory_manager.save_interaction(effective_session_id, query_text, final_ai_response_text_for_memory)
        except Exception as e_mem:
            logger.error(f"AgentCore: Failed to save interaction to memory for session '{effective_session_id}': {e_mem}", exc_info=True)

        # Ensure session_id is part of the final dictionary for QueryResponse model
        # This dict becomes the 'answer' field in QueryResponse. QueryService adds session_id to QueryResponse itself.
        # answer_detail_dict["session_id_debug_agentcore"] = effective_session_id # For debugging if needed
        return answer_detail_dict

    async def load_models_and_resources(self): logger.info("AgentCore: Loading models and resources... Done.")

#--------------------------------------------------------------------------
# File: app/main.py (No functional changes needed here for this refinement)
#--------------------------------------------------------------------------
# Placeholder for brevity.
