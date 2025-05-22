import json
import logging
from typing import Dict, Any, Optional, List as PyList
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document as LangchainDocument

from agent.retrievers.document_retriever import DocumentRetriever
from agent.retrievers.database_interactor import DatabaseInteractor
from agent.synthesis import AnswerSynthesizer
from agent.agent_config import HybridOrchestratorConfig, CriteriaExtractorLLMConfig
from agent.nlu import NLUOutput, ExtractedEntity 
from app.config import settings

logger = logging.getLogger(__name__)

class HybridQueryOrchestrator:
    def __init__(self,
                 doc_retriever: DocumentRetriever,
                 db_interactor: DatabaseInteractor,
                 answer_synthesizer: AnswerSynthesizer,
                 criteria_extractor_llm_config: CriteriaExtractorLLMConfig,
                 config: HybridOrchestratorConfig):
        self.doc_retriever = doc_retriever
        self.db_interactor = db_interactor
        self.answer_synthesizer = answer_synthesizer
        self.criteria_extractor_llm_config = criteria_extractor_llm_config
        self.config = config
        try:
            self.criteria_extraction_llm = self._init_criteria_extraction_llm()
            self.criteria_extraction_prompt = self._get_criteria_extraction_prompt()
            self.criteria_extraction_chain = LLMChain(llm=self.criteria_extraction_llm, prompt=self.criteria_extraction_prompt, verbose=settings.DEBUG_MODE)
            logger.info("HybridQueryOrchestrator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for HybridQueryOrchestrator: {e}", exc_info=settings.DEBUG_MODE)
            raise

    def _init_criteria_extraction_llm(self): # (No change in logic)
        logger.debug(f"Initializing Criteria Extractor LLM: {self.criteria_extractor_llm_config.provider} - {self.criteria_extractor_llm_config.model_name}")
        if self.criteria_extractor_llm_config.provider == "openai":
            if not self.criteria_extractor_llm_config.api_key: raise ValueError("OpenAI API key required for Criteria Extractor.")
            return ChatOpenAI(model_name=self.criteria_extractor_llm_config.model_name, api_key=self.criteria_extractor_llm_config.api_key, temperature=self.criteria_extractor_llm_config.temperature)
        else: raise ValueError(f"Unsupported LLM provider for Criteria Extractor: {self.criteria_extractor_llm_config.provider}")

    def _get_criteria_extraction_prompt(self) -> PromptTemplate:
        # Refined prompt for criteria extraction
        prompt_template_str = """
        You are an AI assistant specializing in extracting specific, actionable criteria from policy documents to inform data queries.
        Given the Policy Document Text and the User's Policy Query Component (which specifies what definition, rule, or condition is being sought), your task is to extract the precise criteria.

        Instructions for Extraction:
        1.  Focus on quantifiable conditions, timeframes (e.g., 'within X days', 'older than Y months', 'less than Z units'), specific terms that act as flags (e.g., "obsolete", "non-compliant"), or thresholds.
        2.  The extracted criteria should be directly stated or clearly and unambiguously derivable from the provided policy text. Do NOT infer or assume criteria not present.
        3.  If multiple distinct criteria are relevant to the policy query component, list them clearly, perhaps using bullet points or a numbered list if appropriate.
        4.  If no specific, actionable, and quantifiable criteria matching the policy query component are found in the provided text, respond *only* with the exact phrase "No specific actionable criteria found".
        5.  The output should be the criteria themselves, not a sentence saying "The criteria are...".

        Chat History (for overall context, if any):
        {chat_history}

        Policy Document Text (Source of truth for criteria):
        ---
        {policy_text}
        ---

        User's Policy Query Component (What specific definition, rule, or condition is the user asking about from the policy? This is your primary guide for extraction.):
        ---
        {policy_query_component}
        ---

        Extracted Criteria (Be precise and stick to the text. E.g., "items with no sales in 180 days", "inventory older than 12 months AND valued below $50", or "No specific actionable criteria found"):
        """
        return PromptTemplate(template=prompt_template_str, input_variables=["chat_history", "policy_text", "policy_query_component"])


    async def _extract_criteria_from_text(self, text_chunks: PyList[LangchainDocument], policy_query_component: str, chat_history: Optional[str] = None) -> Optional[str]:
        if not text_chunks:
            logger.warning("HybridOrchestrator: No text chunks provided for criteria extraction.")
            return None
        if not policy_query_component:
            logger.warning("HybridOrchestrator: No policy_query_component provided for criteria extraction context. This is crucial for relevance.")
            return None 

        context_str = "\n\n".join([chunk.page_content for chunk in text_chunks])
        
        try:
            logger.debug(f"HybridOrchestrator: Calling LLM for criteria extraction. Context length: {len(context_str)}, Policy Query Component: '{policy_query_component}'")
            response = await self.criteria_extraction_chain.arun({ # Use arun for async
                "chat_history": chat_history or "None", 
                "policy_text": context_str, 
                "policy_query_component": policy_query_component
            })
            extracted_criteria = response.strip() # LLMChain.arun returns string
            
            logger.info(f"HybridOrchestrator: Raw extracted criteria from LLM: '{extracted_criteria}'")
            if "no specific actionable criteria found" in extracted_criteria.lower() or \
               "no specific criteria found" in extracted_criteria.lower() or \
               not extracted_criteria:
                logger.warning("HybridOrchestrator: LLM indicated no specific actionable criteria found from document chunks for the given policy component.")
                return None
            return extracted_criteria
        except Exception as e:
            logger.error(f"HybridOrchestrator: Error extracting criteria from text: {e}", exc_info=settings.DEBUG_MODE)
            return None

    async def process_query(self, original_query: str, nlu_output: NLUOutput, chat_history: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"HybridOrchestrator: Processing hybrid query: '{original_query}'")
        
        policy_query_part_for_doc_retrieval = nlu_output.policy_query_component
        policy_query_part_for_criteria_extraction = nlu_output.policy_query_component
        data_query_part_for_db = nlu_output.data_query_component
        
        if not policy_query_part_for_doc_retrieval:
            logger.warning("HybridOrchestrator: NLU missing 'policy_query_component'. Using full query for document search.")
            policy_query_part_for_doc_retrieval = original_query
        if not policy_query_part_for_criteria_extraction: 
             logger.error("HybridOrchestrator: NLU missing 'policy_query_component' which is vital for criteria extraction context. Aborting hybrid query.")
             return {"type":"error_hybrid_missing_nlu_policy_component", "content":"Could not determine the specific policy information needed from your query for the hybrid operation.", "summary": "NLU did not identify policy component."}
        if not data_query_part_for_db:
            logger.warning("HybridOrchestrator: NLU missing 'data_query_component'. Using full query for data part, which might be suboptimal.")
            data_query_part_for_db = original_query

        logger.debug(f"HybridOrchestrator - Doc Retrieval Query: '{policy_query_part_for_doc_retrieval}'")
        logger.debug(f"HybridOrchestrator - Criteria Extraction Context: '{policy_query_part_for_criteria_extraction}'")
        logger.debug(f"HybridOrchestrator - Data Query Component: '{data_query_part_for_db}'")

        doc_chunks = await self.doc_retriever.retrieve_relevant_chunks(query=policy_query_part_for_doc_retrieval, k=3) # k=3 for more context
        if not doc_chunks:
            return {"type":"error_hybrid_no_doc", "content":"Could not find relevant policy documents to extract criteria for your query.", "summary": "Policy context missing for hybrid query."}

        extracted_criteria_text = await self._extract_criteria_from_text(doc_chunks, policy_query_part_for_criteria_extraction, chat_history)
        if not extracted_criteria_text:
            return {"type":"error_hybrid_no_criteria", "content":f"Found policy documents related to '{policy_query_part_for_criteria_extraction}', but could not extract the specific actionable criteria needed to query the database.", "summary": "Criteria extraction failed for hybrid query."}
        logger.info(f"HybridOrchestrator: Successfully extracted criteria: '{extracted_criteria_text}'")

        # Formulate database query using the data_query_part and extracted_criteria_text
        # This prompt for NL-to-SQL needs to be clear that the criteria are from policy.
        db_nl_query = f"Given the following rule derived from company policy: '{extracted_criteria_text}'. Now, using this rule, please address the data request: '{data_query_part_for_db}'."
        
        logger.info(f"HybridOrchestrator: Querying database with refined NL for SQL generation: '{db_nl_query}'")
        db_interaction_result = await self.db_interactor.query_database(natural_language_query=db_nl_query)

        if db_interaction_result.get("error"):
            return {"type":"error_hybrid_db_query", "content":f"Error querying database with extracted criteria: {db_interaction_result['error']}", "summary": "Hybrid DB query failed.", "explanation": f"NL query for DB: '{db_nl_query}'. Attempted SQL (if any): {db_interaction_result.get('generated_sql')}"}
        logger.debug(f"HybridOrchestrator: DB interaction successful. SQL: {db_interaction_result.get('generated_sql')}, Result summary: {db_interaction_result.get('natural_summary')}")

        # Synthesize final answer
        policy_context_for_synthesis = f"Regarding the policy aspect ('{policy_query_part_for_criteria_extraction}'), the relevant criteria found and applied is: \"{extracted_criteria_text}\"."
        data_context_for_synthesis = f"Regarding the data aspect ('{data_query_part_for_db}'), and applying the above policy criteria, the database information found is: {db_interaction_result.get('natural_summary') or db_interaction_result.get('query_result')}."
        
        # The original_query for the synthesizer should be the user's full question to provide overall context.
        # The retrieved_chunks are the structured policy and data findings.
        combined_docs_for_synthesis = [
            LangchainDocument(page_content=policy_context_for_synthesis, metadata={"source": "Policy Interpretation", "derived_from_docs": [d.metadata.get("document_title", "N/A") for d in doc_chunks]}),
            LangchainDocument(page_content=data_context_for_synthesis, metadata={"source": "Database Query", "generated_sql": db_interaction_result.get('generated_sql', "N/A")})
        ]
        
        logger.info("HybridOrchestrator: Synthesizing final answer from combined policy and data context.")
        final_synthesized_response = await self.answer_synthesizer.synthesize_answer(
            original_query=original_query, 
            retrieved_chunks=combined_docs_for_synthesis,
            chat_history=chat_history
        )
        
        final_synthesized_response["type"] = "hybrid_result"
        final_synthesized_response["sources"] = [{"document_title":chunk.metadata.get("document_title"), "document_id": chunk.metadata.get("policy_document_id"), "page_number": chunk.metadata.get("page")} for chunk in doc_chunks]
        final_synthesized_response["sources"].append({"document_title": "Supply Chain Database", "document_id": "DATABASE"})
        
        current_explanation = final_synthesized_response.get("explanation", "Answer derived from policy and database.")
        final_synthesized_response["explanation"] = (f"{current_explanation} "
                                                   f"Policy Criteria Applied: '{extracted_criteria_text}'. "
                                                   f"Database Query (NL for SQL step based on '{data_query_part_for_db}' and criteria). "
                                                   f"Generated SQL: '{db_interaction_result.get('generated_sql')}'.")
        logger.info("HybridOrchestrator: Successfully processed hybrid query.")
        return final_synthesized_response
