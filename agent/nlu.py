import json
import logging
from typing import Dict, Any, Optional, List as PyList
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic # If you switch provider
from langchain.prompts import PromptTemplate, FewShotPromptTemplate # For few-shot examples
from langchain.chains import LLMChain
from pydantic import BaseModel, Field as PydanticField

from agent.agent_config import NLUConfig
from app.config import settings # For DEBUG_MODE

logger = logging.getLogger(__name__)

class ExtractedEntity(BaseModel):
    entity_type: str = PydanticField(description="Type of the extracted entity.")
    value: Any = PydanticField(description="Value of the extracted entity. For 'filter_criteria', this could be a structured dictionary like {'column': 'region', 'operator': 'equals', 'filter_value': 'Southwest'}.")
    original_text: Optional[str] = PydanticField(None, description="Original text span from the query that corresponds to this entity.")

class NLUOutput(BaseModel):
    intent: str = PydanticField(description="The determined user intent.")
    entities: PyList[ExtractedEntity] = PydanticField(default_factory=list, description="List of entities extracted from the query.")
    policy_query_component: Optional[str] = PydanticField(None, description="The part of the user's query that asks about a policy, definition, or rule (especially for hybrid_query).")
    data_query_component: Optional[str] = PydanticField(None, description="The part of the user's query that asks for data from the database (especially for hybrid_query or database_query).")
    confidence: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Confidence score for the intent classification (0.0 to 1.0).")
    requires_clarification: bool = PydanticField(False, description="True if the query is too vague and needs clarification.")
    clarification_question: Optional[str] = PydanticField(None, description="If clarification is needed, a question to ask the user.")
    raw_llm_response: Optional[str] = PydanticField(None, description="Raw response from the LLM for debugging.")

class NLUIntentParser:
    def __init__(self, config: NLUConfig):
        self.config = config
        try:
            self.llm = self._init_llm()
            # self.prompt_template = self._get_zero_shot_prompt_template() # Using zero-shot
            self.prompt_template = self._get_few_shot_prompt_template() # Using few-shot
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=settings.DEBUG_MODE)
            logger.info(f"NLUIntentParser initialized with LLM: {self.config.provider} - {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize NLUIntentParser: {e}", exc_info=settings.DEBUG_MODE)
            raise

    def _init_llm(self):
        logger.debug(f"Initializing NLU LLM: Provider='{self.config.provider}', Model='{self.config.model_name}'")
        if self.config.provider == "openai":
            if not self.config.api_key:
                logger.error("OpenAI API key is missing for NLU LLM.")
                raise ValueError("OpenAI API key is required for NLU LLM.")
            return ChatOpenAI(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                request_timeout=30 
            )
        else:
            logger.error(f"Unsupported LLM provider for NLU: {self.config.provider}")
            raise ValueError(f"Unsupported LLM provider for NLU: {self.config.provider}")

    def _get_few_shot_prompt_template(self) -> FewShotPromptTemplate:
        examples = [
            {
                "user_query": "What is our company policy on inventory write-offs?",
                "chat_history": "None",
                "json_output": json.dumps({
                    "intent": "document_search",
                    "entities": [{"entity_type": "policy_topic", "value": "inventory write-offs", "original_text": "inventory write-offs"}],
                    "policy_query_component": "What is our company policy on inventory write-offs?",
                    "data_query_component": None,
                    "confidence": 0.95,
                    "requires_clarification": False
                })
            },
            {
                "user_query": "How much inventory do we currently have in the Southwest region for product ID 123?",
                "chat_history": "None",
                "json_output": json.dumps({
                    "intent": "database_query",
                    "entities": [
                        {"entity_type": "metric", "value": "inventory quantity", "original_text": "How much inventory"},
                        {"entity_type": "filter_criteria", "value": {"product_id": 123, "region": "Southwest"}, "original_text": "Southwest region for product ID 123"}
                    ],
                    "policy_query_component": None,
                    "data_query_component": "How much inventory do we currently have in the Southwest region for product ID 123?",
                    "confidence": 0.9,
                    "requires_clarification": False
                })
            },
            {
                "user_query": "Which inventory items qualify as no-movers according to our policy, and how many do we currently have?",
                "chat_history": "None",
                "json_output": json.dumps({
                    "intent": "hybrid_query",
                    "entities": [
                        {"entity_type": "policy_defined_term", "value": "no-movers", "original_text": "no-movers"},
                        {"entity_type": "policy_name", "value": "inventory policy", "original_text": "our policy"}, # Implied
                        {"entity_type": "metric", "value": "count of items", "original_text": "how many"}
                    ],
                    "policy_query_component": "Which inventory items qualify as no-movers according to our policy",
                    "data_query_component": "how many no-mover items do we currently have",
                    "confidence": 0.92,
                    "requires_clarification": False
                })
            },
            {
                "user_query": "Tell me about profit.",
                "chat_history": "None",
                "json_output": json.dumps({
                    "intent": "clarification_needed",
                    "entities": [{"entity_type": "clarification_reason", "value": "The query 'Tell me about profit' is too general. Please specify what aspect of profit you are interested in (e.g., profit margins for specific products, overall company profit for a period, or the policy related to profit calculation)."}],
                    "policy_query_component": None,
                    "data_query_component": None,
                    "confidence": 0.8,
                    "requires_clarification": True,
                    "clarification_question": "Could you be more specific about what profit information you're looking for? For example, are you interested in profit margins, overall profit, or a policy related to profit?"
                })
            },
             {
                "user_query": "Hi there!",
                "chat_history": "None",
                "json_output": json.dumps({
                    "intent": "greeting",
                    "entities": [],
                    "policy_query_component": None,
                    "data_query_component": None,
                    "confidence": 1.0,
                    "requires_clarification": False
                })
            }
        ]

        example_prompt = PromptTemplate(
            input_variables=["user_query", "chat_history", "json_output"],
            template="User Query: \"{user_query}\"\nChat History: {chat_history}\nJSON Output:\n{json_output}"
        )

        prefix = """
        You are an expert NLU system for a Supply Chain AI Agent. Your task is to analyze the user's query, considering the chat history if provided.
        Determine their intent, extract relevant entities, and for hybrid queries, clearly separate the policy-related question component from the data-related question component.

        Available Intents: "document_search", "database_query", "hybrid_query", "greeting", "clarification_needed", "unsupported_query", "chit_chat".
        Entity Types: "policy_name", "policy_defined_term", "metric", "filter_criteria" (value can be a string or a structured dict like {{"column": "region", "operator": "equals", "filter_value": "Southwest"}}), "keywords", "date_range", "region", "question_focus", "table_name", "clarification_reason".

        Output Format (Return ONLY a valid JSON object):
        {{
          "intent": "...",
          "entities": [ {{"entity_type": "...", "value": "...", "original_text": "..."}}, ... ],
          "policy_query_component": "For 'hybrid_query' or 'document_search'. Null otherwise.",
          "data_query_component": "For 'hybrid_query' or 'database_query'. Null otherwise.",
          "confidence": 0.0-1.0,
          "requires_clarification": boolean,
          "clarification_question": "If requires_clarification is true, suggest a question to ask the user for more details. Otherwise null."
        }}
        For "hybrid_query", strive to populate both "policy_query_component" and "data_query_component".
        If "clarification_needed", the "clarification_question" should guide the user.
        
        Here are some examples:
        """
        suffix = "User Query: \"{user_query}\"\nChat History: {chat_history}\nJSON Output:\n"

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["user_query", "chat_history"],
            example_separator="\n\n"
        )

    async def parse_query(self, query_text: str, chat_history: Optional[str] = None) -> NLUOutput:
        logger.info(f"NLUParser: Parsing query: '{query_text}'")
        if chat_history:
            logger.debug(f"NLUParser: With chat history (first 100 chars): '{chat_history[:100]}'")
        
        formatted_chat_history = chat_history if chat_history and chat_history.strip() else "None" # Consistent with examples
        raw_response_content = "NLU LLM call did not occur or failed before response." 

        try:
            raw_response_content = await self.chain.arun(user_query=query_text, chat_history=formatted_chat_history)
            logger.debug(f"NLUParser: Raw LLM response string: {raw_response_content}")

            # Clean the response: LLMs sometimes wrap JSON in ```json ... ``` or add explanations.
            if "```json" in raw_response_content:
                raw_response_content = raw_response_content.split("```json")[1].split("```")[0].strip()
            elif raw_response_content.startswith("```") and raw_response_content.endswith("```"):
                 raw_response_content = raw_response_content[3:-3].strip()


            parsed_json = json.loads(raw_response_content)
            
            validated_entities = []
            if "entities" in parsed_json and isinstance(parsed_json["entities"], list):
                for entity_dict in parsed_json["entities"]:
                    if isinstance(entity_dict, dict): 
                        validated_entities.append(ExtractedEntity(**entity_dict))
                    else:
                        logger.warning(f"NLUParser: Found non-dict item in entities list from LLM: {entity_dict}")
            
            nlu_out = NLUOutput(
                intent=parsed_json.get("intent", "unsupported_query"),
                entities=validated_entities,
                policy_query_component=parsed_json.get("policy_query_component"),
                data_query_component=parsed_json.get("data_query_component"),
                confidence=parsed_json.get("confidence"),
                requires_clarification=parsed_json.get("requires_clarification", False),
                clarification_question=parsed_json.get("clarification_question"),
                raw_llm_response=raw_response_content if settings.DEBUG_MODE else None
            )
            logger.info(f"NLUParser: Parsed output - Intent='{nlu_out.intent}', Entities Cnt={len(nlu_out.entities)}, PolicyComp='{nlu_out.policy_query_component}', DataComp='{nlu_out.data_query_component}'")
            return nlu_out

        except json.JSONDecodeError as e_json:
            logger.error(f"NLUParser: Error decoding LLM JSON response: {e_json}. Raw response: {raw_response_content}", exc_info=settings.DEBUG_MODE)
            return NLUOutput(intent="error_parsing_nlu", entities=[ExtractedEntity(entity_type="error", value=f"LLM response not valid JSON. Details: {e_json}. Raw: {raw_response_content[:500]}")], raw_llm_response=raw_response_content)
        except Exception as e_llm: 
            logger.error(f"NLUParser: Error during NLU LLM call or processing for query '{query_text}': {e_llm}", exc_info=settings.DEBUG_MODE)
            return NLUOutput(intent="error_processing_nlu", entities=[ExtractedEntity(entity_type="error", value=f"NLU processing failed: {str(e_llm)}")], raw_llm_response=str(e_llm))

