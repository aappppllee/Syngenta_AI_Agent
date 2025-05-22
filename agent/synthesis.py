import logging
from typing import List as PyList, Dict, Any, Optional
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic # Uncomment if using Anthropic
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document as LangchainDocument
# from langchain.memory import ConversationBufferWindowMemory # For type hinting, though not used directly here

from agent.agent_config import LLMConfig
from app.config import settings # For DEBUG_MODE

logger = logging.getLogger(__name__)

class AnswerSynthesizer:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._init_llm()
        logger.info(f"AnswerSynthesizer initialized with LLM: {self.config.provider} - {self.config.model_name}")

    def _init_llm(self):
        logger.debug(f"Initializing Synthesizer LLM: Provider='{self.config.provider}', Model='{self.config.model_name}'")
        if self.config.provider == "openai":
            if not self.config.api_key:
                logger.error("OpenAI API key is required for OpenAI LLM but not found.")
                raise ValueError("OpenAI API key is required for OpenAI LLM.")
            return ChatOpenAI(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                # max_tokens=1000 # Optional: to control response length for synthesis
                # request_timeout=60 # Optional: longer timeout for synthesis
            )
        # Example for Anthropic
        # elif self.config.provider == "anthropic":
        #     if not self.config.api_key: 
        #         logger.error("Anthropic API key is required for Anthropic LLM but not found.")
        #         raise ValueError("Anthropic API key is required for Anthropic LLM.")
        #     return ChatAnthropic(
        #         model_name=self.config.model_name, # Langchain might use slightly different model names
        #         api_key=self.config.api_key,
        #         temperature=self.config.temperature
        #     )
        else:
            logger.error(f"Unsupported LLM provider for AnswerSynthesizer: {self.config.provider}")
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def synthesize_answer(
        self,
        original_query: str,
        retrieved_chunks: PyList[LangchainDocument], # These are the context documents
        chat_history: Optional[str] = None # Formatted string of chat history
    ) -> Dict[str, Any]:
        """
        Generates a synthesized answer using an LLM based on the original query, 
        retrieved document chunks (context), and optional chat history.

        Args:
            original_query (str): The user's original question.
            retrieved_chunks (List[LangchainDocument]): Relevant document chunks from the retriever or context pieces.
            chat_history (Optional[str]): Formatted string of past turns in the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the synthesized answer details.
                            Expected to match parts of the AnswerDetail Pydantic schema.
        """
        if not retrieved_chunks:
            logger.warning("AnswerSynthesizer: No retrieved chunks provided for synthesis.")
            return {
                "type": "no_info_found_for_synthesis", # More specific type
                "content": "I could not find specific information to construct an answer for your query based on the available context.",
                "summary": "No relevant context found for synthesis.",
                "explanation": "The information retrieval step did not yield results for the given query to synthesize an answer from."
            }

        # Updated prompt to include chat_history placeholder
        # This prompt guides the LLM on how to use the context (chunks) and history to answer the query.
        prompt_template_str = """You are an AI assistant for a supply chain company. Your task is to provide a comprehensive and helpful answer to the user's question.
Base your answer *only* on the provided "Context from documents/data" and consider the "Chat History" for follow-up questions or context.
Be concise and directly answer the question. If the provided context does not contain enough information to answer the question, clearly state that you cannot answer fully based on the provided information.
Do not make up information or answer questions outside of the provided context.

Chat History (if any):
{chat_history}

Context from documents/data:
{context}

User's Current Question: {question}

Helpful and Comprehensive Answer (based *only* on the chat history and context above):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["chat_history", "context", "question"]
        )
        
        # Ensure chat_history is a string, even if empty, for the prompt
        formatted_chat_history = chat_history if chat_history and chat_history.strip() else "No previous conversation history relevant to this query."

        # Using load_qa_chain with "stuff" method.
        # This method "stuffs" all document chunks into a single LLM call.
        # Good for when the total length of chunks + query + history fits within the LLM's context window.
        # Other chain_types: "map_reduce", "refine", "map_rerank" for larger contexts.
        chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT, verbose=settings.DEBUG_MODE)
        
        logger.info(f"Synthesizing answer for query: '{original_query}' with {len(retrieved_chunks)} context chunks.")
        if settings.DEBUG_MODE:
            for i, doc in enumerate(retrieved_chunks):
                logger.debug(f"  Context Chunk {i+1} for Synthesis (source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:200]}...")
            if chat_history:
                logger.debug(f"  Chat History for Synthesis: {formatted_chat_history[:200]}...")


        try:
            # The chain expects 'input_documents' (for context) and 'question'.
            # We also pass 'chat_history' as per our custom prompt.
            # LangchainDocument objects are passed directly in the list for 'input_documents'.
            
            # Using arun for async execution if the chain and underlying LLM support it.
            # If not, chain.run() or chain.__call__() would be used, potentially in an executor.
            # For load_qa_chain with ChatOpenAI, arun should work.
            response_text = await chain.arun({ # arun returns the string output directly
                "input_documents": retrieved_chunks, 
                "question": original_query,
                "chat_history": formatted_chat_history
            })
            
            # If chain.invoke was used, response would be a dict: response = await chain.ainvoke(...) ; response_text = response.get("output_text")

            synthesized_text = response_text.strip()

            logger.info(f"Synthesized answer successfully. Length: {len(synthesized_text)}")
            
            # Basic summary (could be improved, e.g., by asking LLM for a separate summary in another call)
            summary = f"Answer synthesized based on {len(retrieved_chunks)} provided context piece(s)."
            if len(synthesized_text) > 150: # Create a snippet if long
                 summary = synthesized_text[:147] + "..."
            elif not synthesized_text: # Handle empty LLM response
                summary = "LLM provided an empty response for synthesis."
                synthesized_text = "I was unable to formulate a specific answer based on the provided information."
                logger.warning(f"AnswerSynthesizer: LLM returned empty response for query '{original_query}'.")


            return {
                "type": "synthesized_text", # Indicates successful synthesis
                "content": synthesized_text,
                "summary": summary,
                # "sources" will be handled by AgentCore based on original chunk metadata from retriever
                "explanation": "Answer generated by AI based on information retrieved and contextual history."
            }
        except Exception as e:
            logger.error(f"Error during answer synthesis for query '{original_query}': {e}", exc_info=settings.DEBUG_MODE)
            return {
                "type": "error_synthesis",
                "content": "I encountered an issue while trying to formulate an answer. Please try rephrasing or ask again later.",
                "summary": "Synthesis error.",
                "explanation": f"Internal error during answer generation: {str(e)}" if settings.DEBUG_MODE else "Internal error."
            }
