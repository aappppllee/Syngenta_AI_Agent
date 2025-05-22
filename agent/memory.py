import logging
from typing import Dict, Optional, List as PyList, Union # Added Union
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
# from langchain.memory.chat_memory import BaseChatMemory # Removed problematic import
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage 
import asyncio 

from agent.agent_config import MemoryConfig, LLMConfig 
from app.config import settings 

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    def __init__(self, memory_config: MemoryConfig, llm_config: Optional[LLMConfig] = None):
        self.config = memory_config
        self.llm_config = llm_config 
        # Changed type hint to Union of concrete classes
        self._sessions_memory_wrappers: Dict[str, Union[ConversationBufferWindowMemory, ConversationSummaryBufferMemory]] = {} 

        if self.config.type == "sql_chat_history":
            if not self.config.db_connection_string:
                logger.error("db_connection_string not set in MemoryConfig, but type is 'sql_chat_history'. This will fail.")
                raise ValueError("db_connection_string is required in MemoryConfig for sql_chat_history type.")
            logger.info(f"ConversationMemoryManager initialized to use SQLChatMessageHistory. Table: {self.config.table_name}")
        elif self.config.buffer_type == "summary_buffer":
            if not self.llm_config:
                 logger.error("LLMConfig is required for 'summary_buffer' memory type but not provided.")
                 raise ValueError("LLMConfig is required for 'summary_buffer' memory type.")
            if not self.llm_config.api_key and self.llm_config.provider not in ["huggingface"]: # Assuming huggingface might not need key
                logger.error(f"API key for {self.llm_config.provider} required for summary_buffer LLM but not provided.")
                raise ValueError(f"API key for {self.llm_config.provider} required for summary_buffer LLM.")
        else: 
            logger.info(f"ConversationMemoryManager initialized for in-memory type: {self.config.buffer_type}. Data will be volatile if not SQL-backed.")


    def _create_sql_chat_history_instance(self, session_id: str) -> SQLChatMessageHistory:
        """Creates an instance of SQLChatMessageHistory for a given session."""
        if not self.config.db_connection_string: 
            logger.error("Cannot create SQLChatMessageHistory: Database connection string is missing.")
            raise ValueError("Database connection string is required for SQLChatMessageHistory.")
        logger.debug(f"Creating SQLChatMessageHistory for session '{session_id}' on table '{self.config.table_name}' using sync conn: {self.config.db_connection_string}")
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.config.db_connection_string, 
            table_name=self.config.table_name,
        )

    # Changed return type hint
    def _create_new_memory_wrapper(self, session_id: str) -> Union[ConversationBufferWindowMemory, ConversationSummaryBufferMemory]:
        """Creates a new Langchain memory wrapper object for the session."""
        
        chat_message_history_backend: Optional[SQLChatMessageHistory] = None
        if self.config.type == "sql_chat_history":
            chat_message_history_backend = self._create_sql_chat_history_instance(session_id)
            logger.info(f"Using SQLChatMessageHistory as backend for session '{session_id}'.")
        
        buffer_type_to_use = self.config.buffer_type

        if buffer_type_to_use == "buffer_window":
            logger.debug(f"Creating ConversationBufferWindowMemory (k={self.config.k}) for session '{session_id}'. SQL Backend: {bool(chat_message_history_backend)}")
            return ConversationBufferWindowMemory(
                k=self.config.k,
                chat_memory=chat_message_history_backend, 
                memory_key="chat_history",
                input_key="user_query", 
                output_key="ai_response",
                return_messages=False 
            )
        elif buffer_type_to_use == "summary_buffer":
            if not self.llm_config: 
                logger.error("LLMConfig required for summary_buffer but not available.")
                raise ValueError("LLMConfig required for summary_buffer.")
            if not self.llm_config.api_key and self.llm_config.provider not in ["huggingface"]:
                logger.error(f"API key for {self.llm_config.provider} is missing for summary_buffer LLM.")
                raise ValueError(f"API key for {self.llm_config.provider} is required for summary_buffer LLM.")
            
            llm_for_summary = None
            if self.llm_config.provider == "openai":
                llm_for_summary = ChatOpenAI(model_name=self.llm_config.model_name, api_key=self.llm_config.api_key, temperature=0.3)
            elif self.llm_config.provider == "anthropic":
                llm_for_summary = ChatAnthropic(model=self.llm_config.model_name, api_key=self.llm_config.api_key, temperature=0.3)
            elif self.llm_config.provider == "bedrock":
                logger.warning(f"Bedrock provider selected for Summary Buffer LLM ({self.llm_config.model_name}). API Key: {'SET' if self.llm_config.api_key else 'NOT SET'}. Ensure custom Bedrock integration.")
                raise NotImplementedError(f"Summary Buffer LLM provider '{self.llm_config.provider}' with model '{self.llm_config.model_name}' is not fully implemented yet. Requires custom Bedrock integration.")
            else: 
                logger.error(f"LLM provider {self.llm_config.provider} not implemented for summary memory LLM.")
                raise NotImplementedError(f"LLM provider {self.llm_config.provider} not implemented for summary memory.")

            logger.debug(f"Creating ConversationSummaryBufferMemory for session '{session_id}'. SQL Backend: {bool(chat_message_history_backend)}")
            return ConversationSummaryBufferMemory(
                llm=llm_for_summary,
                chat_memory=chat_message_history_backend, 
                max_token_limit=self.config.max_token_limit or 1500,
                memory_key="chat_history",
                input_key="user_query",
                return_messages=False
            )
        else:
            # This path should ideally return a ConversationBufferWindowMemory if it's the default
            logger.error(f"Unsupported memory buffer_type: {buffer_type_to_use}. Defaulting to volatile BufferWindow without SQL.")
            # Ensure this path also returns a type compatible with the Union
            return ConversationBufferWindowMemory(k=self.config.k, memory_key="chat_history", return_messages=False)

    # Changed return type hint
    def get_session_memory(self, session_id: str) -> Union[ConversationBufferWindowMemory, ConversationSummaryBufferMemory]:
        if session_id not in self._sessions_memory_wrappers:
            logger.info(f"No cached memory wrapper for session_id '{session_id}'. Creating new one.")
            self._sessions_memory_wrappers[session_id] = self._create_new_memory_wrapper(session_id)
        return self._sessions_memory_wrappers[session_id]

    async def save_interaction(self, session_id: str, user_query: str, ai_response_text: str):
        memory_wrapper = self.get_session_memory(session_id)
        try:
            logger.debug(f"Saving interaction to memory for session_id '{session_id}': User='{user_query[:50]}...', AI='{ai_response_text[:50]}...'")
            
            if hasattr(memory_wrapper, 'asave_context'): 
                 await memory_wrapper.asave_context(
                    {"user_query": user_query}, 
                    {"ai_response": ai_response_text} 
                )
            elif hasattr(memory_wrapper, 'chat_memory') and isinstance(memory_wrapper.chat_memory, SQLChatMessageHistory):
                sql_history: SQLChatMessageHistory = memory_wrapper.chat_memory # type: ignore
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, sql_history.add_user_message, HumanMessage(content=user_query))
                await loop.run_in_executor(None, sql_history.add_ai_message, AIMessage(content=ai_response_text))
            else: 
                logger.warning(f"Using synchronous save_context for session '{session_id}' as async not available or not SQL-backed as expected.")
                memory_wrapper.save_context({"user_query": user_query}, {"ai_response": ai_response_text})

            logger.info(f"Saved interaction to memory for session_id '{session_id}'.")
        except Exception as e:
            logger.error(f"Error saving interaction to memory for session '{session_id}': {e}", exc_info=settings.DEBUG_MODE)

    def clear_session_memory(self, session_id: str):
        logger.info(f"Attempting to clear memory for session_id '{session_id}'.")
        if session_id in self._sessions_memory_wrappers:
            memory_wrapper = self._sessions_memory_wrappers[session_id]
            if hasattr(memory_wrapper, 'chat_memory') and isinstance(memory_wrapper.chat_memory, SQLChatMessageHistory):
                try:
                    memory_wrapper.chat_memory.clear() 
                    logger.info(f"Cleared DB chat history for session_id '{session_id}'.")
                except Exception as e:
                    logger.error(f"Error clearing DB chat history for session '{session_id}': {e}", exc_info=settings.DEBUG_MODE)
            
            del self._sessions_memory_wrappers[session_id]
            logger.info(f"Cleared in-memory wrapper for session_id '{session_id}'.")
        else: 
            logger.warning(f"No memory wrapper found in cache for session_id '{session_id}' to clear.")
            if self.config.type == "sql_chat_history": 
                try:
                    sql_history_direct = self._create_sql_chat_history_instance(session_id)
                    sql_history_direct.clear() 
                    logger.info(f"Cleared DB chat history directly for session_id '{session_id}' (not found in cache).")
                except Exception as e:
                    logger.error(f"Error directly clearing DB chat history for session '{session_id}': {e}", exc_info=settings.DEBUG_MODE)
