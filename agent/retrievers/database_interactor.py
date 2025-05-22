import asyncio
import logging
from typing import Dict, Any, Optional, List as PyList
from sqlalchemy import create_engine, text # Added text for direct SQL execution if needed
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic # If using Anthropic
from langchain_experimental.sql import SQLDatabaseChain # Using this for now
# from langchain.chains import create_sql_query_chain # For more direct SQL generation
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool # For agents
# from langchain.agents import create_sql_agent, AgentType # For more complex SQL agent

from agent.agent_config import DatabaseInteractorConfig, SQLLLMConfig
from app.config import settings

logger = logging.getLogger(__name__)

class DatabaseInteractor:
    def __init__(self, llm_config: SQLLLMConfig, db_interactor_config: DatabaseInteractorConfig):
        self.llm_config = llm_config
        self.db_interactor_config = db_interactor_config
        
        if not self.db_interactor_config.db_url:
            logger.error("DatabaseInteractor: Synchronous Database URL (db_url) is required but not found in config.")
            raise ValueError("Synchronous Database URL (db_url) is required for DatabaseInteractor.")
        
        self.llm = self._init_llm()
        try:
            self.db_engine = create_engine(self.db_interactor_config.db_url, echo=settings.DEBUG_MODE) # Sync engine
            
            # Test connection (optional but good for early failure detection)
            with self.db_engine.connect() as connection:
                logger.info("DatabaseInteractor: Successfully connected to synchronous database.")

            self.sql_database = SQLDatabase(
                engine=self.db_engine,
                include_tables=self.db_interactor_config.include_tables if self.db_interactor_config.include_tables else None,
                sample_rows_in_table_info=self.db_interactor_config.sample_rows_in_table_info
            )
            
            # Using SQLDatabaseChain for a straightforward NL -> SQL -> Result flow
            self.db_chain = SQLDatabaseChain.from_llm(
                llm=self.llm,
                db=self.sql_database,
                verbose=settings.DEBUG_MODE,
                return_intermediate_steps=True, # Crucial to get the generated SQL
                # top_k=5, # Max number of results to return from SQL query (can be set here or in agent)
                return_direct=self.db_interactor_config.return_direct_sql_response, # If true, returns raw SQL result string; if false, LLM tries to summarize
                use_query_checker=True, # Uses an internal LLM call to check/fix the SQL
                # query_checker_prompt= # Optional: custom prompt for query checker
            )
            logger.info(f"DatabaseInteractor initialized. Querying tables: {self.db_interactor_config.include_tables or 'All discoverable by SQLDatabase'}.")
            if self.db_interactor_config.include_tables:
                logger.debug(f"DatabaseInteractor: SQLDatabase usable table names: {self.sql_database.get_usable_table_names()}")

        except Exception as e:
            logger.error(f"Error initializing DatabaseInteractor SQL components: {e}", exc_info=settings.DEBUG_MODE)
            raise

    def _init_llm(self):
        logger.debug(f"Initializing SQL LLM: Provider='{self.llm_config.provider}', Model='{self.llm_config.model_name}'")
        if self.llm_config.provider == "openai":
            if not self.llm_config.api_key:
                logger.error("OpenAI API key is missing for SQL LLM.")
                raise ValueError("OpenAI API key is required for SQL LLM.")
            return ChatOpenAI(
                model_name=self.llm_config.model_name,
                api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
            )
        # Add other providers here if needed
        # elif self.llm_config.provider == "anthropic":
        #     if not self.llm_config.api_key: raise ValueError("Anthropic API key required.")
        #     return ChatAnthropic(model=self.llm_config.model_name, api_key=self.llm_config.api_key, temperature=self.llm_config.temperature)
        else:
            logger.error(f"Unsupported LLM provider for SQL: {self.llm_config.provider}")
            raise ValueError(f"Unsupported LLM provider for SQL: {self.llm_config.provider}")

    async def query_database(self, natural_language_query: str) -> Dict[str, Any]:
        logger.info(f"DatabaseInteractor: Processing NL query for database: '{natural_language_query}'")
        response_payload = {
            "generated_sql": None,
            "query_result": None, # This will store the raw result from the DB
            "natural_summary": None, # This will store the LLM's summary of the result if not return_direct
            "error": None
        }

        try:
            chain_input = {"query": natural_language_query}
            
            # Run the synchronous Langchain SQLDatabaseChain call in a thread pool executor
            loop = asyncio.get_event_loop()
            logger.debug("DatabaseInteractor: Executing SQLDatabaseChain in executor...")
            # SQLDatabaseChain.invoke is synchronous
            result_dict = await loop.run_in_executor(None, self.db_chain.invoke, chain_input)
            logger.debug(f"DatabaseInteractor: SQLDatabaseChain raw result_dict: {result_dict}")

            # SQLDatabaseChain with return_intermediate_steps=True returns a dict:
            # {'query': 'NL query', 'result': 'Final answer/summary from LLM', 
            #  'intermediate_steps': [{'sql_cmd': 'SQL query', 'sql_result': 'Raw DB result'}]}
            # The 'result' key contains the LLM's natural language answer after executing the SQL.
            # If return_direct=True, 'result' contains the raw SQL output string.

            response_payload["natural_summary"] = result_dict.get("result") # This is the LLM's final answer/summary
            
            intermediate_steps = result_dict.get("intermediate_steps", [])
            if intermediate_steps and isinstance(intermediate_steps, list) and len(intermediate_steps) > 0:
                # The SQL command and its direct result are usually in one of the steps.
                # SQLDatabaseChain often has the SQL generation and execution as one step,
                # or query checking might add more steps. We look for the last 'sql_cmd'.
                last_sql_step = next((step for step in reversed(intermediate_steps) if "sql_cmd" in step and "sql_result" in step), None)
                
                if last_sql_step:
                    response_payload["generated_sql"] = last_sql_step.get("sql_cmd")
                    response_payload["query_result"] = last_sql_step.get("sql_result") # This is the raw data from DB
                else:
                    logger.warning("DatabaseInteractor: Could not extract SQL command and result from intermediate_steps.")
                    response_payload["error"] = "Internal error: Could not extract SQL query and raw result from chain steps."
                    # Log the intermediate steps for debugging if SQL extraction fails
                    logger.debug(f"DatabaseInteractor: Full intermediate_steps for failed SQL extraction: {intermediate_steps}")
            else:
                logger.warning("DatabaseInteractor: No intermediate_steps found in SQLDatabaseChain result.")
                # If return_direct_sql_response was True, the 'result' field of result_dict IS the raw SQL output.
                if self.db_interactor_config.return_direct_sql_response:
                    response_payload["query_result"] = result_dict.get("result")
                    response_payload["natural_summary"] = "Direct SQL result returned, no LLM summary." # Adjust if needed
                else:
                     # If no intermediate steps and not return_direct, natural_summary already has the LLM's answer.
                     # query_result might be missing if not explicitly extracted.
                     logger.warning("DatabaseInteractor: No intermediate_steps and not return_direct. Raw query_result might be missing.")


            logger.info(f"DatabaseInteractor: NL Query='{natural_language_query}' -> SQL='{response_payload['generated_sql']}'")
            logger.debug(f"DatabaseInteractor: Raw Query Result (first 200 chars): {str(response_payload['query_result'])[:200] if response_payload['query_result'] else 'N/A'}")
            logger.info(f"DatabaseInteractor: LLM Natural Summary/Final Answer: {response_payload['natural_summary']}")

        except Exception as e:
            logger.error(f"DatabaseInteractor: Error during database query for NL='{natural_language_query}': {e}", exc_info=settings.DEBUG_MODE)
            response_payload["error"] = f"Failed to process database query: {str(e)}"
        
        return response_payload
