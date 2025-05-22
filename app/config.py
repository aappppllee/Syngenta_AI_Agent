import os
from typing import Optional, List as PyList
from pydantic_settings import BaseSettings, SettingsConfigDict # Import SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Intelligent Supply Chain Agent API"
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@host:port/db") # Async for FastAPI app
    SYNC_DATABASE_URL: Optional[str] = os.getenv("SYNC_DATABASE_URL", "postgresql://user:pass@host:port/db") # Sync for Langchain SQL tools & Chat History
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_changed_for_production_env")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ADMIN_ROLE_NAME: str = "Admin"

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY") # If using Claude
    BEDROCK_API_KEY: Optional[str] = os.getenv("BEDROCK_API_KEY") # Bedrock API Key
    
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4.1-turbo") 
    NLU_LLM_MODEL_NAME: str = os.getenv("NLU_LLM_MODEL_NAME", "gpt-3.5-turbo") 
    SQL_LLM_MODEL_NAME: str = os.getenv("SQL_LLM_MODEL_NAME", "gpt-3.5-turbo") 
    CRITERIA_EXTRACTION_LLM_MODEL_NAME: str = os.getenv("CRITERIA_EXTRACTION_LLM_MODEL_NAME", "gpt-3.5-turbo") 
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large") 
    
    PGVECTOR_CONNECTION_STRING: Optional[str] = os.getenv("PGVECTOR_CONNECTION_STRING") 
    PGVECTOR_COLLECTION_NAME: str = os.getenv("PGVECTOR_COLLECTION_NAME", "policy_document_embeddings")
    
    SUPPLY_CHAIN_TABLE_NAMES: Optional[str] = os.getenv("SUPPLY_CHAIN_TABLE_NAMES", "scm_customers,scm_products,scm_categories,scm_departments,scm_orders,scm_orderitems")
    
    SENSITIVE_DATA_KEYWORDS: PyList[str] = ["profit", "margin", "financials", "salary", "revenue", "cost breakdown", "p&l"]
    FINANCE_ROLE_NAME: str = "Finance User" 
    PLANNING_ROLE_NAME: str = "Planning User"
    
    DATAGO_CSV_FILE_PATH: str = os.getenv("DATAGO_CSV_FILE_PATH", "./DataCoSupplyChainDataset.csv") 
    CHAT_MESSAGE_TABLE_NAME: str = os.getenv("CHAT_MESSAGE_TABLE_NAME", "scm_chat_messages") 

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() 
    LOG_FILE_PATH: Optional[str] = os.getenv("LOG_FILE_PATH", None) 
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"

    # Pydantic V2 style model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'  # Explicitly set extra to 'ignore' (which is default but good to be clear)
    )

settings = Settings()
