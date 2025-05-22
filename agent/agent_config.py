from pydantic import BaseModel, Field
from typing import Optional, List as PyList
from app.config import settings as global_settings # Use an alias to avoid confusion

class EmbeddingConfig(BaseModel):
    provider: str = Field("openai", description="Embedding provider, e.g., 'openai', 'huggingface'")
    model_name: str = Field(default_factory=lambda: global_settings.EMBEDDING_MODEL_NAME, description="Name of the embedding model to use.")
    api_key: Optional[str] = Field(default_factory=lambda: global_settings.OPENAI_API_KEY, description="API key for the embedding provider, if required.")

class VectorStoreConfig(BaseModel):
    type: str = Field("pgvector", description="Type of vector store, e.g., 'pgvector', 'faiss', 'chroma'")
    connection_string: Optional[str] = Field(default_factory=lambda: global_settings.PGVECTOR_CONNECTION_STRING, description="Database connection string for SQL-based vector stores like PGVector.")
    collection_name: str = Field(default_factory=lambda: global_settings.PGVECTOR_COLLECTION_NAME, description="Name of the collection or table within the vector store.")

class DocumentRetrieverConfig(BaseModel):
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunk_size: int = Field(1000, description="Target size for text chunks during document splitting.")
    chunk_overlap: int = Field(200, description="Overlap between consecutive text chunks.")
    pdf_storage_base_path: str = Field("./uploaded_policy_docs/", description="Base filesystem path where PDF documents are stored for indexing.")

class LLMConfig(BaseModel): 
    provider: str = Field("openai", description="LLM provider, e.g., 'openai', 'anthropic'")
    model_name: str = Field(default_factory=lambda: global_settings.LLM_MODEL_NAME, description="Name of the primary LLM for tasks like answer synthesis.")
    api_key: Optional[str] = Field(default_factory=lambda: global_settings.OPENAI_API_KEY if global_settings.OPENAI_API_KEY else global_settings.ANTHROPIC_API_KEY, description="API key for the LLM provider.")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature for LLM responses.")
    max_tokens: Optional[int] = Field(1500, description="Optional maximum tokens for LLM response generation.")

class NLUConfig(BaseModel): 
    provider: str = Field("openai", description="LLM provider for NLU tasks.")
    model_name: str = Field(default_factory=lambda: global_settings.NLU_LLM_MODEL_NAME, description="LLM model for NLU.")
    api_key: Optional[str] = Field(default_factory=lambda: global_settings.OPENAI_API_KEY if global_settings.OPENAI_API_KEY else global_settings.ANTHROPIC_API_KEY)
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Temperature for NLU LLM.")

class SQLLLMConfig(BaseModel): 
    provider: str = Field("openai", description="LLM provider for SQL generation.")
    model_name: str = Field(default_factory=lambda: global_settings.SQL_LLM_MODEL_NAME, description="LLM model for NL-to-SQL conversion.")
    api_key: Optional[str] = Field(default_factory=lambda: global_settings.OPENAI_API_KEY if global_settings.OPENAI_API_KEY else global_settings.ANTHROPIC_API_KEY)
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Temperature for SQL LLM.")

class CriteriaExtractorLLMConfig(BaseModel):
    provider: str = Field("openai", description="LLM provider for criteria extraction.")
    model_name: str = Field(default_factory=lambda: global_settings.CRITERIA_EXTRACTION_LLM_MODEL_NAME, description="LLM model for extracting criteria from text.")
    api_key: Optional[str] = Field(default_factory=lambda: global_settings.OPENAI_API_KEY if global_settings.OPENAI_API_KEY else global_settings.ANTHROPIC_API_KEY)
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Temperature for criteria extraction LLM.")

class DatabaseInteractorConfig(BaseModel):
    db_url: Optional[str] = Field(default_factory=lambda: global_settings.SYNC_DATABASE_URL, description="Synchronous database URL for Langchain SQLDatabase.")
    include_tables: Optional[PyList[str]] = Field(
        default_factory=lambda: [tbl.strip() for tbl in global_settings.SUPPLY_CHAIN_TABLE_NAMES.split(',')] if global_settings.SUPPLY_CHAIN_TABLE_NAMES else [],
        description="List of table names the SQL agent can query."
    )
    sample_rows_in_table_info: int = Field(3, ge=0, description="Number of sample rows to include in table schema info for LLM context.")
    max_sql_iterations: int = Field(5, ge=1, description="Max iterations for SQL agent if it uses self-correction.")
    return_direct_sql_response: bool = Field(False, description="If true, DatabaseInteractor returns raw SQL result; if false, LLM attempts to summarize it.")

class HybridOrchestratorConfig(BaseModel):
    pass 

class AccessControlConfig(BaseModel):
    pass 

class MemoryConfig(BaseModel):
    type: str = Field("sql_chat_history", description="Type of memory store ('in_memory' or 'sql_chat_history').")
    buffer_type: str = Field("buffer_window", description="Type of Langchain memory wrapper ('buffer_window', 'summary_buffer').")
    k: int = Field(5, ge=1, description="Window size for ConversationBufferWindowMemory.")
    max_token_limit: Optional[int] = Field(1500, ge=100, description="Token limit for ConversationSummaryBufferMemory.")
    db_connection_string: Optional[str] = Field(default_factory=lambda: global_settings.SYNC_DATABASE_URL, description="Synchronous DB connection string for chat history.")
    table_name: str = Field(default_factory=lambda: global_settings.CHAT_MESSAGE_TABLE_NAME, description="Database table name for storing chat messages.")

class AgentCoreConfig(BaseModel):
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    nlu_config: NLUConfig = Field(default_factory=NLUConfig)
    sql_llm_config: SQLLLMConfig = Field(default_factory=SQLLLMConfig)
    criteria_extractor_llm_config: CriteriaExtractorLLMConfig = Field(default_factory=CriteriaExtractorLLMConfig)
    doc_retriever_config: DocumentRetrieverConfig = Field(default_factory=DocumentRetrieverConfig)
    db_interactor_config: DatabaseInteractorConfig = Field(default_factory=DatabaseInteractorConfig)
    hybrid_orchestrator_config: HybridOrchestratorConfig = Field(default_factory=HybridOrchestratorConfig)
    access_control_config: AccessControlConfig = Field(default_factory=AccessControlConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
