import logging
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document as LangchainDocument

from app.db.database import AsyncSessionLocal
from app.db.crud import policy_document_crud, audit_log_crud
from agent.agent_config import DocumentRetrieverConfig
from app.config import settings
from app.models.db_models import ActionTypeEnum, PolicyDocument as DBPolicyDocument # Alias to avoid confusion
from app.schemas.audit_log_schemas import AuditLogCreate

logger = logging.getLogger(__name__)

# Define a type alias for LangchainDocument for clarity
LangchainDoc = LangchainDocument

class DocumentRetriever:
    def __init__(self, config: DocumentRetrieverConfig):
        self.config = config
        logger.info(f"Initializing DocumentRetriever with config: {config.model_dump(exclude={'embedding_config': {'api_key'}})}")
        try:
            self.embedding_model = self._init_embedding_model()
            self.vector_store = self._init_vector_store()

            if not os.path.exists(self.config.pdf_storage_base_path):
                os.makedirs(self.config.pdf_storage_base_path, exist_ok=True)
                logger.info(f"Created PDF storage directory: {self.config.pdf_storage_base_path}")
            else:
                logger.info(f"PDF storage directory already exists: {self.config.pdf_storage_base_path}")
        except Exception as e:
            logger.error(f"CRITICAL Error initializing DocumentRetriever: {e}", exc_info=settings.DEBUG_MODE)
            raise

    def _init_embedding_model(self):
        emb_config = self.config.embedding_config
        logger.debug(f"Initializing embedding model: Provider='{emb_config.provider}', Model='{emb_config.model_name}'")
        if emb_config.provider == "openai":
            if not emb_config.api_key:
                logger.error("OpenAI API key is required for OpenAI embeddings but not found.")
                raise ValueError("OpenAI API key is required for OpenAI embeddings.")
            return OpenAIEmbeddings(
                model=emb_config.model_name,
                api_key=emb_config.api_key
            )
        else:
            logger.error(f"Unsupported embedding provider: {emb_config.provider}")
            raise ValueError(f"Unsupported embedding provider: {emb_config.provider}")

    def _init_vector_store(self) -> PGVector:
        vs_config = self.config.vector_store_config
        logger.debug(f"Initializing vector store: Type='{vs_config.type}', Collection='{vs_config.collection_name}'")
        if vs_config.type == "pgvector":
            if not vs_config.connection_string:
                logger.error("PGVector connection string is required but not found.")
                raise ValueError("PGVector connection string is required.")
            try:
                store = PGVector(
                    connection_string=vs_config.connection_string,
                    embedding_function=self.embedding_model,
                    collection_name=vs_config.collection_name,
                    distance_strategy=DistanceStrategy.COSINE
                )
                logger.info(f"PGVector store initialized successfully for collection '{vs_config.collection_name}'.")
                return store
            except Exception as e:
                logger.error(f"Failed to initialize PGVector store: {e}", exc_info=settings.DEBUG_MODE)
                raise
        else:
            logger.error(f"Unsupported vector store type: {vs_config.type}")
            raise ValueError(f"Unsupported vector store type: {vs_config.type}")

    async def _load_and_split_document(
        self,
        db_doc_model: DBPolicyDocument,
        text_splitter: RecursiveCharacterTextSplitter
    ) -> List[LangchainDoc]:
        """
        Loads a single document from its storage path and splits it into chunks.
        Returns a list of LangchainDocument chunks.
        """
        if not db_doc_model.storage_path or not os.path.exists(db_doc_model.storage_path):
            logger.warning(f"Document ID {db_doc_model.document_id} ('{db_doc_model.title}') has invalid storage_path '{db_doc_model.storage_path}' or file not found. Skipping.")
            return []

        logger.debug(f"Processing document for chunking: '{db_doc_model.title}' from '{db_doc_model.storage_path}'")
        try:
            if db_doc_model.storage_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(db_doc_model.storage_path)
            elif db_doc_model.storage_path.lower().endswith(".txt"):
                loader = TextLoader(db_doc_model.storage_path, encoding="utf-8")
            else:
                logger.warning(f"Unsupported file type for document ID {db_doc_model.document_id}: {db_doc_model.storage_path}. Skipping.")
                return []

            loaded_pages = loader.load() # Returns a list of LangchainDocument objects

            for page_doc in loaded_pages:
                page_doc.metadata["policy_document_id"] = str(db_doc_model.document_id)
                page_doc.metadata["document_title"] = db_doc_model.title
                page_doc.metadata["original_file_name"] = db_doc_model.original_file_name or os.path.basename(db_doc_model.storage_path)
                if "page" not in page_doc.metadata:
                    page_doc.metadata["page"] = 0

            page_chunks = text_splitter.split_documents(loaded_pages)

            for i, chunk_doc in enumerate(page_chunks):
                chunk_doc.metadata["chunk_sequence_within_doc"] = i
            
            logger.debug(f"Document '{db_doc_model.title}' (ID: {db_doc_model.document_id}) yielded {len(page_chunks)} chunks.")
            return page_chunks
        except Exception as e_load_split:
            logger.error(f"Error loading/splitting document ID {db_doc_model.document_id} ('{db_doc_model.title}'): {e_load_split}", exc_info=settings.DEBUG_MODE)
            return []

    async def _get_documents_to_index_from_db(
        self, db: AsyncSession, policy_document_ids: Optional[List[int]] = None
    ) -> AsyncGenerator[DBPolicyDocument, None]:
        """
        Asynchronously yields database document models to be indexed, one by one or in small batches.
        This avoids loading all document metadata into memory at once.
        """
        if policy_document_ids:
            for doc_id in policy_document_ids:
                db_doc = await policy_document_crud.get_policy_document(db, document_id=doc_id)
                if db_doc:
                    yield db_doc
                else:
                    logger.warning(f"Document ID {doc_id} not found in database for indexing.")
        else:
            # Implement pagination for fetching all documents to avoid high memory usage
            offset = 0
            limit = 100 # Batch size for fetching from DB
            while True:
                docs_batch = await policy_document_crud.get_policy_documents(db, skip=offset, limit=limit)
                if not docs_batch:
                    break
                for db_doc in docs_batch:
                    yield db_doc
                offset += len(docs_batch)
                if len(docs_batch) < limit: # Last page
                    break

    async def index_documents_from_db(
        self,
        policy_document_ids: Optional[List[int]] = None,
        batch_size_for_vector_store: int = 100 # Number of chunks to send to vector store at once
    ):
        logger.info(f"Starting document indexing process. Target IDs: {policy_document_ids if policy_document_ids else 'All available'}. Batch size for vector store: {batch_size_for_vector_store}")
        db: AsyncSession = AsyncSessionLocal()
        total_docs_processed = 0
        total_chunks_generated = 0
        successfully_indexed_docs_count = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True
        )

        chunks_batch_for_embedding: List[LangchainDoc] = []

        try:
            async for db_doc_model in self._get_documents_to_index_from_db(db, policy_document_ids):
                total_docs_processed += 1
                doc_chunks = await self._load_and_split_document(db_doc_model, text_splitter)
                
                if doc_chunks:
                    chunks_batch_for_embedding.extend(doc_chunks)
                    total_chunks_generated += len(doc_chunks)
                    successfully_indexed_docs_count +=1 # Mark as processed for indexing if chunks were generated

                # If current batch of chunks reaches size or no more docs, embed and add to vector store
                if len(chunks_batch_for_embedding) >= batch_size_for_vector_store or (not policy_document_ids and total_docs_processed % 10 == 0): # Heuristic for "all docs"
                    if chunks_batch_for_embedding:
                        logger.info(f"Embedding and adding {len(chunks_batch_for_embedding)} chunks to vector store...")
                        try:
                            if hasattr(self.vector_store, 'aadd_documents'):
                                await self.vector_store.aadd_documents(chunks_batch_for_embedding)
                            else:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, self.vector_store.add_documents, chunks_batch_for_embedding)
                            logger.info(f"Successfully added {len(chunks_batch_for_embedding)} chunks.")
                        except Exception as e_pgvector_batch:
                            logger.error(f"Error adding batch of {len(chunks_batch_for_embedding)} documents to PGVector: {e_pgvector_batch}", exc_info=settings.DEBUG_MODE)
                            # Decide on error handling: skip batch, retry, or halt. For now, log and continue.
                            # Could mark docs in this batch as failed.
                        chunks_batch_for_embedding = [] # Reset batch

            # Add any remaining chunks
            if chunks_batch_for_embedding:
                logger.info(f"Embedding and adding final batch of {len(chunks_batch_for_embedding)} chunks to vector store...")
                try:
                    if hasattr(self.vector_store, 'aadd_documents'):
                        await self.vector_store.aadd_documents(chunks_batch_for_embedding)
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self.vector_store.add_documents, chunks_batch_for_embedding)
                    logger.info(f"Successfully added final {len(chunks_batch_for_embedding)} chunks.")
                except Exception as e_pgvector_final:
                    logger.error(f"Error adding final batch of {len(chunks_batch_for_embedding)} documents to PGVector: {e_pgvector_final}", exc_info=settings.DEBUG_MODE)
                chunks_batch_for_embedding = []

            if total_chunks_generated > 0:
                await audit_log_crud.create_audit_log(db, AuditLogCreate(
                    action_type=ActionTypeEnum.DOC_INDEXED,
                    details=f"Indexing complete. Processed {total_docs_processed} documents, generated {total_chunks_generated} chunks. Successfully submitted chunks for {successfully_indexed_docs_count} documents. Target IDs: {policy_document_ids if policy_document_ids else 'All'}.",
                    accessed_resource=self.config.vector_store_config.collection_name
                ))
                await db.commit()
            else:
                logger.info("No processable chunks were generated from the documents.")
                await db.rollback() # No audit log if nothing was processed

        except Exception as e_main_index:
            logger.error(f"An error occurred during the main document indexing process: {e_main_index}", exc_info=settings.DEBUG_MODE)
            await db.rollback()
        finally:
            await db.close()
        logger.info(f"Document indexing process finished. Processed {total_docs_processed} docs, generated {total_chunks_generated} chunks. Submitted chunks for {successfully_indexed_docs_count} documents.")

    async def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = 5, # Default k value
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[LangchainDoc]:
        if not self.vector_store:
            logger.error("DocumentRetriever: Vector store not initialized. Cannot retrieve.")
            return []

        # Optimization: Potentially cache results for identical (query, k, filter_criteria) tuples
        # if queries are frequently repeated and data doesn't change often.
        # Requires a proper caching mechanism (e.g., Redis with TTL).
        # cache_key = f"retrieve_chunks:{query}:{k}:{json.dumps(filter_criteria, sort_keys=True)}"
        # cached_result = await get_from_cache(cache_key)
        # if cached_result:
        #     logger.info(f"Retrieved {len(cached_result)} chunks from cache for query '{query}'.")
        #     return cached_result

        logger.debug(f"DocumentRetriever: Retrieving top {k} chunks for query: '{query}', with filter: {filter_criteria}")
        try:
            if hasattr(self.vector_store, 'asimilarity_search'):
                relevant_docs = await self.vector_store.asimilarity_search(query, k=k, filter=filter_criteria)
            else:
                loop = asyncio.get_event_loop()
                relevant_docs = await loop.run_in_executor(None, self.vector_store.similarity_search, query, k, filter_criteria)

            logger.info(f"DocumentRetriever: Retrieved {len(relevant_docs)} chunks for query '{query}'.")
            if settings.DEBUG_MODE:
                for i, doc in enumerate(relevant_docs):
                    logger.debug(f"  Chunk {i+1}: Source='{doc.metadata.get('document_title', 'N/A')}', Page='{doc.metadata.get('page', 'N/A')}', Content Snippet='{doc.page_content[:150]}...'")
            
            # await save_to_cache(cache_key, relevant_docs, ttl=3600) # Example: cache for 1 hour
            return relevant_docs
        except Exception as e:
            logger.error(f"DocumentRetriever: Error during similarity search for query '{query}': {e}", exc_info=settings.DEBUG_MODE)
            return []
