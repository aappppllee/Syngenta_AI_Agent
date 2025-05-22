from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List as PyList, Optional

from app.models.db_models import DocumentEmbedding
# from app.schemas.embedding_schemas import DocumentEmbeddingCreate # Usually not created via API like this

# Note: Direct CRUD for embeddings is often handled by the vector store library 
# (e.g., Langchain's PGVector wrapper manages its own table and data).
# These functions are provided for completeness if you need to interact with
# your defined `DocumentEmbedding` ORM model directly, perhaps for observation,
# custom management, or if you are populating this table manually alongside Langchain.

async def get_document_embedding_by_id(db: AsyncSession, embedding_id: str) -> Optional[DocumentEmbedding]:
    """Retrieve a single document embedding by its ID."""
    result = await db.execute(
        select(DocumentEmbedding).filter(DocumentEmbedding.embedding_id == embedding_id)
    )
    return result.scalars().first()

async def get_embeddings_for_policy_document(
    db: AsyncSession, 
    policy_document_id: int, 
    skip: int = 0, 
    limit: int = 1000 # Default to a large limit if fetching all chunks for a doc
) -> PyList[DocumentEmbedding]:
    """Retrieve all embedding records associated with a specific policy document ID."""
    result = await db.execute(
        select(DocumentEmbedding)
        .filter(DocumentEmbedding.policy_document_id == policy_document_id)
        .order_by(DocumentEmbedding.created_at) # Or by a chunk sequence number if stored in metadata
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def create_document_embedding(
    db: AsyncSession, 
    embedding_id: str,
    policy_document_id: int,
    chunk_text: str,
    embedding_vector: PyList[float], # Expecting the raw vector
    metadata: Optional[str] = None # JSON string for metadata
) -> DocumentEmbedding:
    """
    Creates a new document embedding record.
    This is for manual insertion into *your* DocumentEmbedding table.
    Langchain's PGVector will manage its own table separately.
    """
    db_embedding = DocumentEmbedding(
        embedding_id=embedding_id,
        policy_document_id=policy_document_id,
        chunk_text=chunk_text,
        embedding_vector=embedding_vector, # Ensure this matches Vector(dimensions)
        metadata=metadata
    )
    db.add(db_embedding)
    await db.flush()
    return db_embedding

async def delete_embeddings_for_policy_document(db: AsyncSession, policy_document_id: int) -> int:
    """
    Deletes all embedding records associated with a specific policy_document_id
    from *your* DocumentEmbedding table. Returns the number of deleted records.
    """
    # This is a more direct way to delete if you are managing your own embedding table.
    # For Langchain's PGVector, you'd use its specific deletion methods.
    
    # Fetch IDs first to know what will be deleted (optional, for logging or return)
    stmt_select_ids = select(DocumentEmbedding.embedding_id).filter(DocumentEmbedding.policy_document_id == policy_document_id)
    result_ids = await db.execute(stmt_select_ids)
    ids_to_delete = result_ids.scalars().all()

    if not ids_to_delete:
        return 0

    # Perform delete
    # Note: SQLAlchemy 2.0 style for delete is slightly different for bulk operations.
    # For simplicity with ORM objects:
    count = 0
    for emb_id in ids_to_delete:
        emb_obj = await get_document_embedding_by_id(db, emb_id) # Fetch by ID
        if emb_obj:
            await db.delete(emb_obj)
            count +=1
    
    if count > 0:
        await db.flush()
    return count

# If Langchain's PGVector is used, it manages its own table (e.g., 'langchain_pg_embedding').
# You would interact with that store via Langchain's PGVector methods like `add_documents` and `delete`.
# The DocumentEmbedding model and these CRUDs are for if you choose to ALSO store/manage embeddings
# in a separate, application-defined table.
