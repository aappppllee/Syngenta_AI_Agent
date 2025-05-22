from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List as PyList, Optional, Dict, Any # Added Dict, Any

from app.models.db_models import DocumentEmbedding
# from app.schemas.embedding_schemas import DocumentEmbeddingCreate # Usually not created via API like this

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
    limit: int = 1000 
) -> PyList[DocumentEmbedding]:
    """Retrieve all embedding records associated with a specific policy document ID."""
    result = await db.execute(
        select(DocumentEmbedding)
        .filter(DocumentEmbedding.policy_document_id == policy_document_id)
        .order_by(DocumentEmbedding.created_at) 
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
    # The 'metadata_payload' parameter name is kept generic for the function signature.
    # It will be assigned to the ORM's 'embedding_metadata' field.
    metadata_payload: Optional[Dict[str, Any]] = None # Changed from Optional[str] to Optional[Dict[str, Any]] to match JSONB
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
        # embedding_vector=embedding_vector, # Assuming vector is not stored in this ORM table
        embedding_metadata=metadata_payload # Assign to the renamed ORM field
    )
    db.add(db_embedding)
    await db.flush()
    return db_embedding

async def delete_embeddings_for_policy_document(db: AsyncSession, policy_document_id: int) -> int:
    """
    Deletes all embedding records associated with a specific policy_document_id
    from *your* DocumentEmbedding table. Returns the number of deleted records.
    """
    stmt_select_ids = select(DocumentEmbedding.embedding_id).filter(DocumentEmbedding.policy_document_id == policy_document_id)
    result_ids = await db.execute(stmt_select_ids)
    ids_to_delete = result_ids.scalars().all()

    if not ids_to_delete:
        return 0
    
    count = 0
    # Instead of fetching one by one, we can construct a delete statement
    # However, to return the count of actual ORM objects, this is one way.
    # A more efficient delete would be:
    # from sqlalchemy import delete
    # stmt_delete = delete(DocumentEmbedding).where(DocumentEmbedding.policy_document_id == policy_document_id)
    # result = await db.execute(stmt_delete)
    # await db.flush()
    # return result.rowcount

    # Current approach (less efficient for bulk but works):
    for emb_id_uuid in ids_to_delete: # Assuming embedding_id is UUID
        # Need to query by UUID if embedding_id is PG_UUID(as_uuid=True)
        emb_obj = await db.get(DocumentEmbedding, emb_id_uuid) # Use db.get for PK lookup
        if emb_obj:
            await db.delete(emb_obj)
            count +=1
    
    if count > 0:
        await db.flush()
    return count

