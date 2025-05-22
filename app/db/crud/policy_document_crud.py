from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List as PyList, Optional # Use PyList

from app.models.db_models import PolicyDocument
from app.schemas.policy_document_schemas import PolicyDocumentCreate, PolicyDocumentUpdate

async def get_policy_document(db: AsyncSession, document_id: int) -> Optional[PolicyDocument]:
    """Retrieve a single policy document by its ID."""
    result = await db.execute(
        select(PolicyDocument).filter(PolicyDocument.document_id == document_id)
    )
    return result.scalars().first()

async def get_policy_document_by_title(db: AsyncSession, title: str) -> Optional[PolicyDocument]:
    """Retrieve a single policy document by its title."""
    result = await db.execute(
        select(PolicyDocument).filter(PolicyDocument.title == title)
    )
    return result.scalars().first()

async def get_policy_documents(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100, 
    document_type: Optional[str] = None,
    keyword_search: Optional[str] = None # Simple keyword search on title/summary
) -> PyList[PolicyDocument]:
    """
    Retrieve a list of policy documents, with optional pagination, type filtering,
    and basic keyword search on title or summary.
    """
    query = select(PolicyDocument).order_by(PolicyDocument.title)
    
    if document_type:
        query = query.filter(PolicyDocument.document_type == document_type)
    
    if keyword_search:
        search_term = f"%{keyword_search}%"
        query = query.filter(
            (PolicyDocument.title.ilike(search_term)) | 
            (PolicyDocument.summary.ilike(search_term)) |
            (PolicyDocument.keywords.ilike(search_term)) # If keywords is a text field
        )
        
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_policy_document(db: AsyncSession, doc: PolicyDocumentCreate) -> PolicyDocument:
    """Create a new policy document record in the database."""
    db_doc = PolicyDocument(**doc.model_dump())
    db.add(db_doc)
    await db.flush() # To get the document_id if needed before commit
    # await db.refresh(db_doc) # To get DB defaults like created_at, updated_at
    return db_doc

async def update_policy_document(
    db: AsyncSession, 
    document_id: int, 
    doc_update_data: PolicyDocumentUpdate
) -> Optional[PolicyDocument]:
    """Update an existing policy document."""
    db_doc = await get_policy_document(db, document_id)
    if not db_doc:
        return None
    
    update_data = doc_update_data.model_dump(exclude_unset=True) # Get only provided fields
    for key, value in update_data.items():
        setattr(db_doc, key, value)
    
    db.add(db_doc) # Add updated object to session
    await db.flush()
    # await db.refresh(db_doc)
    return db_doc

async def delete_policy_document(db: AsyncSession, document_id: int) -> Optional[PolicyDocument]:
    """
    Delete a policy document from the database.
    This will also delete associated embeddings if cascade is set up correctly on the relationship.
    """
    db_doc = await get_policy_document(db, document_id)
    if not db_doc:
        return None
    
    # If DocumentEmbedding.policy_document relationship has cascade="all, delete-orphan",
    # deleting db_doc should automatically delete its embeddings.
    await db.delete(db_doc)
    await db.flush()
    return db_doc # Returns the object before it's fully expunged from session post-commit
