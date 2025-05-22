import logging
from typing import List as PyList, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.db.crud import policy_document_crud, audit_log_crud
from app.models.db_models import PolicyDocument, SystemUser, ActionTypeEnum
from app.schemas.policy_document_schemas import PolicyDocumentCreate, PolicyDocumentUpdate, PolicyDocumentSchema
from app.schemas.audit_log_schemas import AuditLogCreate

logger = logging.getLogger(__name__)

class PolicyDocumentService:
    def __init__(self, db: AsyncSession = Depends(get_db_session)):
        self.db = db

    async def create_document(self, doc_in: PolicyDocumentCreate, current_user: SystemUser) -> PolicyDocumentSchema:
        logger.info(f"User '{current_user.username}' attempting to create policy document: '{doc_in.title}'")
        existing_doc = await policy_document_crud.get_policy_document_by_title(self.db, title=doc_in.title)
        if existing_doc:
            logger.warning(f"Policy document creation failed: Title '{doc_in.title}' already exists.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Policy document with this title already exists")
        
        created_doc = await policy_document_crud.create_policy_document(self.db, doc=doc_in)
        logger.info(f"Policy document '{created_doc.title}' created successfully with ID: {created_doc.document_id}")

        await audit_log_crud.create_audit_log(self.db, AuditLogCreate(
            user_id=current_user.user_id, action_type=ActionTypeEnum.DOC_CREATED,
            details=f"User '{current_user.username}' created policy document: '{created_doc.title}' (ID: {created_doc.document_id})",
            accessed_resource=f"document:{created_doc.document_id}"
        ))
        return PolicyDocumentSchema.model_validate(created_doc)

    async def get_document(self, document_id: int) -> Optional[PolicyDocumentSchema]:
        logger.debug(f"Fetching policy document with ID: {document_id}")
        db_doc = await policy_document_crud.get_policy_document(self.db, document_id=document_id)
        if db_doc is None:
            logger.warning(f"Policy document with ID {document_id} not found.")
            return None
        return PolicyDocumentSchema.model_validate(db_doc)

    async def get_documents(self, skip: int = 0, limit: int = 100, document_type: Optional[str] = None, keyword_search: Optional[str] = None) -> PyList[PolicyDocumentSchema]:
        logger.debug(f"Fetching policy documents. Skip: {skip}, Limit: {limit}, Type: {document_type}, Keywords: {keyword_search}")
        db_docs = await policy_document_crud.get_policy_documents(self.db, skip=skip, limit=limit, document_type=document_type, keyword_search=keyword_search)
        return [PolicyDocumentSchema.model_validate(doc) for doc in db_docs]

    async def update_document(self, document_id: int, doc_in: PolicyDocumentUpdate, current_user: SystemUser) -> Optional[PolicyDocumentSchema]:
        logger.info(f"User '{current_user.username}' attempting to update policy document ID: {document_id}")
        updated_doc = await policy_document_crud.update_policy_document(self.db, document_id=document_id, doc_update_data=doc_in)
        if updated_doc is None:
            logger.warning(f"Policy document update failed: Document ID {document_id} not found.")
            return None
        
        logger.info(f"Policy document ID {document_id} updated successfully.")
        await audit_log_crud.create_audit_log(self.db, AuditLogCreate(
            user_id=current_user.user_id, action_type=ActionTypeEnum.DOC_UPDATED,
            details=f"User '{current_user.username}' updated policy document: '{updated_doc.title}' (ID: {document_id}). Changes: {doc_in.model_dump(exclude_unset=True)}",
            accessed_resource=f"document:{document_id}"
        ))
        return PolicyDocumentSchema.model_validate(updated_doc)

    async def delete_document(self, document_id: int, current_user: SystemUser) -> Optional[PolicyDocumentSchema]:
        logger.info(f"User '{current_user.username}' attempting to delete policy document ID: {document_id}")
        deleted_doc = await policy_document_crud.delete_policy_document(self.db, document_id=document_id)
        if deleted_doc is None:
            logger.warning(f"Policy document deletion failed: Document ID {document_id} not found.")
            return None
            
        logger.info(f"Policy document '{deleted_doc.title}' (ID: {document_id}) deleted successfully.")
        await audit_log_crud.create_audit_log(self.db, AuditLogCreate(
            user_id=current_user.user_id, action_type=ActionTypeEnum.DOC_DELETED,
            details=f"User '{current_user.username}' deleted policy document: '{deleted_doc.title}' (ID: {document_id})",
            accessed_resource=f"document:{document_id}"
        ))
        return PolicyDocumentSchema.model_validate(deleted_doc)
