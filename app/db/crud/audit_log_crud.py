from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List as PyList, Optional # Use PyList to avoid conflict

from app.models.db_models import AuditLog, ActionTypeEnum # ActionTypeEnum might be used for filtering if needed
from app.schemas.audit_log_schemas import AuditLogCreate

async def create_audit_log(db: AsyncSession, log_entry: AuditLogCreate) -> AuditLog:
    """
    Creates a new audit log entry in the database.
    """
    db_log = AuditLog(
        user_id=log_entry.user_id,
        action_type=log_entry.action_type,
        query_text=log_entry.query_text,
        accessed_resource=log_entry.accessed_resource,
        access_granted=log_entry.access_granted,
        details=log_entry.details,
        user_role_context=log_entry.user_role_context,
        user_region_context=log_entry.user_region_context
    )
    db.add(db_log)
    await db.flush() # To get log_id if needed before commit
    # await db.refresh(db_log) # To get DB defaults like timestamp
    return db_log

async def get_audit_log_by_id(db: AsyncSession, log_id: int) -> Optional[AuditLog]:
    """
    Retrieves a single audit log entry by its ID.
    """
    result = await db.execute(select(AuditLog).filter(AuditLog.log_id == log_id))
    return result.scalars().first()

async def get_audit_logs(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100, 
    user_id: Optional[int] = None,
    action_type: Optional[ActionTypeEnum] = None
) -> PyList[AuditLog]:
    """
    Retrieves a list of audit log entries, with optional pagination and filtering.
    Logs are returned in descending order of timestamp (most recent first).
    """
    query = select(AuditLog).order_by(AuditLog.timestamp.desc())
    
    if user_id is not None:
        query = query.filter(AuditLog.user_id == user_id)
    
    if action_type is not None:
        query = query.filter(AuditLog.action_type == action_type)
        
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

# No update or delete operations are typically provided for audit logs
# to maintain integrity, unless specific archival/GDPR requirements exist.
