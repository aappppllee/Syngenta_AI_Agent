from typing import Optional, Any, List as PyList
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.db_models import ActionTypeEnum # Import the enum

class AuditLogBase(BaseModel):
    action_type: ActionTypeEnum
    query_text: Optional[str] = None
    accessed_resource: Optional[str] = None
    access_granted: Optional[bool] = None
    details: Optional[str] = None
    user_role_context: Optional[str] = None
    user_region_context: Optional[str] = None

class AuditLogCreate(AuditLogBase):
    user_id: Optional[int] = None # Can be null for system actions

class AuditLogInDBBase(AuditLogBase):
    log_id: int
    user_id: Optional[int]
    timestamp: datetime

    class Config:
        from_attributes = True

class AuditLogSchema(AuditLogInDBBase): # For API responses
    # Optionally, include related user details if needed, e.g., username
    # user: Optional[UserSchema] = None # Would require UserSchema import and relationship loading
    pass
