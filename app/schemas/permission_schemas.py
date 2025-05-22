from typing import Optional, List as PyList
from pydantic import BaseModel, Field
from app.schemas.role_schemas import RoleSchema as BaseRoleSchema # Import base RoleSchema

class SCMPermissionBase(BaseModel):
    permission_name: str = Field(..., min_length=3, max_length=255, description="Unique name for the permission (e.g., 'view_financial_reports', 'query:table:scm_orders')")
    description: Optional[str] = Field(None, max_length=500, description="Detailed description of what the permission allows.")
    category: Optional[str] = Field(None, max_length=100, description="Category for grouping permissions (e.g., 'Financial', 'Inventory', 'Document').")

class SCMPermissionCreate(SCMPermissionBase):
    pass

class SCMPermissionUpdate(BaseModel): # Allow partial updates
    permission_name: Optional[str] = Field(None, min_length=3, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)

class SCMPermissionSchema(SCMPermissionBase):
    permission_id: int
    
    class Config:
        from_attributes = True

# Schemas for Role-Permission assignments (used in admin router)
class RolePermissionAssignment(BaseModel):
    role_id: int
    permission_id: int

# Schema for returning a Role with its permissions
class RoleWithPermissionsSchema(BaseRoleSchema): # Extends the basic RoleSchema
    permissions: PyList[SCMPermissionSchema] = Field(default_factory=list)
