from typing import Optional, List as PyList
from pydantic import BaseModel, Field

# Forward declaration for SCMPermissionSchema to handle potential circular imports
# This is one way; another is to define SCMPermissionSchema first or use UpdateForwardRefs.
# For simplicity, we'll assume SCMPermissionSchema is defined elsewhere and will be resolved.
# Alternatively, if RoleWithPermissionsSchema is the primary way to show roles with permissions,
# the basic RoleSchema can remain simple.

class RoleBase(BaseModel):
    role_name: str = Field(..., min_length=3, max_length=100, description="Name of the role")
    description: Optional[str] = Field(None, max_length=500, description="Description of the role")

class RoleCreate(RoleBase):
    pass

class RoleUpdate(RoleBase):
    role_name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class RoleInDBBase(RoleBase):
    role_id: int
    
    class Config:
        from_attributes = True

class RoleSchema(RoleInDBBase):
    """Basic schema for returning role data without nested permissions by default."""
    pass

# If you need a schema that explicitly includes permissions when returning a Role,
# you'd define it here or in permission_schemas.py and use it in specific endpoints.
# Example (requires SCMPermissionSchema to be defined and importable):
# from app.schemas.permission_schemas import SCMPermissionSchema
# class RoleWithPermissionsSchema(RoleInDBBase):
#     permissions: PyList[SCMPermissionSchema] = []
