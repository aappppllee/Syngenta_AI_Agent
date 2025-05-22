from typing import Optional, List as PyList
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from app.schemas.role_schemas import RoleSchema # Import RoleSchema

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_]+$", description="Username, alphanumeric and underscores only.")
    email: EmailStr = Field(..., description="User's email address.")
    assigned_region: str = Field("Global", max_length=100, description="Geographic region assigned to the user.")
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="User's password.")
    role_id: int = Field(..., description="ID of the role assigned to the user.")

class UserUpdate(BaseModel): # For updating user details by admin
    username: Optional[str] = Field(None, min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_]+$")
    email: Optional[EmailStr] = None
    assigned_region: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    role_id: Optional[int] = None
    # Password update should be a separate endpoint/process for security by user themself or admin reset

class UserInDBBase(UserBase):
    user_id: int
    role_id: int # Keep role_id for direct reference
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserSchema(UserInDBBase): # Schema for returning user data, including full role details
    role: RoleSchema # Embed full role information

# Schema for internal use, e.g., when fetching user from DB by auth service
class UserWithPasswordSchema(UserInDBBase):
    hashed_password: str
