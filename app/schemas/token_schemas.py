from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    """Pydantic model for the JWT access token response."""
    access_token: str
    token_type: str # Typically "bearer"

class TokenData(BaseModel):
    """Pydantic model for data extracted from a JWT token (payload)."""
    username: Optional[str] = None # Corresponds to 'sub' claim usually
    user_id: Optional[int] = None 
    # Add other fields you might store in the token, like roles or scopes
    # For example:
    # role: Optional[str] = None 
