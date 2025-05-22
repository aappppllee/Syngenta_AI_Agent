from typing import Optional, List as PyList
from pydantic import BaseModel, Field
from datetime import datetime

class PolicyDocumentBase(BaseModel):
    title: str = Field(..., min_length=3, max_length=255, description="Title of the policy document.")
    original_file_name: Optional[str] = Field(None, max_length=255, description="Original name of the uploaded file.")
    storage_path: Optional[str] = Field(None, max_length=500, description="Path where the document content is stored.")
    document_type: Optional[str] = Field("Policy", max_length=100, description="Type of the document (e.g., Policy, Procedure, Guideline).")
    summary: Optional[str] = Field(None, description="A brief summary of the document's content.")
    keywords: Optional[str] = Field(None, description="Comma-separated keywords or a JSON string of keywords for searchability.")

class PolicyDocumentCreate(PolicyDocumentBase):
    pass

class PolicyDocumentUpdate(BaseModel): # Allow partial updates
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    original_file_name: Optional[str] = Field(None, max_length=255)
    storage_path: Optional[str] = Field(None, max_length=500)
    document_type: Optional[str] = Field(None, max_length=100)
    summary: Optional[str] = None
    keywords: Optional[str] = None

class PolicyDocumentInDBBase(PolicyDocumentBase):
    document_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PolicyDocumentSchema(PolicyDocumentInDBBase): # Schema for returning document data
    pass
