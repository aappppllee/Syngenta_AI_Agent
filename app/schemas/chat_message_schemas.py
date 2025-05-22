from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List as PyList
from datetime import datetime
import json # For parsing the message blob

# These schemas represent the structure of how Langchain's SQLChatMessageHistory
# stores messages as JSON blobs in the 'message' column of SCMChatMessage table.

class ChatMessageStoredData(BaseModel):
    """Represents the 'data' part of the JSON in SCMChatMessage.message."""
    content: str
    additional_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatMessageStored(BaseModel):
    """Represents the full JSON structure in SCMChatMessage.message."""
    type: str # "human", "ai", "system", etc.
    data: ChatMessageStoredData

# Schema for API responses if you were to list messages directly from SCMChatMessage table
class ChatMessageSchema(BaseModel):
    id: int
    session_id: str
    message_content: ChatMessageStored # Parsed JSON content from the 'message' field
    created_at: datetime

    @classmethod
    def from_orm_with_parsed_message(cls, db_message: Any) -> "ChatMessageSchema":
        """
        Custom constructor to parse the JSON 'message' field from the ORM object.
        'db_message' is expected to be an instance of SCMChatMessage ORM model.
        """
        message_blob_str = db_message.message
        parsed_message_content = {"type": "unknown", "data": {"content": "Error parsing message"}}
        try:
            if isinstance(message_blob_str, str):
                parsed_json = json.loads(message_blob_str)
                # Validate against ChatMessageStored structure
                parsed_message_content = ChatMessageStored(**parsed_json).model_dump()
            elif isinstance(message_blob_str, dict): # If already parsed (e.g. by SQLAlchemy JSON type)
                 parsed_message_content = ChatMessageStored(**message_blob_str).model_dump()

        except (json.JSONDecodeError, TypeError, Exception) as e:
            # Log error or handle as appropriate
            print(f"Error parsing SCMChatMessage.message JSON: {e}. Blob: {message_blob_str}")
            # Fallback or raise error

        return cls(
            id=db_message.id,
            session_id=db_message.session_id,
            message_content=parsed_message_content, # This should be a dict now
            created_at=db_message.created_at
        )

    class Config:
        from_attributes = True
