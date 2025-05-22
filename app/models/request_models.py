from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """
    Pydantic model for the request body of the /query endpoint.
    """
    query_text: str = Field(..., min_length=1, description="The natural language query from the user.")
    session_id: Optional[str] = Field(None, description="Optional session ID for maintaining conversational context. If None, a new session may be started by the server.")
    # Example of how you might pass additional context if needed, though user context is usually derived from auth token.
    # query_context_override: Optional[Dict[str, Any]] = Field(None, description="Optional advanced context to override or supplement agent's understanding.")

    class Config:
        json_schema_extra = { # Changed from 'example' to 'json_schema_extra' for Pydantic V2 compatibility for examples
            "example": {
                "query_text": "What is our company policy on inventory write-offs?",
                "session_id": "session_abc123" # Client should send this back for follow-ups
            }
        }

# You can add other request models here if needed for other endpoints,
# for example, for specific admin actions if they take complex bodies.
