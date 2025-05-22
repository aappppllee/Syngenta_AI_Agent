from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List as PyList, Optional
import json # For handling the message JSON blob

from app.models.db_models import SCMChatMessage
from app.schemas.chat_message_schemas import ChatMessageStored # For creating the message blob

# Note: Langchain's SQLChatMessageHistory handles its own CRUD operations
# (add_messages, get_messages, clear) directly with the database table it manages.
# These CRUD functions below are for interacting with the SCMChatMessage ORM model
# IF you need to inspect or manually manipulate the chat history table outside of
# the Langchain memory object's direct operations.

async def get_chat_messages_by_session_crud(
    db: AsyncSession, 
    session_id: str, 
    skip: int = 0, 
    limit: int = 100
) -> PyList[SCMChatMessage]:
    """
    Retrieve SCMChatMessage ORM objects for a given session ID, ordered by creation time.
    """
    result = await db.execute(
        select(SCMChatMessage)
        .filter(SCMChatMessage.session_id == session_id)
        .order_by(SCMChatMessage.created_at.asc()) # Ascending for chronological order
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def add_chat_message_crud(
    db: AsyncSession, 
    session_id: str, 
    message_type: str, # "human", "ai", "system"
    content: str,
    additional_kwargs: Optional[dict] = None
) -> SCMChatMessage:
    """
    Manually adds a chat message to the SCMChatMessage table.
    Typically, Langchain's SQLChatMessageHistory.add_messages (or equivalents like add_user_message)
    would be used to populate this table. This function is for direct insertion if needed.
    """
    # Construct the JSON blob for the 'message' field as Langchain's SQLChatMessageHistory does
    message_data = {
        "type": message_type,
        "data": {
            "content": content,
            "additional_kwargs": additional_kwargs or {}
        }
    }
    message_json_blob = json.dumps(message_data)

    db_message = SCMChatMessage(
        session_id=session_id,
        message=message_json_blob
        # created_at is server_default
    )
    db.add(db_message)
    await db.flush() # To get id and created_at if needed before commit
    # await db.refresh(db_message)
    return db_message

async def clear_chat_messages_for_session_crud(db: AsyncSession, session_id: str) -> int:
    """
    Manually clears all chat messages for a given session_id from SCMChatMessage table.
    Returns the number of messages deleted.
    Langchain's SQLChatMessageHistory().clear() method should be preferred.
    """
    # A more direct delete without fetching first:
    # stmt = delete(SCMChatMessage).where(SCMChatMessage.session_id == session_id)
    # result = await db.execute(stmt)
    # return result.rowcount
    
    # Fetch and delete (less efficient for many messages but shows objects)
    messages_to_delete = await get_chat_messages_by_session_crud(db, session_id, limit=10000) # High limit
    count = 0
    if messages_to_delete:
        for msg in messages_to_delete:
            await db.delete(msg)
            count += 1
        await db.flush()
    return count
