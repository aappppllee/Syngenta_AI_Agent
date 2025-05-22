import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any 

from app.services.query_service import QueryService
from app.models.request_models import QueryRequest
from app.models.response_models import QueryResponse
from app.dependencies import get_current_active_user 
from app.models.db_models import SystemUser 
from app.config import settings 

logger = logging.getLogger(__name__)

router_query = APIRouter()

# The QueryService instance is created in main.py and intended to be attached 
# to this router object (e.g., router_query._query_service_instance = query_service_instance_from_main).
# This is a simplified DI approach for the skeleton. A more robust method would use
# FastAPI's `Depends` for the service itself, requiring AgentCore to also be injectable.

@router_query.post(
    "/", 
    response_model=QueryResponse, 
    status_code=status.HTTP_200_OK,
    summary="Submit a Natural Language Query to the Agent"
)
async def handle_agent_query(
    request: QueryRequest, 
    current_user: SystemUser = Depends(get_current_active_user), # Enforces authentication
):
    """
    Endpoint to submit a natural language query to the intelligent agent.
    Requires authentication. The agent will process the query based on the user's
    permissions, context, and conversation history (if a session_id is provided).
    """
    # Accessing the service instance (assuming it's attached in main.py)
    if not hasattr(router_query, "_query_service_instance") or router_query._query_service_instance is None:
        logger.critical("QueryService instance not found on router_query. This indicates a setup error in main.py during startup.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query service is not configured or available. Please contact support."
        )
    
    query_service_instance: QueryService = router_query._query_service_instance
    
    logger.info(f"QueryRouter: Received query '{request.query_text}' from user '{current_user.username}', session '{request.session_id}'.")
    try:
        response = await query_service_instance.process_query(request, current_user)
        return response
    except HTTPException as http_exc: 
        logger.warning(f"HTTPException while processing query for user '{current_user.username}': {http_exc.detail}", exc_info=settings.DEBUG_MODE if http_exc.status_code >= 500 else False)
        raise http_exc 
    except Exception as e:
        logger.error(f"Unexpected error in QueryRouter handling agent query for user '{current_user.username}': {e}", exc_info=settings.DEBUG_MODE)
        # This will be caught by the global exception handler in main.py, 
        # but raising a specific HTTPException here can be more informative if desired.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query. Please try again later."
        )

