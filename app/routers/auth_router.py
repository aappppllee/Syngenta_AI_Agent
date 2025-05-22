import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional 

from app.services.auth_service import AuthService
from app.schemas.token_schemas import Token
from app.schemas.user_schemas import UserSchema, UserCreate 
from app.models.db_models import SystemUser
from app.dependencies import get_current_active_user, get_db_session
from app.config import settings

logger = logging.getLogger(__name__)

router_auth = APIRouter() 

@router_auth.post("/login", response_model=Token, summary="User Login")
async def login_for_access_token_endpoint(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: AsyncSession = Depends(get_db_session)
):
    """
    Login endpoint for existing users.
    Takes username and password from form data (x-www-form-urlencoded).
    Returns an access token upon successful authentication.
    """
    logger.info(f"Login attempt for username: {form_data.username}")
    auth_service = AuthService(db=db) 
    try:
        token = await auth_service.login_for_access_token(form_data=form_data)
        return token
    except HTTPException as http_exc: 
        logger.warning(f"Login failed for {form_data.username}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during login for {form_data.username}: {e}", exc_info=settings.DEBUG_MODE)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during login."
        )

@router_auth.get("/me", response_model=UserSchema, summary="Get Current User Details")
async def read_users_me(
    current_user: SystemUser = Depends(get_current_active_user), 
    db: AsyncSession = Depends(get_db_session) 
):
    """
    Endpoint to get the details of the currently authenticated and active user.
    """
    logger.info(f"Fetching details for current user: {current_user.username}")
    auth_service = AuthService(db=db)
    try:
        user_details = await auth_service.get_current_user_details(current_user=current_user)
        return user_details
    except Exception as e:
        logger.error(f"Error fetching details for user {current_user.username}: {e}", exc_info=settings.DEBUG_MODE)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch user details."
        )

@router_auth.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED, summary="Register New User (Public)", tags=["Users"])
async def register_new_user(
    user_in: UserCreate, 
    db: AsyncSession = Depends(get_db_session),
    # This is a public registration endpoint. For admin creation, use /admin/users/create
):
    """
    Register a new user. This endpoint is public.
    For admin-controlled user creation, use the endpoint under /admin/users.
    """
    logger.info(f"Public registration attempt for username: {user_in.username}")
    auth_service = AuthService(db=db)
    try:
        # For public registration, performing_admin_id is None.
        created_user_orm = await auth_service.create_new_user_service(user_in=user_in, performing_admin_id=None)
        return UserSchema.model_validate(created_user_orm)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error during public user registration for {user_in.username}: {e}", exc_info=settings.DEBUG_MODE)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during user registration."
        )

