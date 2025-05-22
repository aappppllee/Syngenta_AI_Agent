import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError # Already imported in security, but good for clarity here

from app.db.database import get_db_session
from app.db.crud import user_crud # user_crud is a module
from app.models.db_models import SystemUser # For type hinting
from app.schemas.token_schemas import TokenData
from app.utils.security import decode_access_token
from app.config import settings

logger = logging.getLogger(__name__)

# OAuth2PasswordBearer scheme. 
# tokenUrl should point to your login endpoint.
# Changed to be an absolute path from the API root.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login") 

async def get_current_user_from_token(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db_session)
) -> SystemUser:
    """
    Dependency to get the current user from a JWT token.
    Decodes the token, validates its claims, and fetches the user from the database.
    Eagerly loads the user's role and the role's permissions.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None: # Token is invalid (e.g., malformed, wrong signature) or expired
        logger.warning("Token decoding failed or token expired.")
        raise credentials_exception
    
    user_id_from_token: Optional[int] = payload.get("user_id")
    username_from_token: Optional[str] = payload.get("sub") # Subject, often username

    if user_id_from_token is None and username_from_token is None:
        logger.warning("Token payload missing 'user_id' and 'sub' (username).")
        raise credentials_exception
    
    user: Optional[SystemUser] = None
    if user_id_from_token is not None:
        logger.debug(f"Attempting to fetch user by ID: {user_id_from_token} from token.")
        user = await user_crud.get_user(db, user_id=user_id_from_token) 
    elif username_from_token is not None: 
        logger.debug(f"Attempting to fetch user by username: {username_from_token} from token 'sub'.")
        user = await user_crud.get_user_by_username(db, username=username_from_token) 
        
    if user is None:
        logger.warning(f"User not found in DB for token payload (user_id: {user_id_from_token}, username: {username_from_token}).")
        raise credentials_exception
    
    if settings.DEBUG_MODE:
        if user.role:
            logger.debug(f"User {user.username} has role: {user.role.role_name} with {len(user.role.permissions)} permissions.")
        else:
            logger.warning(f"User {user.username} fetched but role object is None.")

    return user

async def get_current_active_user(
    current_user: SystemUser = Depends(get_current_user_from_token)
) -> SystemUser:
    """
    Dependency to get the current active user.
    Builds on get_current_user_from_token and checks if the user's 'is_active' flag is true.
    """
    if not current_user.is_active:
        logger.warning(f"Authentication attempt by inactive user: {current_user.username}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    logger.debug(f"Authenticated active user: {current_user.username} (Role: {current_user.role.role_name if current_user.role else 'N/A'})")
    return current_user

async def get_current_admin_user(
    current_user: SystemUser = Depends(get_current_active_user)
) -> SystemUser:
    """
    Dependency to ensure the current user is an active admin.
    Relies on the user's role name being settings.ADMIN_ROLE_NAME.
    """
    if not current_user.role or current_user.role.role_name != settings.ADMIN_ROLE_NAME:
        logger.warning(f"Admin access denied for user: {current_user.username} (Role: {current_user.role.role_name if current_user.role else 'N/A'})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have administrator privileges"
        )
    logger.debug(f"Admin access GRANTED for user: {current_user.username}")
    return current_user

