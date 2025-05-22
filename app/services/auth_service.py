import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm # For form data login
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import timedelta # For token expiration

from app.db.database import get_db_session
from app.db.crud import user_crud, audit_log_crud, role_crud # Added role_crud for user creation role check
from app.models.db_models import SystemUser, ActionTypeEnum
from app.schemas.user_schemas import UserSchema, UserCreate
from app.schemas.token_schemas import Token
from app.schemas.audit_log_schemas import AuditLogCreate
from app.utils.security import verify_password, create_access_token
from app.config import settings

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, db: AsyncSession = Depends(get_db_session)): # Inject DB session
        self.db = db

    async def authenticate_user(self, username: str, password: str) -> Optional[SystemUser]:
        """
        Authenticates a user by username and password.
        Returns the user object if authentication is successful, None otherwise.
        """
        logger.debug(f"Attempting to authenticate user: {username}")
        user = await user_crud.get_user_by_username(self.db, username=username)
        if not user:
            logger.warning(f"Authentication failed: User '{username}' not found.")
            return None
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password for user '{username}'.")
            return None
        logger.info(f"User '{username}' authenticated successfully.")
        return user

    async def login_for_access_token(self, form_data: OAuth2PasswordRequestForm) -> Token:
        """
        Handles the login process and returns an access token.
        Uses OAuth2PasswordRequestForm for standard username/password form data.
        """
        user = await self.authenticate_user(username=form_data.username, password=form_data.password)
        if not user:
            # Audit log for failed login attempt (user might not exist or wrong password)
            # We don't know user_id here, so log with username if possible
            try:
                log_entry_fail = AuditLogCreate(
                    action_type=ActionTypeEnum.LOGIN,
                    details=f"Failed login attempt for username: {form_data.username}.",
                    access_granted=False
                )
                await audit_log_crud.create_audit_log(self.db, log_entry=log_entry_fail)
                # Commit is handled by get_db_session dependency
            except Exception as e_log:
                logger.error(f"Failed to create audit log for failed login: {e_log}")

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            logger.warning(f"Login attempt by inactive user: {user.username}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        # Data to be encoded in the token. "sub" (subject) is standard.
        # Include user_id for easier lookup from token, and role/region for quick access control checks.
        token_data = {
            "sub": user.username, 
            "user_id": user.user_id,
            "role": user.role.role_name if user.role else None, # Role name from eager loaded role
            "region": user.assigned_region
        }
        access_token = create_access_token(
            data=token_data, expires_delta=access_token_expires
        )
        
        # Audit Log for successful login
        try:
            log_entry_success = AuditLogCreate(
                user_id=user.user_id,
                action_type=ActionTypeEnum.LOGIN,
                details=f"User {user.username} logged in successfully.",
                user_role_context=user.role.role_name if user.role else "N/A",
                user_region_context=user.assigned_region,
                access_granted=True
            )
            await audit_log_crud.create_audit_log(self.db, log_entry=log_entry_success)
        except Exception as e_log:
            logger.error(f"Failed to create audit log for successful login: {e_log}")
            # Don't fail login if audit log fails, but log it.

        logger.info(f"Access token generated for user: {user.username}")
        return Token(access_token=access_token, token_type="bearer")

    async def get_current_user_details(self, current_user: SystemUser) -> UserSchema:
        """
        Returns the details of the currently authenticated user.
        The current_user object is injected by the get_current_active_user dependency.
        """
        logger.debug(f"Fetching details for current user: {current_user.username}")
        # The SystemUser ORM model (with its role eagerly loaded) is converted to UserSchema Pydantic model.
        return UserSchema.model_validate(current_user) # Pydantic V2

    async def create_new_user_service(self, user_in: UserCreate, performing_admin_id: Optional[int] = None) -> SystemUser:
        """
        Service method to create a new user.
        Typically called by an admin-protected endpoint.
        """
        logger.info(f"Attempting to create new user: {user_in.username}")
        db_user_by_username = await user_crud.get_user_by_username(self.db, username=user_in.username)
        if db_user_by_username:
            logger.warning(f"User creation failed: Username '{user_in.username}' already registered.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
        
        db_user_by_email = await user_crud.get_user_by_email(self.db, email=user_in.email)
        if db_user_by_email:
            logger.warning(f"User creation failed: Email '{user_in.email}' already registered.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
        
        # Check if role_id exists
        role = await role_crud.get_role(self.db, role_id=user_in.role_id)
        if not role:
            logger.warning(f"User creation failed: Role with id {user_in.role_id} not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Role with id {user_in.role_id} not found")

        created_user = await user_crud.create_user(db=self.db, user=user_in)
        logger.info(f"User '{created_user.username}' created successfully with ID: {created_user.user_id}")
        
        # Audit log for user creation
        try:
            log_details = f"User '{created_user.username}' (ID: {created_user.user_id}) created with role '{role.role_name}'."
            if performing_admin_id:
                log_details = f"Admin (ID: {performing_admin_id}) created {log_details}"
            
            log_entry = AuditLogCreate(
                user_id=performing_admin_id, # ID of admin performing action, or None if system/self-registration
                action_type=ActionTypeEnum.USER_CREATED,
                details=log_details,
                accessed_resource=f"user:{created_user.user_id}",
                user_role_context=role.role_name # Role of the created user
            )
            await audit_log_crud.create_audit_log(self.db, log_entry=log_entry)
        except Exception as e_log:
            logger.error(f"Failed to create audit log for user creation: {e_log}")

        # Fetch the user again to ensure the role relationship is loaded for the response schema
        user_with_role = await user_crud.get_user(self.db, user_id=created_user.user_id)
        if not user_with_role: # Should not happen if create_user was successful
            logger.error(f"Failed to fetch newly created user {created_user.user_id} with role details.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving created user details.")
        return user_with_role # This object will be validated by UserSchema
