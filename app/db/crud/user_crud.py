from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload # For eager loading role
from typing import List as PyList, Optional

from app.models.db_models import SystemUser
from app.schemas.user_schemas import UserCreate, UserUpdate
from app.utils.security import get_password_hash # Import the actual hasher

async def get_user(db: AsyncSession, user_id: int) -> Optional[SystemUser]:
    """Retrieve a single user by ID, eagerly loading their role."""
    result = await db.execute(
        select(SystemUser).options(selectinload(SystemUser.role).selectinload(Role.permissions)).filter(SystemUser.user_id == user_id)
    ) # Also load permissions of the role
    return result.scalars().first()

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[SystemUser]:
    """Retrieve a single user by username, eagerly loading their role and role's permissions."""
    # Importing Role here to avoid circular dependency at module level if Role model imports something from this file indirectly
    from app.models.db_models import Role 
    result = await db.execute(
        select(SystemUser)
        .options(selectinload(SystemUser.role).selectinload(Role.permissions)) # Eager load role and its permissions
        .filter(SystemUser.username == username)
    )
    return result.scalars().first()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[SystemUser]:
    """Retrieve a single user by email, eagerly loading their role and role's permissions."""
    from app.models.db_models import Role
    result = await db.execute(
        select(SystemUser)
        .options(selectinload(SystemUser.role).selectinload(Role.permissions))
        .filter(SystemUser.email == email)
    )
    return result.scalars().first()

async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> PyList[SystemUser]:
    """Retrieve a list of users, with pagination, eagerly loading their roles and role's permissions."""
    from app.models.db_models import Role
    result = await db.execute(
        select(SystemUser)
        .options(selectinload(SystemUser.role).selectinload(Role.permissions))
        .order_by(SystemUser.username)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def create_user(db: AsyncSession, user: UserCreate) -> SystemUser:
    """Create a new user."""
    hashed_password = get_password_hash(user.password)
    db_user = SystemUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role_id=user.role_id,
        assigned_region=user.assigned_region,
        is_active=user.is_active if user.is_active is not None else True
    )
    db.add(db_user)
    await db.flush()
    # To get the role object populated after creation for the return value:
    # await db.refresh(db_user, attribute_names=['role']) # If role relationship isn't auto-populated by flush
    # Or, fetch the user again with eager loading if needed immediately by caller.
    # For now, returning db_user as is. The role object might not be loaded yet.
    return db_user

async def update_user(db: AsyncSession, user_id: int, user_update_data: UserUpdate) -> Optional[SystemUser]:
    """Update an existing user's details. Password updates should be handled separately."""
    db_user = await get_user(db, user_id) # This already eager loads role.permissions
    if not db_user:
        return None
    
    update_data = user_update_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key == "password": 
            # Password should be updated via a dedicated secure mechanism, not this generic update.
            # e.g., a `change_password` function that takes old/new password.
            continue 
        setattr(db_user, key, value)
        
    db.add(db_user)
    await db.flush()
    # await db.refresh(db_user, attribute_names=['role']) # Refresh to ensure role object is current if role_id changed
    return db_user

async def delete_user(db: AsyncSession, user_id: int) -> Optional[SystemUser]:
    """
    Delete a user. 
    Consider implications: what happens to their audit logs? (user_id in AuditLog is nullable)
    For SCM data, if user_id was a creator/modifier, how is that handled? (Not in current SCM models)
    """
    db_user = await get_user(db, user_id)
    if not db_user:
        return None
    
    # Physical delete:
    await db.delete(db_user)
    await db.flush()
    return db_user # The object is now detached
