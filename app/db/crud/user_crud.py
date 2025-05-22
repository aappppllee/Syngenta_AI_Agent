from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload 
from typing import List as PyList, Optional

from app.models.db_models import SystemUser, Role # Import Role
from app.schemas.user_schemas import UserCreate, UserUpdate
from app.utils.security import get_password_hash 

async def get_user(db: AsyncSession, user_id: int) -> Optional[SystemUser]:
    """Retrieve a single user by ID, eagerly loading their role and the role's permissions."""
    result = await db.execute(
        select(SystemUser)
        .options(
            selectinload(SystemUser.role).selectinload(Role.permissions) # Role.permissions requires Role to be defined
        )
        .filter(SystemUser.user_id == user_id)
    ) 
    return result.scalars().first()

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[SystemUser]:
    """Retrieve a single user by username, eagerly loading their role and role's permissions."""
    result = await db.execute(
        select(SystemUser)
        .options(selectinload(SystemUser.role).selectinload(Role.permissions)) 
        .filter(SystemUser.username == username)
    )
    return result.scalars().first()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[SystemUser]:
    """Retrieve a single user by email, eagerly loading their role and role's permissions."""
    result = await db.execute(
        select(SystemUser)
        .options(selectinload(SystemUser.role).selectinload(Role.permissions))
        .filter(SystemUser.email == email)
    )
    return result.scalars().first()

async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> PyList[SystemUser]:
    """Retrieve a list of users, with pagination, eagerly loading their roles and role's permissions."""
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
    await db.refresh(db_user, attribute_names=['role']) # Refresh to load the role relationship
    return db_user

async def update_user(db: AsyncSession, user_id: int, user_update_data: UserUpdate) -> Optional[SystemUser]:
    """Update an existing user's details. Password updates should be handled separately."""
    db_user = await get_user(db, user_id) 
    if not db_user:
        return None
    
    update_data = user_update_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key == "password": 
            continue 
        setattr(db_user, key, value)
        
    db.add(db_user)
    await db.flush()
    await db.refresh(db_user, attribute_names=['role']) # Refresh to ensure role is current if role_id changed
    return db_user

async def delete_user(db: AsyncSession, user_id: int) -> Optional[SystemUser]:
    """
    Delete a user. 
    """
    db_user = await get_user(db, user_id)
    if not db_user:
        return None
    
    await db.delete(db_user)
    await db.flush()
    return db_user

