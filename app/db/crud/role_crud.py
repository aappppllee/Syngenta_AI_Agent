from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload # For eager loading permissions
from typing import List as PyList, Optional

from app.models.db_models import Role
from app.schemas.role_schemas import RoleCreate, RoleUpdate

async def get_role(db: AsyncSession, role_id: int) -> Optional[Role]:
    """Retrieve a single role by its ID, eagerly loading its permissions."""
    result = await db.execute(
        select(Role).options(selectinload(Role.permissions)).filter(Role.role_id == role_id)
    )
    return result.scalars().first()

async def get_role_by_name(db: AsyncSession, name: str) -> Optional[Role]:
    """Retrieve a single role by its name, eagerly loading its permissions."""
    result = await db.execute(
        select(Role).options(selectinload(Role.permissions)).filter(Role.role_name == name)
    )
    return result.scalars().first()

async def get_roles(db: AsyncSession, skip: int = 0, limit: int = 100) -> PyList[Role]:
    """Retrieve a list of roles, with optional pagination, eagerly loading permissions."""
    result = await db.execute(
        select(Role).options(selectinload(Role.permissions)).order_by(Role.role_name).offset(skip).limit(limit)
    )
    return result.scalars().all()

async def create_role(db: AsyncSession, role: RoleCreate) -> Role:
    """Create a new role. Permissions are assigned separately."""
    db_role = Role(role_name=role.role_name, description=role.description)
    db.add(db_role)
    await db.flush()
    # await db.refresh(db_role)
    return db_role

async def update_role(db: AsyncSession, role_id: int, role_update_data: RoleUpdate) -> Optional[Role]:
    """Update an existing role's details (name, description). Permissions are managed separately."""
    db_role = await get_role(db, role_id) # This will load permissions too, though not modified here
    if not db_role:
        return None
    
    update_data = role_update_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_role, key, value)
    
    db.add(db_role)
    await db.flush()
    # await db.refresh(db_role)
    return db_role

async def delete_role(db: AsyncSession, role_id: int) -> Optional[Role]:
    """
    Delete a role. 
    This will also attempt to remove its associations from scm_role_permissions table
    if Role.permissions relationship is configured with cascade="all, delete-orphan" on the secondary,
    or if handled by DB constraints. For safety, explicitly clearing is better if unsure.
    Also, ensure no users are currently assigned this role before deletion in a production system.
    """
    db_role = await get_role(db, role_id) # Eager loads permissions
    if not db_role:
        return None

    # Explicitly clear the many-to-many relationship before deleting the role instance
    # This ensures entries in the association table (scm_role_permissions_table) are removed.
    if db_role.permissions:
        db_role.permissions.clear()
        await db.flush() # Persist the clearing of the association

    await db.delete(db_role)
    await db.flush()
    return db_role
