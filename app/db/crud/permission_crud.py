from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload # To eager load related roles/permissions
from typing import List as PyList, Optional

from app.models.db_models import SCMPermission, Role, scm_role_permissions_table # Import association table
from app.schemas.permission_schemas import SCMPermissionCreate, SCMPermissionUpdate

async def get_permission(db: AsyncSession, permission_id: int) -> Optional[SCMPermission]:
    """Retrieve a single permission by its ID."""
    result = await db.execute(
        select(SCMPermission).filter(SCMPermission.permission_id == permission_id)
    )
    return result.scalars().first()

async def get_permission_by_name(db: AsyncSession, permission_name: str) -> Optional[SCMPermission]:
    """Retrieve a single permission by its unique name."""
    result = await db.execute(
        select(SCMPermission).filter(SCMPermission.permission_name == permission_name)
    )
    return result.scalars().first()

async def get_permissions(db: AsyncSession, skip: int = 0, limit: int = 100) -> PyList[SCMPermission]:
    """Retrieve a list of permissions, with optional pagination."""
    result = await db.execute(select(SCMPermission).order_by(SCMPermission.category, SCMPermission.permission_name).offset(skip).limit(limit))
    return result.scalars().all()

async def create_permission(db: AsyncSession, permission: SCMPermissionCreate) -> SCMPermission:
    """Create a new permission record in the database."""
    db_permission = SCMPermission(**permission.model_dump())
    db.add(db_permission)
    await db.flush() # To get permission_id if needed before commit
    # await db.refresh(db_permission) # To get DB defaults like created_at, if not defined in model
    return db_permission

async def update_permission(db: AsyncSession, permission_id: int, permission_update: SCMPermissionUpdate) -> Optional[SCMPermission]:
    """Update an existing permission."""
    db_permission = await get_permission(db, permission_id)
    if not db_permission:
        return None
    
    update_data = permission_update.model_dump(exclude_unset=True) # Get only provided fields
    for key, value in update_data.items():
        setattr(db_permission, key, value)
    
    db.add(db_permission) # Add updated object to session
    await db.flush()
    # await db.refresh(db_permission)
    return db_permission

async def delete_permission(db: AsyncSession, permission_id: int) -> Optional[SCMPermission]:
    """Delete a permission from the database.
    Note: This will also remove associations from scm_role_permissions if cascade is set up correctly on the relationship,
    or if manually handled. SQLAlchemy's default for M2M might require explicit removal from association table
    or configuring cascade delete on the relationship in the models.
    For simplicity, we assume cascade might handle it or it's handled by DB constraints.
    A safer approach is to manually delete from the association table first if cascade is not 'all, delete-orphan' on both sides.
    """
    db_permission = await get_permission(db, permission_id)
    if not db_permission:
        return None

    # Manually remove associations if cascade is not configured (safer)
    # This requires the association table to be imported and used directly.
    # Example:
    # from app.models.db_models import scm_role_permissions_table
    # await db.execute(scm_role_permissions_table.delete().where(scm_role_permissions_table.c.permission_id == permission_id))
    
    # If Role.permissions relationship has cascade="all, delete" or similar for the secondary table,
    # or if SCMPermission.roles has it, then deleting SCMPermission might handle it.
    # However, for M2M, it's often best to explicitly manage the association table entries or ensure
    # the ORM relationship is configured to handle this (e.g., by clearing role.permissions list for all roles having this permission).
    # For now, we just delete the permission. DB foreign key constraints on scm_role_permissions might prevent this
    # if entries exist, unless ON DELETE CASCADE is set at the DB level.
    
    await db.delete(db_permission)
    await db.flush()
    return db_permission # Returns the object before it's fully expunged from session post-commit

async def assign_permission_to_role(db: AsyncSession, role_id: int, permission_id: int) -> bool:
    """Assigns a permission to a role. Returns True if successful or already assigned, False if role/permission not found."""
    # Use db.get for primary key lookups for potentially better performance and simplicity
    role = await db.get(Role, role_id, options=[selectinload(Role.permissions)])
    permission = await db.get(SCMPermission, permission_id)

    if not role or not permission:
        return False # Role or Permission not found
    
    if permission not in role.permissions:
        role.permissions.append(permission)
        db.add(role) # Mark role as dirty
        await db.flush() # Persist the change in association table
    return True

async def revoke_permission_from_role(db: AsyncSession, role_id: int, permission_id: int) -> bool:
    """Revokes a permission from a role. Returns True if successful or not assigned, False if role/permission not found."""
    role = await db.get(Role, role_id, options=[selectinload(Role.permissions)])
    permission = await db.get(SCMPermission, permission_id)

    if not role or not permission:
        return False # Role or Permission not found
        
    if permission in role.permissions:
        role.permissions.remove(permission)
        db.add(role) # Mark role as dirty
        await db.flush() # Persist the change in association table
    return True

async def get_permissions_for_role(db: AsyncSession, role_id: int) -> PyList[SCMPermission]:
    """Retrieve all permissions assigned to a specific role."""
    role = await db.get(Role, role_id, options=[selectinload(Role.permissions)])
    if role:
        return role.permissions
    return [] # Return empty list if role not found
