from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload 
from sqlalchemy import and_, delete, insert # Import delete and insert
from typing import List as PyList, Optional

from app.models.db_models import SCMPermission, Role, scm_role_permissions_table 
from app.schemas.permission_schemas import SCMPermissionCreate, SCMPermissionUpdate

async def get_permission(db: AsyncSession, permission_id: int) -> Optional[SCMPermission]:
    """Retrieve a single permission by its ID."""
    return await db.get(SCMPermission, permission_id)

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
    await db.flush() 
    await db.refresh(db_permission) 
    return db_permission

async def update_permission(db: AsyncSession, permission_id: int, permission_update: SCMPermissionUpdate) -> Optional[SCMPermission]:
    """Update an existing permission."""
    db_permission = await db.get(SCMPermission, permission_id) 
    if not db_permission:
        return None
    
    update_data = permission_update.model_dump(exclude_unset=True) 
    for key, value in update_data.items():
        setattr(db_permission, key, value)
    
    db.add(db_permission) 
    await db.flush()
    await db.refresh(db_permission)
    return db_permission

async def delete_permission(db: AsyncSession, permission_id: int) -> Optional[SCMPermission]:
    """Delete a permission from the database."""
    db_permission = await db.get(SCMPermission, permission_id)
    if not db_permission:
        return None
    
    # Before deleting the permission, ensure related entries in the association table are handled.
    # This can be done by direct delete or relying on DB cascades if set up.
    # For explicit control:
    stmt_delete_associations = delete(scm_role_permissions_table).where(
        scm_role_permissions_table.c.permission_id == permission_id
    )
    await db.execute(stmt_delete_associations)
    
    await db.delete(db_permission)
    await db.flush()
    return db_permission 

async def assign_permission_to_role(db: AsyncSession, role_id: int, permission_id: int) -> bool:
    """Assigns a permission to a role by directly inserting into the association table if not present."""
    # Check if role and permission exist
    role = await db.get(Role, role_id)
    permission = await db.get(SCMPermission, permission_id)

    if not role or not permission:
        # logger.warning(f"Role (ID: {role_id}) or Permission (ID: {permission_id}) not found for assignment.")
        return False

    # Check if the association already exists by querying the association table directly
    association_exists_stmt = select(scm_role_permissions_table).where(
        and_(
            scm_role_permissions_table.c.role_id == role_id,
            scm_role_permissions_table.c.permission_id == permission_id
        )
    ).limit(1)

    result = await db.execute(association_exists_stmt)
    existing_association = result.one_or_none()

    if existing_association is None: # If no existing association, then insert directly
        insert_stmt = insert(scm_role_permissions_table).values(role_id=role_id, permission_id=permission_id)
        await db.execute(insert_stmt)
        await db.flush() # Persist the change in the association table
        # logger.info(f"Assigned permission {permission_id} to role {role_id} via direct insert.")
    # else:
        # logger.debug(f"Permission {permission_id} already assigned to role {role_id}.")
        
    return True

async def revoke_permission_from_role(db: AsyncSession, role_id: int, permission_id: int) -> bool:
    """Revokes a permission from a role by directly deleting from the association table."""
    # Check if role and permission exist (optional, as delete won't fail if they don't, but good for logic)
    role = await db.get(Role, role_id)
    permission = await db.get(SCMPermission, permission_id)

    if not role or not permission:
        # logger.warning(f"Role (ID: {role_id}) or Permission (ID: {permission_id}) not found for revocation check.")
        # Still proceed to attempt delete from association table, as it might exist even if objects are gone.
        pass

    delete_stmt = delete(scm_role_permissions_table).where(
        and_(
            scm_role_permissions_table.c.role_id == role_id,
            scm_role_permissions_table.c.permission_id == permission_id
        )
    )
    result = await db.execute(delete_stmt)
    await db.flush() # Persist the change
    
    # result.rowcount will tell how many rows were deleted.
    # if result.rowcount > 0:
        # logger.info(f"Revoked permission {permission_id} from role {role_id} via direct delete. Rows affected: {result.rowcount}")
    # else:
        # logger.debug(f"Permission {permission_id} was not assigned to role {role_id} (no rows deleted).")
    return True # Return True indicating the operation was attempted.

async def get_permissions_for_role(db: AsyncSession, role_id: int) -> PyList[SCMPermission]:
    """Retrieve all permissions assigned to a specific role."""
    # This still relies on the ORM relationship with selectinload, which is generally fine for reads.
    # If this also causes issues, a more complex query joining Role, scm_role_permissions_table,
    # and SCMPermission would be needed.
    role = await db.get(Role, role_id, options=[selectinload(Role.permissions)])
    if role and role.permissions: 
        return list(role.permissions) 
    return [] 
