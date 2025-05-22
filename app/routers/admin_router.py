import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request # Added Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select # Added for checking users with role
from typing import List as PyList, Optional, Dict # Added Dict

# from agent.core import agent_core_instance # REMOVE THIS LINE
from app.db.database import get_db_session, AsyncSessionLocal
from app.dependencies import get_current_admin_user 
from app.models.db_models import SystemUser, Role as DBRole, PolicyDocument as DBPolicyDocument, ActionTypeEnum, SCMPermission as DBPermission, SCMChatMessage
from app.schemas import user_schemas, role_schemas, policy_document_schemas, audit_log_schemas, permission_schemas, chat_message_schemas
from app.db.crud import user_crud, role_crud, policy_document_crud, permission_crud, audit_log_crud as crud_audit_log, chat_message_crud
# from app.data_loading.load_dataco_csv import load_data_from_csv # This is handled in main.py now
from app.config import settings
from agent.core import AgentCore # Import the class for type hinting if needed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin", 
    tags=["Admin Management"],
    dependencies=[Depends(get_current_admin_user)] 
)

# Helper function to get agent_core_instance from request
def get_agent_core(request: Request) -> AgentCore:
    agent_core = getattr(request.app.state, 'agent_core', None)
    if not agent_core:
        logger.critical("AgentCore instance not found on app.state. This indicates a setup error in main.py.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent core service is not configured or available. Please contact support."
        )
    return agent_core

# --- User Management Endpoints (Admin) ---
@router.get("/users", response_model=PyList[user_schemas.UserSchema], summary="List All Users (Admin)")
async def admin_read_users(
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(get_db_session)
):
    logger.info(f"Admin: Fetching users (skip={skip}, limit={limit}).")
    users = await user_crud.get_users(db, skip=skip, limit=limit)
    return [user_schemas.UserSchema.model_validate(user) for user in users]


@router.get("/users/{user_id}", response_model=user_schemas.UserSchema, summary="Get User by ID (Admin)")
async def admin_read_user(user_id: int, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching user with ID: {user_id}.")
    db_user = await user_crud.get_user(db, user_id=user_id)
    if db_user is None:
        logger.warning(f"Admin: User with ID {user_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_schemas.UserSchema.model_validate(db_user)

@router.put("/users/{user_id}", response_model=user_schemas.UserSchema, summary="Update User (Admin)")
async def admin_update_user(
    user_id: int, 
    user_in: user_schemas.UserUpdate, 
    db: AsyncSession = Depends(get_db_session),
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    logger.info(f"Admin '{admin_user.username}': Attempting to update user ID {user_id} with data: {user_in.model_dump(exclude_unset=True)}")
    if user_in.role_id is not None:
        role = await role_crud.get_role(db, role_id=user_in.role_id)
        if not role:
            logger.warning(f"Admin: Update user failed. Role with ID {user_in.role_id} not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Role with ID {user_in.role_id} not found for update.")

    updated_user_orm = await user_crud.update_user(db, user_id=user_id, user_update_data=user_in)
    if updated_user_orm is None:
        logger.warning(f"Admin: Update user failed. User ID {user_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    try:
        await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
            user_id=admin_user.user_id, action_type=ActionTypeEnum.USER_UPDATED,
            details=f"Admin '{admin_user.username}' updated user ID {user_id}. Changes: {user_in.model_dump(exclude_unset=True)}",
            accessed_resource=f"user:{user_id}"
        ))
    except Exception as e_log:
        logger.error(f"Failed to create audit log for user update: {e_log}", exc_info=settings.DEBUG_MODE)

    user_for_response = await user_crud.get_user(db, user_id=updated_user_orm.user_id) 
    return user_schemas.UserSchema.model_validate(user_for_response)


# --- Role Management Endpoints (Admin) ---
@router.post("/roles", response_model=role_schemas.RoleSchema, status_code=status.HTTP_201_CREATED, summary="Create Role (Admin)")
async def admin_create_role(
    role_in: role_schemas.RoleCreate, 
    db: AsyncSession = Depends(get_db_session),
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    logger.info(f"Admin '{admin_user.username}': Attempting to create role: {role_in.role_name}")
    existing_role = await role_crud.get_role_by_name(db, name=role_in.role_name)
    if existing_role:
        logger.warning(f"Admin: Role creation failed. Role name '{role_in.role_name}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role name already exists")
    created_role = await role_crud.create_role(db, role=role_in)
    
    try:
        await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
            user_id=admin_user.user_id, action_type=ActionTypeEnum.ROLE_CREATED,
            details=f"Admin '{admin_user.username}' created role: {created_role.role_name} (ID: {created_role.role_id})",
            accessed_resource=f"role:{created_role.role_id}"
        ))
    except Exception as e_log:
        logger.error(f"Failed to create audit log for role creation: {e_log}", exc_info=settings.DEBUG_MODE)
    return role_schemas.RoleSchema.model_validate(created_role)

@router.get("/roles", response_model=PyList[permission_schemas.RoleWithPermissionsSchema], summary="List All Roles with Permissions (Admin)")
async def admin_read_roles_with_permissions(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching roles with permissions (skip={skip}, limit={limit}).")
    roles = await role_crud.get_roles(db, skip=skip, limit=limit) 
    return [permission_schemas.RoleWithPermissionsSchema.model_validate(role) for role in roles]


@router.get("/roles/{role_id}", response_model=permission_schemas.RoleWithPermissionsSchema, summary="Get Role by ID with Permissions (Admin)")
async def admin_read_role_with_permissions(role_id: int, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching role ID {role_id} with permissions.")
    db_role = await role_crud.get_role(db, role_id=role_id) 
    if db_role is None:
        logger.warning(f"Admin: Role ID {role_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return permission_schemas.RoleWithPermissionsSchema.model_validate(db_role)


@router.put("/roles/{role_id}", response_model=role_schemas.RoleSchema, summary="Update Role (Admin)")
async def admin_update_role(
    role_id: int, 
    role_in: role_schemas.RoleUpdate, 
    db: AsyncSession = Depends(get_db_session),
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    logger.info(f"Admin '{admin_user.username}': Attempting to update role ID {role_id}")
    updated_role = await role_crud.update_role(db, role_id=role_id, role_update_data=role_in) 
    if updated_role is None:
        logger.warning(f"Admin: Update role failed. Role ID {role_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    
    try:
        await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
            user_id=admin_user.user_id, action_type=ActionTypeEnum.ROLE_UPDATED,
            details=f"Admin '{admin_user.username}' updated role ID {role_id}. Changes: {role_in.model_dump(exclude_unset=True)}",
            accessed_resource=f"role:{role_id}"
        ))
    except Exception as e_log:
        logger.error(f"Failed to create audit log for role update: {e_log}", exc_info=settings.DEBUG_MODE)
    return role_schemas.RoleSchema.model_validate(updated_role)


@router.delete("/roles/{role_id}", response_model=role_schemas.RoleSchema, summary="Delete Role (Admin)")
async def admin_delete_role(
    role_id: int, 
    db: AsyncSession = Depends(get_db_session),
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    logger.info(f"Admin '{admin_user.username}': Attempting to delete role ID {role_id}")
    # Check if any user is assigned this role
    from sqlalchemy import select # Local import for this specific query
    users_with_role_result = await db.execute(select(SystemUser).filter(SystemUser.role_id == role_id).limit(1)) 
    if users_with_role_result.scalars().first():
        logger.warning(f"Admin: Delete role failed. Role ID {role_id} is currently assigned to users.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role cannot be deleted, it is currently assigned to users.")

    deleted_role = await role_crud.delete_role(db, role_id=role_id) 
    if deleted_role is None:
        logger.warning(f"Admin: Delete role failed. Role ID {role_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    
    try:
        await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
            user_id=admin_user.user_id, action_type=ActionTypeEnum.ROLE_DELETED,
            details=f"Admin '{admin_user.username}' deleted role: {deleted_role.role_name} (ID: {role_id})",
            accessed_resource=f"role:{role_id}"
        ))
    except Exception as e_log:
        logger.error(f"Failed to create audit log for role deletion: {e_log}", exc_info=settings.DEBUG_MODE)
    return role_schemas.RoleSchema.model_validate(deleted_role)


# --- Policy Document Management Endpoints (Admin) ---
@router.post("/documents", response_model=policy_document_schemas.PolicyDocumentSchema, status_code=status.HTTP_201_CREATED, summary="Create Policy Document Metadata (Admin)")
async def admin_create_policy_document(
    doc_in: policy_document_schemas.PolicyDocumentCreate,
    db: AsyncSession = Depends(get_db_session),
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    logger.info(f"Admin '{admin_user.username}': Creating policy document: {doc_in.title}")
    existing_doc = await policy_document_crud.get_policy_document_by_title(db, title=doc_in.title)
    if existing_doc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Policy document with this title already exists")
    created_doc = await policy_document_crud.create_policy_document(db, doc=doc_in)
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.DOC_CREATED,details=f"Admin '{admin_user.username}' created doc: {created_doc.title}"))
    return created_doc

@router.get("/documents", response_model=PyList[policy_document_schemas.PolicyDocumentSchema], summary="List Policy Documents (Admin)")
async def admin_read_policy_documents(skip: int = 0, limit: int = 100, document_type: Optional[str] = None, keyword_search: Optional[str] = None, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching policy documents.")
    documents = await policy_document_crud.get_policy_documents(db, skip=skip, limit=limit, document_type=document_type, keyword_search=keyword_search)
    return documents

@router.get("/documents/{document_id}", response_model=policy_document_schemas.PolicyDocumentSchema, summary="Get Policy Document by ID (Admin)")
async def admin_read_policy_document(document_id: int, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching policy document ID {document_id}.")
    db_doc = await policy_document_crud.get_policy_document(db, document_id=document_id)
    if db_doc is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Policy document not found")
    return db_doc

@router.put("/documents/{document_id}", response_model=policy_document_schemas.PolicyDocumentSchema, summary="Update Policy Document Metadata (Admin)")
async def admin_update_policy_document(document_id: int, doc_in: policy_document_schemas.PolicyDocumentUpdate, db: AsyncSession = Depends(get_db_session), admin_user: SystemUser = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin_user.username}': Updating policy document ID {document_id}")
    updated_doc = await policy_document_crud.update_policy_document(db, document_id=document_id, doc_update_data=doc_in)
    if updated_doc is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Policy document not found")
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.DOC_UPDATED,details=f"Admin '{admin_user.username}' updated doc ID {document_id}"))
    return updated_doc

@router.delete("/documents/{document_id}", response_model=policy_document_schemas.PolicyDocumentSchema, summary="Delete Policy Document Metadata (Admin)")
async def admin_delete_policy_document(document_id: int, db: AsyncSession = Depends(get_db_session), admin_user: SystemUser = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin_user.username}': Deleting policy document ID {document_id}")
    deleted_doc = await policy_document_crud.delete_policy_document(db, document_id=document_id)
    if deleted_doc is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Policy document not found")
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.DOC_DELETED,details=f"Admin '{admin_user.username}' deleted doc: {deleted_doc.title}"))
    return deleted_doc


# --- Permission Management Endpoints (Admin) ---
@router.post("/permissions", response_model=permission_schemas.SCMPermissionSchema, status_code=status.HTTP_201_CREATED, summary="Create Permission (Admin)")
async def admin_create_permission(permission_in: permission_schemas.SCMPermissionCreate, db: AsyncSession = Depends(get_db_session), admin_user: SystemUser = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin_user.username}': Creating permission: {permission_in.permission_name}")
    existing = await permission_crud.get_permission_by_name(db, permission_name=permission_in.permission_name)
    if existing: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Permission name already exists")
    permission = await permission_crud.create_permission(db, permission=permission_in)
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.PERMISSION_CREATED, details=f"Admin created permission: {permission.permission_name}"))
    return permission

@router.get("/permissions", response_model=PyList[permission_schemas.SCMPermissionSchema], summary="List All Permissions (Admin)")
async def admin_read_permissions(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching permissions.")
    permissions = await permission_crud.get_permissions(db, skip=skip, limit=limit)
    return permissions

@router.post("/roles/assign-permission", status_code=status.HTTP_200_OK, summary="Assign Permission to Role (Admin)")
async def admin_assign_permission_to_role(assignment: permission_schemas.RolePermissionAssignment, db: AsyncSession = Depends(get_db_session), admin_user: SystemUser = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin_user.username}': Assigning perm {assignment.permission_id} to role {assignment.role_id}")
    success = await permission_crud.assign_permission_to_role(db, role_id=assignment.role_id, permission_id=assignment.permission_id)
    if not success: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role or Permission not found, or assignment failed.")
    role = await role_crud.get_role(db, assignment.role_id); perm = await permission_crud.get_permission(db, assignment.permission_id)
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.PERMISSION_ASSIGNED, details=f"Admin assigned perm '{perm.permission_name if perm else 'N/A'}' to role '{role.role_name if role else 'N/A'}'"))
    return {"message": "Permission assigned successfully"}

@router.post("/roles/revoke-permission", status_code=status.HTTP_200_OK, summary="Revoke Permission from Role (Admin)")
async def admin_revoke_permission_from_role(assignment: permission_schemas.RolePermissionAssignment, db: AsyncSession = Depends(get_db_session), admin_user: SystemUser = Depends(get_current_admin_user)):
    logger.info(f"Admin '{admin_user.username}': Revoking perm {assignment.permission_id} from role {assignment.role_id}")
    success = await permission_crud.revoke_permission_from_role(db, role_id=assignment.role_id, permission_id=assignment.permission_id)
    if not success: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role or Permission not found, or permission not assigned/revocation failed.")
    role = await role_crud.get_role(db, assignment.role_id); perm = await permission_crud.get_permission(db, assignment.permission_id)
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(user_id=admin_user.user_id, action_type=ActionTypeEnum.PERMISSION_REVOKED, details=f"Admin revoked perm '{perm.permission_name if perm else 'N/A'}' from role '{role.role_name if role else 'N/A'}'"))
    return {"message": "Permission revoked successfully"}

@router.get("/roles/{role_id}/permissions", response_model=PyList[permission_schemas.SCMPermissionSchema], summary="Get Permissions for a Role (Admin)")
async def admin_get_permissions_for_role(role_id: int, db: AsyncSession = Depends(get_db_session)):
    logger.info(f"Admin: Fetching permissions for role ID {role_id}")
    role = await role_crud.get_role(db, role_id=role_id)
    if not role: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return role.permissions


# --- Endpoint for Document Indexing ---
@router.post("/index-documents", status_code=status.HTTP_202_ACCEPTED, summary="Trigger Document Indexing (Admin)")
async def trigger_document_indexing_admin_endpoint( 
    request: Request, # Added request to access app.state
    background_tasks: BackgroundTasks, 
    admin_user: SystemUser = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db_session) 
):
    agent_core = get_agent_core(request) # Get agent_core from app.state
    logger.info(f"Admin user {admin_user.username} triggered document indexing via admin router.")
    
    async def background_index_task():
        task_db_session = AsyncSessionLocal()
        try:
            logger.info("Background task: Starting document indexing...")
            if agent_core and hasattr(agent_core, 'document_retriever'):
                 await agent_core.document_retriever.index_documents_from_db() 
                 logger.info("Background task: Document indexing completed by retriever.")
                 await crud_audit_log.create_audit_log(task_db_session, audit_log_schemas.AuditLogCreate(
                    user_id=admin_user.user_id, action_type=ActionTypeEnum.DOC_INDEXED, # Changed to DOC_INDEXED
                    details=f"Background document indexing triggered by admin '{admin_user.username}' completed.",
                    accessed_resource="all_documents_for_indexing"
                 ))
                 await task_db_session.commit()
            else:
                logger.error("Background task: agent_core_instance or document_retriever not available for indexing.")
        except Exception as e_index:
            logger.error(f"Background task: Error during document indexing: {e_index}", exc_info=settings.DEBUG_MODE)
            await task_db_session.rollback() 
        finally:
            await task_db_session.close()

    background_tasks.add_task(background_index_task)
    
    await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
        user_id=admin_user.user_id, action_type=ActionTypeEnum.ADMIN_ACTION,
        details=f"Admin '{admin_user.username}' initiated document indexing process.",
        accessed_resource="document_indexing_trigger"
    ))
    await db.commit() # Commit the audit log for triggering
    return {"message": "Document indexing process initiated in the background. Check server logs."}

# --- Endpoint for clearing chat session memory (Admin) ---
@router.delete("/sessions/{session_id}/clear-memory", status_code=status.HTTP_200_OK, summary="Clear Chat Session Memory (Admin)")
async def admin_clear_chat_session_memory(
    session_id: str,
    request: Request, # Added request to access app.state
    admin_user: SystemUser = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db_session) 
):
    agent_core = get_agent_core(request) # Get agent_core from app.state
    logger.info(f"Admin '{admin_user.username}' attempting to clear memory for session ID: {session_id}")
    
    if not agent_core or not hasattr(agent_core, 'memory_manager'): # Check the retrieved agent_core
        logger.error("Admin clear memory: Agent core or memory manager not available.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Memory service not configured.")

    try:
        agent_core.memory_manager.clear_session_memory(session_id) 
        logger.info(f"Memory cleared for session ID: {session_id} by admin '{admin_user.username}'.")
        
        await crud_audit_log.create_audit_log(db, audit_log_schemas.AuditLogCreate(
            user_id=admin_user.user_id, action_type=ActionTypeEnum.ADMIN_ACTION,
            details=f"Admin '{admin_user.username}' cleared memory for session ID: {session_id}",
            accessed_resource=f"session_memory:{session_id}"
        ))
        await db.commit() # Commit the audit log
        return {"message": f"Memory for session ID '{session_id}' cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing memory for session ID '{session_id}': {e}", exc_info=settings.DEBUG_MODE)
        await db.rollback() # Rollback audit log if clearing failed
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to clear memory: {str(e)}")

