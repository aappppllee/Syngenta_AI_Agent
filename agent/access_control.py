import logging
from typing import Dict, Any, Optional, Tuple, List as PyList
from sqlalchemy.ext.asyncio import AsyncSession 
from app.db.database import AsyncSessionLocal 
from app.db.crud import permission_crud 
from agent.agent_config import AccessControlConfig
from agent.nlu import ExtractedEntity
from app.config import settings

logger = logging.getLogger(__name__)

class AccessController:
    def __init__(self, config: AccessControlConfig):
        self.config = config
        logger.info("AccessController initialized (Database-Driven with refined mapping).")

    async def _get_user_permissions(self, role_name: str) -> PyList[str]:
        db_session = AsyncSessionLocal()
        try:
            # Local import to avoid potential circular dependency issues at module load time
            from app.db.crud.role_crud import get_role_by_name 
            role = await get_role_by_name(db_session, name=role_name) # This should eager load permissions
            if role and hasattr(role, 'permissions') and role.permissions:
                user_perms = [p.permission_name for p in role.permissions]
                logger.debug(f"Permissions for role '{role_name}': {user_perms}")
                return user_perms
            logger.debug(f"No permissions found or role '{role_name}' does not exist.")
            return []
        except Exception as e:
            logger.error(f"Error fetching permissions for role '{role_name}': {e}", exc_info=settings.DEBUG_MODE)
            return []
        finally:
            await db_session.close()
    
    def _map_resource_to_permission_strings(
        self, 
        action: str, 
        resource_type: str, 
        resource_name: Optional[str] = None,
        target_region: Optional[str] = None,
        intent: Optional[str] = None 
    ) -> PyList[str]:
        permissions: PyList[str] = []
        name_norm = str(resource_name).lower().replace(" ", "_").replace(".", "_") if resource_name else "generic"
        region_norm = target_region.lower().replace(" ", "_") if target_region else None

        if region_norm:
            permissions.append(f"{action}:{resource_type}:{name_norm}:region_{region_norm}")
        
        permissions.append(f"{action}:{resource_type}:{name_norm}")
        permissions.append(f"{action}:{resource_type}:generic")
        
        if "database" in resource_type: 
            permissions.append(f"{action}:database_generic")
        if "document" in resource_type:
            permissions.append(f"{action}:document_generic")
        if "financial" in resource_type:
             permissions.append(f"{action}:financial_data_generic")

        if intent:
            permissions.append(f"perform_intent:{intent}")

        unique_perms = sorted(list(set(p for p in permissions if p)))
        logger.debug(f"AccessController: Generated potential permission strings: {unique_perms} for action='{action}', type='{resource_type}', name='{name_norm}', region='{region_norm}'")
        return unique_perms


    async def check_permission(
        self,
        user_context: Dict[str, Any], 
        intent: str,
        entities: PyList[ExtractedEntity], 
        resource_details: Optional[Dict[str, Any]] = None 
    ) -> Tuple[bool, Optional[str]]:
        user_role = user_context.get("role_name")
        user_assigned_region = user_context.get("assigned_region", "Global").lower()

        logger.info(f"AccessController: UserRole='{user_role}', UserRegion='{user_assigned_region}', Intent='{intent}'")
        if resource_details: logger.debug(f"AccessController: ResourceDetails for check: {resource_details}")

        if not user_role: 
            logger.warning("AccessController: User role not found in context.")
            return False, "User role not found in context."
        if user_role == settings.ADMIN_ROLE_NAME: 
            logger.info("AccessController: Admin user, access granted by default.")
            return True, None

        user_permissions_set = set(await self._get_user_permissions(user_role))
        if not user_permissions_set: 
            logger.warning(f"AccessController: No DB permissions found for role '{user_role}'. Access denied by default.")
            return False, f"Role '{user_role}' has no defined permissions."

        action = "view" 
        if intent == "database_query": action = "query"
        elif intent == "document_search": action = "view"
        elif intent == "hybrid_query": action = "execute_hybrid"
        elif intent == "greeting" or intent == "chit_chat": action = "interact"
        else: action = "perform" 

        res_type = resource_details.get("resource_type", "unknown_resource").lower()
        res_name = resource_details.get("resource_name", "generic").lower()
        res_target_region = resource_details.get("target_region") 
        
        potential_required_permissions = self._map_resource_to_permission_strings(
            action=action, resource_type=res_type, resource_name=res_name,
            target_region=res_target_region, intent=intent
        )
        
        if not potential_required_permissions:
            logger.warning(f"AccessController: No specific permission mapping found for intent '{intent}' and resource '{res_type}:{res_name}'. Denying.")
            return False, f"Action '{intent}' on resource '{res_type}:{res_name}' is not mapped to a required permission."

        logger.debug(f"AccessController: User '{user_context.get('username')}' (Role: {user_role}) has DB permissions: {user_permissions_set}")
        logger.debug(f"AccessController: Checking against potential required permissions: {potential_required_permissions}")

        granted_by_permission = None
        for req_perm in potential_required_permissions:
            if req_perm in user_permissions_set:
                granted_by_permission = req_perm
                break 
        
        if granted_by_permission:
            logger.info(f"AccessController: Permission '{granted_by_permission}' GRANTED for role '{user_role}'.")
            
            if user_assigned_region != "global" and res_target_region:
                is_granted_perm_region_specific_to_target = f":region_{res_target_region.lower()}" in granted_by_permission
                
                if res_target_region != user_assigned_region:
                    if not is_granted_perm_region_specific_to_target:
                         reason = (f"Geographic restriction: Your region is '{user_context.get('assigned_region')}', "
                                  f"but query targets region '{res_target_region}'. Permission '{granted_by_permission}' does not grant cross-region access.")
                         logger.warning(f"AccessController: {reason}")
                         return False, reason
            
            return True, None 
        else:
            reason = f"Access denied. Role '{user_role}' lacks any of the required permissions: {potential_required_permissions} for resource type '{res_type}', name '{res_name}' (action: {action})."
            logger.warning(f"AccessController: {reason}")
            return False, reason
