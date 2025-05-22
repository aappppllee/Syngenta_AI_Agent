import logging
import os
import traceback # For detailed error logging in global handler
import uuid # Though session ID generation is now more in AgentCore/QueryService

from fastapi import FastAPI, Request, status, Depends, BackgroundTasks # Added BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.routers import query_router as app_query_router
from app.routers import auth_router as app_auth_router
from app.routers import admin_router as app_admin_router
from app.services.query_service import QueryService
from agent.core import AgentCore
from agent.agent_config import (
    AgentCoreConfig, LLMConfig, NLUConfig, SQLLLMConfig, 
    CriteriaExtractorLLMConfig, DocumentRetrieverConfig, 
    EmbeddingConfig, VectorStoreConfig, AccessControlConfig, 
    HybridOrchestratorConfig, MemoryConfig
)
from app.config import settings
from app.db.database import create_db_and_tables, AsyncSessionLocal, get_db_session
from app.db.crud import role_crud, user_crud, audit_log_crud as crud_audit_log, policy_document_crud, permission_crud
from app.schemas.role_schemas import RoleCreate
from app.schemas.user_schemas import UserCreate
from app.schemas.policy_document_schemas import PolicyDocumentCreate
from app.schemas.permission_schemas import SCMPermissionCreate
from app.models.db_models import ActionTypeEnum, SystemUser, Role as DBRole, SCMPermission as DBPermission
from app.schemas.audit_log_schemas import AuditLogCreate as SchemaAuditLogCreate
from app.dependencies import get_current_admin_user
from app.utils.logging_utils import setup_logging

# --- Setup Logging ---
setup_logging() # Configure logging as one of the first actions
logger = logging.getLogger(__name__) # Get logger for main.py

# --- Initialize Agent Core and Services ---
logger.info("Starting Agent Core and Services initialization...")
try:
    llm_provider = "openai" # Default
    if settings.ANTHROPIC_API_KEY and not settings.OPENAI_API_KEY: 
        llm_provider = "anthropic"
        logger.info("Using Anthropic as LLM provider based on API key availability.")
    elif not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
        logger.warning("CRITICAL: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set. LLM functionality will fail.")
    else:
        logger.info(f"Using OpenAI as LLM provider (key found: {bool(settings.OPENAI_API_KEY)}).")


    embedding_provider = "openai" # Default, as per preferred stack
    if not settings.OPENAI_API_KEY and embedding_provider == "openai":
        logger.warning("OPENAI_API_KEY is not set, OpenAI embeddings will fail.")

    agent_core_cfg = AgentCoreConfig(
        llm_config=LLMConfig(
            provider=llm_provider, 
            model_name=settings.LLM_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY if llm_provider=="openai" else settings.ANTHROPIC_API_KEY,
            temperature=0.1
        ),
        nlu_config=NLUConfig(
            provider=llm_provider, 
            model_name=settings.NLU_LLM_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY if llm_provider=="openai" else settings.ANTHROPIC_API_KEY,
            temperature=0.0
        ),
        sql_llm_config=SQLLLMConfig(
            provider=llm_provider, 
            model_name=settings.SQL_LLM_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY if llm_provider=="openai" else settings.ANTHROPIC_API_KEY,
            temperature=0.0
        ),
        criteria_extractor_llm_config=CriteriaExtractorLLMConfig(
            provider=llm_provider, 
            model_name=settings.CRITERIA_EXTRACTION_LLM_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY if llm_provider=="openai" else settings.ANTHROPIC_API_KEY,
            temperature=0.0
        ),
        doc_retriever_config=DocumentRetrieverConfig(
            embedding_config=EmbeddingConfig(
                provider=embedding_provider, 
                model_name=settings.EMBEDDING_MODEL_NAME, 
                api_key=settings.OPENAI_API_KEY # Assuming OpenAI Embeddings
            ),
            vector_store_config=VectorStoreConfig(
                type="pgvector", 
                connection_string=settings.PGVECTOR_CONNECTION_STRING, 
                collection_name=settings.PGVECTOR_COLLECTION_NAME
            ),
            pdf_storage_base_path=os.path.abspath("./uploaded_policy_docs/")
        ),
        db_interactor_config=DatabaseInteractorConfig(
            db_url=settings.SYNC_DATABASE_URL, 
            include_tables=settings.SUPPLY_CHAIN_TABLE_NAMES.split(',') if settings.SUPPLY_CHAIN_TABLE_NAMES else []
        ),
        hybrid_orchestrator_config=HybridOrchestratorConfig(), 
        access_control_config=AccessControlConfig(), 
        memory_config=MemoryConfig(
            type="sql_chat_history", 
            buffer_type="buffer_window", 
            k=5, 
            db_connection_string=settings.SYNC_DATABASE_URL, 
            table_name=settings.CHAT_MESSAGE_TABLE_NAME
        )
    )
    agent_core_instance = AgentCore(config=agent_core_cfg)
    query_service_instance = QueryService(agent_core=agent_core_instance)
    logger.info("Agent Core and Query Service initialized successfully.")
except Exception as e_init:
    logger.critical(f"CRITICAL ERROR during AgentCoreConfig or AgentCore/QueryService initialization: {e_init}", exc_info=True)
    # Depending on the error, you might want to exit or run in a degraded mode.
    # For now, let it proceed so FastAPI can start and report errors via API if parts are broken.
    agent_core_instance = None # Ensure it's None if init failed
    query_service_instance = None


app = FastAPI(
    title=settings.APP_NAME, 
    description="API for interacting with the intelligent supply chain agent.",
    version="0.1.0", # Consider making this configurable
    debug=settings.DEBUG_MODE
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend's domain(s)
    allow_credentials=True,
    allow_methods=["*"], # Or specify like ["GET", "POST"]
    allow_headers=["*"], # Or specify allowed headers
)

# --- Refined Global Exception Handler ---
@app.exception_handler(Exception)
async def unified_exception_handler(request: Request, exc: Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = type(exc).__name__
    error_message = str(exc)
    
    # Log the full error with traceback
    tb_str = traceback.format_exc()
    logger.error(
        f"Unhandled exception for request {request.method} {request.url.path}: {error_type} - {error_message}\nTraceback:\n{tb_str}"
    )

    # Customize status code for specific known HTTP exceptions if they are not handled by FastAPI's default handlers
    if isinstance(exc, HTTPException): # Re-raise HTTPException to let FastAPI handle it
        # This ensures HTTPException's status_code and detail are used.
        # However, our goal is a unified JSON response.
        # So, we can extract info from HTTPException and format it.
        status_code = exc.status_code
        error_message = exc.detail
        # For HTTPExceptions, we might not want to show a generic "internal error" message.
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": error_type,
                    "message": error_message, # Use detail from HTTPException
                    "detail": None # Usually detail is already the message for HTTPException
                }
            }
        )
    
    return JSONResponse(
        status_code=status_code, # Default 500 for unhandled non-HTTPExceptions
        content={
            "error": {
                "type": error_type,
                "message": "An unexpected internal error occurred." if status_code == 500 else error_message,
                "detail": error_message if settings.DEBUG_MODE else "Internal server error. Please contact support."
            }
        },
    )

async def seed_initial_data():
    db_session = AsyncSessionLocal()
    logger.info("Starting initial data seeding...")
    try:
        # 1. Seed Roles
        roles_data = [
            {"role_name": settings.ADMIN_ROLE_NAME, "description": "Administrator"}, {"role_name": settings.FINANCE_ROLE_NAME, "description": "Handles financial data"},
            {"role_name": settings.PLANNING_ROLE_NAME, "description": "Handles planning data"}, {"role_name": "Supply Chain Analyst", "description": "General SC Analyst"},
            {"role_name": "Regional Manager India", "description": "Manager for India region"}
        ]
        created_roles_map: Dict[str, DBRole] = {}
        for r_data in roles_data:
            role = await role_crud.get_role_by_name(db_session, r_data["role_name"])
            if not role: 
                role = await role_crud.create_role(db_session, RoleCreate(**r_data))
                logger.info(f"Seeded role: {role.role_name}")
                await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(action_type=ActionTypeEnum.ROLE_CREATED, details=f"Role '{role.role_name}' seeded."))
            created_roles_map[role.role_name] = role
        
        # 2. Seed Admin & Regional User
        admin_role = created_roles_map.get(settings.ADMIN_ROLE_NAME)
        if admin_role and not await user_crud.get_user_by_username(db_session, "admin"):
            admin = await user_crud.create_user(db_session, UserCreate(username="admin", email="admin@example.com", password="adminpassword", role_id=admin_role.role_id, assigned_region="Global"))
            logger.info(f"Seeded admin user: {admin.username}"); await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(action_type=ActionTypeEnum.USER_CREATED, details=f"Admin user '{admin.username}' seeded."))
        
        regional_mgr_role = created_roles_map.get("Regional Manager India")
        if regional_mgr_role and not await user_crud.get_user_by_username(db_session, "mgr_india"):
            mgr_india = await user_crud.create_user(db_session, UserCreate(username="mgr_india", email="mgr.india@example.com", password="passwordindia", role_id=regional_mgr_role.role_id, assigned_region="India"))
            logger.info(f"Seeded regional manager: {mgr_india.username}"); await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(action_type=ActionTypeEnum.USER_CREATED, details=f"User '{mgr_india.username}' seeded."))

        # 3. Seed Permissions
        permissions_to_seed = [
            {"permission_name": "perform_intent:greeting", "category": "General"}, {"permission_name": "perform_intent:document_search", "category": "General"},
            {"permission_name": "perform_intent:database_query", "category": "General"}, {"permission_name": "perform_intent:hybrid_query", "category": "General"},
            {"permission_name": "view:document:generic", "category": "Document"}, {"permission_name": "view:document:inventory_management_policy", "category": "Document"},
            {"permission_name": "view:document:supplier_code_of_conduct", "category": "Document"}, {"permission_name": "view:document:financial_reports_q1_2023", "category": "DocumentFinancial"},
            {"permission_name": "query:database_table:generic", "category": "Database"}, {"permission_name": "query:database_table:scm_orders", "category": "DatabaseSCM"},
            {"permission_name": "query:database_table:scm_products", "category": "DatabaseSCM"}, {"permission_name": "query:database_table:scm_orders:region_india", "category": "DatabaseSCM"},
            {"permission_name": "query:database_table:scm_orders:region_emea", "category": "DatabaseSCM"},
            {"permission_name": "query:financial_data:general_financial_metrics", "category": "Financial"}, {"permission_name": "access_financial_data", "category": "Financial"},
            {"permission_name": "execute_hybrid:general_financial_metrics:inventory_management_policy", "category": "Hybrid"}, {"permission_name": "execute_hybrid:database_generic:document_generic", "category": "Hybrid"},
        ]
        created_permissions_map: Dict[str, DBPermission] = {}
        for p_data in permissions_to_seed:
            p_name = p_data["permission_name"]
            perm = await permission_crud.get_permission_by_name(db_session, p_name)
            if not perm: perm = await permission_crud.create_permission(db_session, SCMPermissionCreate(**p_data)); logger.info(f"Seeded permission: {perm.permission_name}"); await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(action_type=ActionTypeEnum.PERMISSION_CREATED, details=f"Permission '{perm.permission_name}' seeded."))
            created_permissions_map[perm.permission_name] = perm

        # 4. Assign Permissions to Roles
        if admin_role_obj := created_roles_map.get(settings.ADMIN_ROLE_NAME):
            for perm_obj in created_permissions_map.values(): await permission_crud.assign_permission_to_role(db_session, admin_role_obj.role_id, perm_obj.permission_id)
            logger.info(f"Assigned all seeded permissions to Admin role.")
        
        if finance_role_obj := created_roles_map.get(settings.FINANCE_ROLE_NAME):
            perms = ["access_financial_data", "query:financial_data:general_financial_metrics", "query:database_table:generic", "view:document:generic", "execute_hybrid:general_financial_metrics:inventory_management_policy", "perform_intent:database_query", "perform_intent:document_search", "perform_intent:hybrid_query"]
            for p_name in perms:
                if p_obj := created_permissions_map.get(p_name): await permission_crud.assign_permission_to_role(db_session, finance_role_obj.role_id, p_obj.permission_id)
            logger.info(f"Assigned permissions to Finance role.")
        
        if analyst_role_obj := created_roles_map.get("Supply Chain Analyst"):
            perms = ["view:document:generic", "view:document:inventory_management_policy", "query:database_table:generic", "query:database_table:scm_products", "execute_hybrid:database_generic:document_generic", "perform_intent:greeting", "perform_intent:document_search", "perform_intent:database_query", "perform_intent:hybrid_query"]
            for p_name in perms:
                if p_obj := created_permissions_map.get(p_name): await permission_crud.assign_permission_to_role(db_session, analyst_role_obj.role_id, p_obj.permission_id)
            logger.info(f"Assigned permissions to Supply Chain Analyst role.")

        if regional_mgr_india_obj := created_roles_map.get("Regional Manager India"):
            perms = ["view:document:generic", "query:database_table:scm_orders:region_india", "perform_intent:greeting", "perform_intent:document_search", "perform_intent:database_query"]
            for p_name in perms:
                if p_obj := created_permissions_map.get(p_name): await permission_crud.assign_permission_to_role(db_session, regional_mgr_india_obj.role_id, p_obj.permission_id)
            logger.info(f"Assigned permissions to Regional Manager India role.")

        # Seed dummy policy doc (as before, condensed)
        # ... (dummy PDF creation logic from previous steps, ensure it uses agent_core_cfg for path) ...
        
        await db_session.commit()
        logger.info("Initial data seeding completed successfully.")
    except Exception as e_seed:
        await db_session.rollback()
        logger.error(f"Error during initial data seeding: {e_seed}", exc_info=True)
    finally:
        await db_session.close()

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Application startup: {settings.APP_NAME} - Debug: {settings.DEBUG_MODE}")
    logger.info(f"Database URL (async for app): {settings.DATABASE_URL}")
    logger.info(f"Database URL (sync for Langchain tools): {settings.SYNC_DATABASE_URL}")
    
    if agent_core_cfg and agent_core_cfg.doc_retriever_config: # Check if config is available
        pdf_dir = agent_core_cfg.doc_retriever_config.pdf_storage_base_path
        if not os.path.exists(pdf_dir):
            try:
                os.makedirs(pdf_dir, exist_ok=True)
                logger.info(f"Created PDF storage directory: {pdf_dir}")
            except Exception as e_dir:
                logger.error(f"Failed to create PDF storage directory {pdf_dir}: {e_dir}")
    else:
        logger.warning("AgentCoreConfig or DocRetrieverConfig not fully initialized, skipping PDF directory creation check.")


    try:
        await create_db_and_tables()
        await seed_initial_data()
        if agent_core_instance: # Check if agent_core_instance was successfully created
            await agent_core_instance.load_models_and_resources()
        else:
            logger.critical("AgentCore instance is None, skipping model loading. Check initialization errors.")
        
        # Make query_service_instance available to the router
        # This direct attribute assignment is a simplification for the skeleton.
        # More robust DI might use a shared context or a Depends factory for services.
        if query_service_instance and hasattr(app_query_router, 'router_query'):
             app_query_router.router_query._query_service_instance = query_service_instance
        else:
            logger.warning("QueryService instance or query_router.router_query not available for DI simulation.")

        logger.info("Application startup complete and resources initialized.")
    except Exception as e_startup:
        logger.critical(f"CRITICAL ERROR during application startup sequence: {e_startup}", exc_info=True)
        # Consider sys.exit(1) if startup is critically failed

# --- Admin endpoint to trigger SCM data loading ---
# Moved from admin_router.py to main.py to use BackgroundTasks from app instance
# This is a common pattern if the router itself doesn't have the app instance for BG tasks.
# Alternatively, pass `app` or `BackgroundTasks` to the router.
from app.data_loading.load_dataco_csv import load_data_from_csv # Ensure this is importable

async def background_load_scm_data_main(db_session: AsyncSession, admin_user_id: int, csv_path: str):
    """Wrapper to run data loading in background and log completion/error."""
    logger.info(f"Background task started: Loading SCM data from {csv_path}")
    result = {"status": "unknown", "message": "Task did not complete or an error occurred before result assignment."}
    try:
        result = await load_data_from_csv(db_session, csv_file_path=csv_path)
        await db_session.commit() # Commit after successful load
        log_details = f"SCM data loading from {csv_path} completed. Status: {result.get('status')}, Message: {result.get('message')}"
        action = ActionTypeEnum.DATA_LOADED if result.get('status') == 'success' else ActionTypeEnum.ADMIN_ACTION
        await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(user_id=admin_user_id, action_type=action, details=log_details, accessed_resource="SCM Tables"))
        await db_session.commit() # Commit audit log
    except Exception as e_bg_load:
        await db_session.rollback()
        logger.error(f"Error in background_load_scm_data: {e_bg_load}", exc_info=True)
        result = {"status": "error", "message": str(e_bg_load)}
        log_details = f"SCM data loading from {csv_path} failed. Error: {str(e_bg_load)}"
        try:
            await crud_audit_log.create_audit_log(db_session, SchemaAuditLogCreate(user_id=admin_user_id, action_type=ActionTypeEnum.ADMIN_ACTION, details=log_details, accessed_resource="SCM Tables"))
            await db_session.commit()
        except Exception as log_e:
            logger.error(f"Failed to log data loading error: {log_e}", exc_info=True)
            await db_session.rollback()
    finally:
        await db_session.close()
        logger.info(f"Background task finished: Loading SCM data. Final Result: {result}")


@app.post("/api/v1/admin/load-scm-data", status_code=status.HTTP_202_ACCEPTED, tags=["Administration - Data Loading"], summary="Trigger SCM Data Loading from CSV")
async def trigger_scm_data_loading_endpoint( # Renamed to avoid conflict if admin_router has same path
    background_tasks: BackgroundTasks,
    admin_user: SystemUser = Depends(get_current_admin_user)
):
    csv_path = settings.DATAGO_CSV_FILE_PATH
    if not os.path.exists(csv_path):
        logger.error(f"SCM Data Load Trigger: CSV file not found at {csv_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"CSV file not found at configured path: {csv_path}")

    logger.info(f"Admin user {admin_user.username} triggered SCM data loading from: {csv_path}")
    db_session_for_task = AsyncSessionLocal() # Create a new session for the background task
    background_tasks.add_task(background_load_scm_data_main, db_session_for_task, admin_user.user_id, csv_path)
    return {"message": "SCM data loading process initiated in the background. Check server logs for progress and completion."}


@app.on_event("shutdown")
async def shutdown_event(): 
    logger.info("Application shutdown initiated...")
    # Add any other cleanup tasks here if needed
    logger.info("Application shutdown complete.")

# --- API Routers ---
app.include_router(app_auth_router.router_auth, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(app_query_router.router_query, prefix="/api/v1/query", tags=["Agent Queries"])
app.include_router(app_admin_router.router, prefix="/api/v1", tags=["Administration"]) # admin_router itself has /admin prefix

@app.get("/", tags=["Root"])
async def read_root(): 
    logger.debug("Root endpoint '/' accessed.")
    return {"message": f"Welcome to {settings.APP_NAME}!"}
