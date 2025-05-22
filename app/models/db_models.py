import enum
import uuid
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text,
    Enum as SQLAlchemyEnum, Numeric, Date, Table, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY # For pgvector if storing vectors directly in ORM
from sqlalchemy.sql import func # For server_default=func.now()

from app.db.database import Base # Import Base from your database setup

# --- Enum for Action Types (used in AuditLog) ---
class ActionTypeEnum(enum.Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY_SUBMITTED = "query_submitted"
    DOC_VIEWED = "document_viewed"
    DOC_INDEXED = "document_indexed"
    DOC_CREATED = "document_created"
    DOC_UPDATED = "document_updated"
    DOC_DELETED = "document_deleted"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_CREATED = "role_created"
    ROLE_UPDATED = "role_updated"
    ROLE_DELETED = "role_deleted"
    PERMISSION_CREATED = "permission_created"
    PERMISSION_ASSIGNED = "permission_assigned"
    PERMISSION_REVOKED = "permission_revoked"
    PERMISSION_DENIED = "permission_denied"
    DATA_LOADED = "data_loaded"
    ADMIN_ACTION = "admin_action"
    SYSTEM_EVENT = "system_event"

# --- Association Table for Role-Permissions (Many-to-Many) ---
scm_role_permissions_table = Table(
    "scm_role_permissions",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("scm_roles.role_id"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("scm_permissions.permission_id"), primary_key=True),
)

# --- System User and Security Models ---
class SystemUser(Base):
    __tablename__ = "system_users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    assigned_region = Column(String(100), default="Global", nullable=False)
    role_id = Column(Integer, ForeignKey("scm_roles.role_id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    role = relationship("Role", back_populates="users", lazy="selectin") # Eager load role
    audit_logs = relationship("AuditLog", back_populates="user")

class Role(Base):
    __tablename__ = "scm_roles"
    role_id = Column(Integer, primary_key=True, index=True)
    role_name = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    users = relationship("SystemUser", back_populates="role")
    permissions = relationship(
        "SCMPermission",
        secondary=scm_role_permissions_table,
        back_populates="roles",
        lazy="selectin" # Eager load permissions for a role
    )

class SCMPermission(Base):
    __tablename__ = "scm_permissions"
    permission_id = Column(Integer, primary_key=True, index=True)
    permission_name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True) # e.g., 'Document', 'Database', 'Financial'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    roles = relationship(
        "Role",
        secondary=scm_role_permissions_table,
        back_populates="permissions"
    )

# --- Policy Document and Embedding Models ---
class PolicyDocument(Base):
    __tablename__ = "policy_documents"
    document_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    original_file_name = Column(String(255), nullable=True)
    storage_path = Column(String(500), nullable=True) # Path to the actual document file
    document_type = Column(String(100), default="Policy", nullable=True)
    summary = Column(Text, nullable=True)
    keywords = Column(Text, nullable=True) # Could be comma-separated or JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    embeddings = relationship("DocumentEmbedding", back_populates="policy_document", cascade="all, delete-orphan")

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    # Using a UUID for embedding_id as it's common with vector stores
    embedding_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_document_id = Column(Integer, ForeignKey("policy_documents.document_id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False) # The actual text chunk
    # The embedding_vector itself is usually stored in PGVector's own table if using Langchain's PGVector store.
    # If you want to store it redundantly or for other purposes in *this* table:
    # embedding_vector = Column(ARRAY(Numeric), nullable=True) # Or use pgvector.sqlalchemy.Vector type
    embedding_metadata = Column("embedding_metadata", JSONB, nullable=True) # Renamed from 'metadata'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    policy_document = relationship("PolicyDocument", back_populates="embeddings")

# --- Audit Log Model ---
class AuditLog(Base):
    __tablename__ = "audit_logs"
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("system_users.user_id"), nullable=True, index=True) # Nullable for system actions
    action_type = Column(SQLAlchemyEnum(ActionTypeEnum), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    query_text = Column(Text, nullable=True) # For query submissions
    accessed_resource = Column(String(255), nullable=True) # e.g., table name, document ID
    access_granted = Column(Boolean, nullable=True) # For permission checks
    details = Column(Text, nullable=True) # General details about the action
    user_role_context = Column(String(100), nullable=True) # Role of user at time of action
    user_region_context = Column(String(100), nullable=True) # Region of user at time of action

    user = relationship("SystemUser", back_populates="audit_logs")

# --- Chat Message History Model (for Langchain SQLChatMessageHistory) ---
class SCMChatMessage(Base):
    __tablename__ = "scm_chat_messages" # Matches CHAT_MESSAGE_TABLE_NAME in config
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    # The 'message' column stores the JSON blob from Langchain's BaseMessage.to_json()
    message = Column(Text, nullable=False) # Langchain stores as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (UniqueConstraint('session_id', 'created_at', name='uq_session_created_at'),)


# --- Supply Chain Management (SCM) Data Models ---
# These models represent the tables loaded from DataCoSupplyChainDataset.csv

class SCMDepartments(Base):
    __tablename__ = "scm_departments"
    # CSV Column: "Department Id" -> ORM: department_id -> DB Column: DepartmentId
    DepartmentId = Column("DepartmentId", Integer, primary_key=True, index=True) # Matches CSV header after potential mapping
    DepartmentName = Column("DepartmentName", String(255), nullable=False) # Matches CSV

    categories = relationship("SCMCategories", back_populates="department")

class SCMCategories(Base):
    __tablename__ = "scm_categories"
    # CSV Column: "Category Id" -> ORM: category_id -> DB Column: CategoryId
    CategoryId = Column("CategoryId", Integer, primary_key=True, index=True)
    CategoryName = Column("CategoryName", String(255), nullable=False)
    # CSV Column: "Department Id" -> ORM: department_id -> DB Column: DepartmentId (FK)
    DepartmentId = Column("DepartmentId", Integer, ForeignKey("scm_departments.DepartmentId"), nullable=False)

    department = relationship("SCMDepartments", back_populates="categories")
    products = relationship("SCMProducts", back_populates="category")

class SCMCustomers(Base):
    __tablename__ = "scm_customers"
    # CSV: "Customer Id" -> ORM: customer_id -> DB: CustomerId
    CustomerId = Column("CustomerId", Integer, primary_key=True, index=True)
    CustomerFname = Column("CustomerFname", String(100)) # first_name
    CustomerLname = Column("CustomerLname", String(100)) # last_name
    CustomerEmail = Column("CustomerEmail", String(255), unique=True) # email
    CustomerPassword = Column("CustomerPassword", String(255)) # password - consider hashing if this were real user data
    CustomerSegment = Column("CustomerSegment", String(100)) # segment
    CustomerStreet = Column("CustomerStreet", String(255)) # street
    CustomerCity = Column("CustomerCity", String(100)) # city
    CustomerState = Column("CustomerState", String(50)) # state
    CustomerZipcode = Column("CustomerZipcode", String(20)) # zipcode
    CustomerCountry = Column("CustomerCountry", String(100)) # country

    orders = relationship("SCMOrders", back_populates="customer")

class SCMProducts(Base):
    __tablename__ = "scm_products"
    # CSV: "Product Card Id" -> ORM: product_id -> DB: ProductCardId
    ProductCardId = Column("ProductCardId", Integer, primary_key=True, index=True)
    ProductName = Column("ProductName", String(255), nullable=False) # product_name
    ProductDescription = Column("ProductDescription", Text, nullable=True) # description
    ProductPrice = Column("ProductPrice", Numeric(10, 2), nullable=False) # price
    ProductImage = Column("ProductImage", String(500), nullable=True) # image_url
    ProductStatus = Column("ProductStatus", Integer, default=0) # status (0 or 1)
    # CSV: "Product Category Id" -> ORM: category_id -> DB: CategoryId (FK)
    CategoryId = Column("CategoryId", Integer, ForeignKey("scm_categories.CategoryId"), nullable=False)

    category = relationship("SCMCategories", back_populates="products")
    order_items = relationship("SCMOrderItems", back_populates="product")

class SCMOrders(Base):
    __tablename__ = "scm_orders"
    # CSV: "Order Id" -> ORM: order_id -> DB: OrderId
    OrderId = Column("OrderId", Integer, primary_key=True, index=True)
    # CSV: "Order Customer Id" -> ORM: customer_id -> DB: CustomerId (FK)
    CustomerId = Column("CustomerId", Integer, ForeignKey("scm_customers.CustomerId"), nullable=False)
    OrderDate = Column("OrderDate", DateTime(timezone=True), nullable=False) # order_date (DateOrders)
    ShippingDate = Column("ShippingDate", DateTime(timezone=True), nullable=True) # shipping_date (DateOrders)
    DaysForShippingReal = Column("DaysForShippingReal", Integer, nullable=True) # days_for_shipping_real
    DaysForShipmentScheduled = Column("DaysForShipmentScheduled", Integer, nullable=True) # days_for_shipment_scheduled
    DeliveryStatus = Column("DeliveryStatus", String(100), nullable=True) # delivery_status
    LateDeliveryRisk = Column("LateDeliveryRisk", Integer, nullable=True) # late_delivery_risk
    BenefitPerOrder = Column("BenefitPerOrder", Numeric(10, 2), nullable=True) # benefit_per_order
    SalesPerCustomer = Column("SalesPerCustomer", Numeric(10, 2), nullable=True) # sales_per_customer (this is order-specific)
    OrderProfitPerOrder = Column("OrderProfitPerOrder", Numeric(10, 2), nullable=True) # order_profit_per_order
    OrderCity = Column("OrderCity", String(100), nullable=True) # order_city
    OrderCountry = Column("OrderCountry", String(100), nullable=True) # order_country
    OrderRegion = Column("OrderRegion", String(100), nullable=True) # order_region
    OrderState = Column("OrderState", String(100), nullable=True) # order_state
    OrderZipcode = Column("OrderZipcode", String(20), nullable=True) # order_zipcode
    OrderStatus = Column("OrderStatus", String(50), nullable=True) # order_status
    ShippingMode = Column("ShippingMode", String(100), nullable=True) # shipping_mode
    Market = Column("Market", String(50), nullable=True) # market
    Type = Column("Type", String(50), nullable=True) # payment_type
    Latitude = Column("Latitude", Numeric(9, 6), nullable=True) # latitude
    Longitude = Column("Longitude", Numeric(9, 6), nullable=True) # longitude

    customer = relationship("SCMCustomers", back_populates="orders")
    order_items = relationship("SCMOrderItems", back_populates="order", cascade="all, delete-orphan")

class SCMOrderItems(Base):
    __tablename__ = "scm_orderitems"
    # CSV: "Order Item Id" -> ORM: order_item_id -> DB: OrderItemId
    OrderItemId = Column("OrderItemId", Integer, primary_key=True, index=True)
    # CSV: "Order Id" -> ORM: order_id -> DB: OrderId (FK)
    OrderId = Column("OrderId", Integer, ForeignKey("scm_orders.OrderId"), nullable=False)
    # CSV: "Order Item Cardprod Id" -> ORM: product_id -> DB: ProductCardId (FK)
    ProductCardId = Column("ProductCardId", Integer, ForeignKey("scm_products.ProductCardId"), nullable=False)
    OrderItemQuantity = Column("OrderItemQuantity", Integer, nullable=False) # quantity
    OrderItemProductPrice = Column("OrderItemProductPrice", Numeric(10, 2), nullable=False) # price_at_order
    OrderItemDiscount = Column("OrderItemDiscount", Numeric(10, 2), default=0.0) # discount
    OrderItemDiscountRate = Column("OrderItemDiscountRate", Numeric(5, 4), default=0.0) # discount_rate
    OrderItemProfitRatio = Column("OrderItemProfitRatio", Numeric(5, 4), nullable=True) # profit_ratio
    Sales = Column("Sales", Numeric(10, 2), default=0.0) # Sales (from CSV)
    OrderItemTotal = Column("OrderItemTotal", Numeric(10, 2), default=0.0) # total_amount

    order = relationship("SCMOrders", back_populates="order_items")
    product = relationship("SCMProducts", back_populates="order_items")

