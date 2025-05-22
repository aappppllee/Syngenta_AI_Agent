from pydantic import BaseModel, Field
from typing import Optional, List as PyList
from datetime import datetime, date

# These are example Pydantic schemas for the SCM data models.
# They would be used if you create API endpoints to directly expose SCM data.
# For now, the agent interacts with SCM data via NL-to-SQL.

class SCMCustomerSchema(BaseModel):
    customer_id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: str
    segment: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    
    class Config:
        from_attributes = True

class SCMDepartmentSchema(BaseModel):
    department_id: int
    department_name: str
    class Config: from_attributes = True

class SCMCategorySchema(BaseModel):
    category_id: int
    category_name: str
    department_id: int
    # department: Optional[SCMDepartmentSchema] = None # Example for nesting related models
    class Config: from_attributes = True

class SCMProductSchema(BaseModel):
    product_id: int
    product_name: str
    price: float # Assuming price is float after Numeric conversion from DB
    description: Optional[str] = None
    category_id: int
    # category: Optional[SCMCategorySchema] = None # Example for nesting
    class Config: from_attributes = True

class SCMOrderItemSchema(BaseModel):
    order_item_id: int
    order_id: int
    product_id: int
    quantity: int
    price_at_order: float # Assuming float after Numeric conversion
    discount: Optional[float] = 0.0
    sales: Optional[float] = None # Assuming float after Numeric conversion
    # product: Optional[SCMProductSchema] = None # Example for nesting
    class Config: from_attributes = True

class SCMOrderSchema(BaseModel):
    order_id: int
    customer_id: int
    order_date: datetime # Matches ORM model
    shipping_date: Optional[datetime] = None # Matches ORM model
    delivery_status: Optional[str] = None
    order_status: Optional[str] = None
    order_city: Optional[str] = None
    order_region: Optional[str] = None
    order_country: Optional[str] = None
    # customer: Optional[SCMCustomerSchema] = None # Example for nesting
    order_items: PyList[SCMOrderItemSchema] = Field(default_factory=list) # Example of nesting related items
    class Config: from_attributes = True
