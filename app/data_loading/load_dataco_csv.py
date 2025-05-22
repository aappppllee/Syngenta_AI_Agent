import pandas as pd
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, DataError
from datetime import datetime
import math # For checking nan/inf for numeric types

from app.db.database import AsyncSessionLocal # For standalone execution example
from app.models.db_models import (
    SCMDepartments, SCMCategories, SCMCustomers, SCMProducts, SCMOrders, SCMOrderItems
)
from app.config import settings # For DATAGO_CSV_FILE_PATH

logger = logging.getLogger(__name__)

# --- Data Cleaning and Conversion Helpers ---

def to_numeric_safe(series, errors='coerce', downcast=None):
    """Converts a pandas Series to numeric, handling errors and inf/-inf."""
    numeric_series = pd.to_numeric(series, errors=errors)
    # Replace inf/-inf with NaN, then let fillna handle it if needed
    numeric_series.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    if downcast:
        return numeric_series.astype(downcast, errors='ignore') # errors='ignore' for Int64 if NaNs present
    return numeric_series

def parse_date_robust(date_str):
    """Parses date strings with multiple format attempts, returns None on failure."""
    if pd.isna(date_str) or date_str == '' or str(date_str).lower() == 'nan':
        return None
    # Common formats from the dataset, try most specific first
    formats_to_try = [
        '%m/%d/%Y %H:%M', # e.g., 2/21/2018 2:21
        '%m/%d/%y %H:%M', # e.g., 2/21/18 2:21
        '%Y-%m-%d %H:%M:%S', # Standard ISO-like
    ]
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(date_str, format=fmt, errors='raise')
        except (ValueError, TypeError):
            continue
    try: # Pandas' general parser as a last resort
        dt_obj = pd.to_datetime(date_str, errors='raise')
        # Check if the parsed date is within a reasonable range if needed
        # For example, if dates are expected to be after 1900
        if dt_obj.year < 1900 or dt_obj.year > datetime.now().year + 5: # Basic sanity check
             logger.warning(f"Date '{date_str}' parsed to an unlikely year: {dt_obj.year}. Treating as invalid.")
             return None
        return dt_obj
    except (ValueError, TypeError) as e:
        logger.warning(f"Robust date parsing failed for: '{date_str}'. Error: {e}. Returning None.")
        return None

async def load_data_from_csv(db: AsyncSession, csv_file_path: str = settings.DATAGO_CSV_FILE_PATH):
    logger.info(f"Starting data loading from CSV: {csv_file_path}")
    try:
        # Specify dtype for problematic columns during read if known, or handle later
        df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False)
        logger.info(f"CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {csv_file_path}")
        return {"status": "error", "message": f"CSV file not found: {csv_file_path}"}
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=settings.DEBUG_MODE)
        return {"status": "error", "message": f"Error reading CSV: {str(e)}"}

    # --- Column Renaming Map (CSV Header -> ORM Attribute Name) ---
    column_rename_map = {
        "Type": "payment_type", # For SCMOrders
        "Days for shipping (real)": "days_for_shipping_real",
        "Days for shipment (scheduled)": "days_for_shipment_scheduled",
        "Benefit per order": "benefit_per_order",
        "Sales per customer": "sales_per_customer", # This is order-specific sales
        "Delivery Status": "delivery_status",
        "Late_delivery_risk": "late_delivery_risk",
        "Category Id": "category_id", # Used for SCMCategories.category_id
        "Category Name": "category_name",
        "Customer City": "city", # For SCMCustomers.city
        "Customer Country": "country",
        "Customer Email": "email",
        "Customer Fname": "first_name",
        "Customer Id": "customer_id",
        "Customer Lname": "last_name",
        "Customer Password": "password", # Stored as is for SCMCustomers
        "Customer Segment": "segment",
        "Customer State": "state",
        "Customer Street": "street",
        "Customer Zipcode": "zipcode",
        "Department Id": "department_id", # Used for SCMDepartments.department_id and SCMCategories.department_id (FK)
        "Department Name": "department_name",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Market": "market",
        "Order City": "order_city",
        "Order Country": "order_country",
        "Order Customer Id": "customer_id", # For SCMOrders.customer_id (FK)
        "Order Id": "order_id", # For SCMOrders.order_id and SCMOrderItems.order_id (FK)
        "Order Item Cardprod Id": "product_id", # For SCMOrderItems.product_id (FK)
        "Order Item Discount": "discount",
        "Order Item Discount Rate": "discount_rate",
        "Order Item Id": "order_item_id",
        "Order Item Product Price": "price_at_order",
        "Order Item Profit Ratio": "profit_ratio",
        "Order Item Quantity": "quantity",
        "Order Item Total": "total_amount",
        "Order Profit Per Order": "order_profit_per_order",
        "Order Region": "order_region",
        "Order State": "order_state",
        "Order Status": "order_status",
        "Order Zipcode": "order_zipcode",
        "Product Card Id": "product_id", # For SCMProducts.product_id
        "Product Category Id": "category_id", # For SCMProducts.category_id (FK)
        "Product Description": "description", # For SCMProducts.description
        "Product Image": "image_url",
        "Product Name": "product_name",
        "Product Price": "price", # For SCMProducts.price
        "Product Status": "status",
        "shipping date (DateOrders)": "shipping_date",
        "Shipping Mode": "shipping_mode",
        "order date (DateOrders)": "order_date"
        # "Sales" column in CSV is used for SCMOrderItems.sales
    }
    df.rename(columns=column_rename_map, inplace=True)
    logger.debug(f"DataFrame columns after renaming: {df.columns.tolist()}")

    # --- Data Type Conversion and Cleaning ---
    date_cols_to_convert = ['order_date', 'shipping_date']
    for col in date_cols_to_convert:
        if col in df.columns:
            logger.debug(f"Converting date column: {col}")
            df[col] = df[col].apply(parse_date_robust)
        else: logger.warning(f"Date column '{col}' not found in DataFrame after renaming.")

    # Define numeric columns based on ORM attribute names
    numeric_float_attributes = [
        'benefit_per_order', 'sales_per_customer', 'order_profit_per_order',
        'price', 'price_at_order', 'discount', 'discount_rate', 'profit_ratio',
        'sales', 'total_amount', 'latitude', 'longitude'
    ]
    for col in numeric_float_attributes:
        if col in df.columns: df[col] = to_numeric_safe(df[col])
        else: logger.warning(f"Numeric float column '{col}' not found.")
            
    numeric_int_attributes = [
        'days_for_shipping_real', 'days_for_shipment_scheduled', 'late_delivery_risk',
        'quantity', 'status' # Product Status
    ]
    for col in numeric_int_attributes:
        if col in df.columns: df[col] = to_numeric_safe(df[col]).astype('Int64') # Nullable Integer
        else: logger.warning(f"Numeric int column '{col}' not found.")

    # --- Helper for Batch Insert with ON CONFLICT DO NOTHING ---
    async def batch_insert_ignoring_conflicts(model_cls, records: PyList[Dict], pk_column_names: PyList[str]):
        if not records: return
        # Filter out records with NaN in PK columns, as they can't be inserted
        valid_records = [r for r in records if all(not pd.isna(r.get(pk)) for pk in pk_column_names)]
        if not valid_records:
            logger.warning(f"No valid records to insert for {model_cls.__tablename__} after PK NaN check.")
            return

        stmt = pg_insert(model_cls).values(valid_records)
        stmt = stmt.on_conflict_do_nothing(index_elements=pk_column_names) # DB column names for index
        try:
            await db.execute(stmt)
            logger.info(f"Executed batch insert/ignore for {len(valid_records)} records into {model_cls.__tablename__}.")
        except DataError as de: # Catch issues like incorrect data type for a column
            logger.error(f"DataError during batch insert for {model_cls.__tablename__}: {de}", exc_info=settings.DEBUG_MODE)
            # Optionally, try to insert row by row to find the problematic one for debugging
        except IntegrityError as ie: # Catch other integrity errors (though on_conflict should handle PK)
            logger.error(f"IntegrityError during batch insert for {model_cls.__tablename__}: {ie}", exc_info=settings.DEBUG_MODE)
        except Exception as e:
            logger.error(f"Generic error during batch insert for {model_cls.__tablename__}: {e}", exc_info=settings.DEBUG_MODE)


    # --- Load SCMDepartments ---
    logger.info("Processing SCMDepartments...")
    # ORM attribute: department_id, DB column (PK): DepartmentId
    departments_df = df[['department_id', 'department_name']].drop_duplicates(subset=['department_id']).dropna(subset=['department_id'])
    await batch_insert_ignoring_conflicts(SCMDepartments, departments_df.to_dict(orient='records'), ['DepartmentId'])

    # --- Load SCMCategories ---
    logger.info("Processing SCMCategories...")
    # ORM attributes: category_id, category_name, department_id (FK)
    # DB columns (PK): CategoryId
    categories_df = df[['category_id', 'category_name', 'department_id']].drop_duplicates(subset=['category_id']).dropna(subset=['category_id', 'department_id'])
    await batch_insert_ignoring_conflicts(SCMCategories, categories_df.to_dict(orient='records'), ['CategoryId'])

    # --- Load SCMCustomers ---
    logger.info("Processing SCMCustomers...")
    # ORM attribute: customer_id, DB column (PK): CustomerId
    customer_cols = ['customer_id', 'first_name', 'last_name', 'email', 'password', 'segment', 'street', 'city', 'state', 'zipcode', 'country']
    customers_df = df[customer_cols].drop_duplicates(subset=['customer_id']).dropna(subset=['customer_id'])
    await batch_insert_ignoring_conflicts(SCMCustomers, customers_df.to_dict(orient='records'), ['CustomerId'])

    # --- Load SCMProducts ---
    logger.info("Processing SCMProducts...")
    # ORM attributes: product_id, category_id (FK)
    # DB columns (PK): ProductCardId
    product_cols = ['product_id', 'product_name', 'description', 'price', 'image_url', 'status', 'category_id']
    products_df = df[product_cols].drop_duplicates(subset=['product_id']).dropna(subset=['product_id', 'category_id'])
    products_df['status'] = products_df['status'].fillna(0).astype(int) # Ensure status is int, default 0
    products_df['price'] = products_df['price'].fillna(0.0) # Ensure price has a default
    await batch_insert_ignoring_conflicts(SCMProducts, products_df.to_dict(orient='records'), ['ProductCardId'])

    # --- Load SCMOrders ---
    logger.info("Processing SCMOrders...")
    # ORM attributes: order_id, customer_id (FK)
    # DB columns (PK): OrderId
    order_cols = [
        'order_id', 'customer_id', 'order_date', 'shipping_date', 'days_for_shipping_real',
        'days_for_shipment_scheduled', 'delivery_status', 'late_delivery_risk', 'benefit_per_order',
        'sales_per_customer', 'order_profit_per_order', 'order_city', 'order_country', 'order_region',
        'order_state', 'order_zipcode', 'order_status', 'shipping_mode', 'market', 'payment_type',
        'latitude', 'longitude'
    ]
    orders_df = df[order_cols].drop_duplicates(subset=['order_id']).dropna(subset=['order_id', 'customer_id', 'order_date'])
    # Convert NaT to None for SQLAlchemy compatibility with nullable DateTime fields
    orders_df['shipping_date'] = orders_df['shipping_date'].apply(lambda x: x if pd.notnull(x) else None)
    orders_df['order_date'] = orders_df['order_date'].apply(lambda x: x if pd.notnull(x) else None) # order_date is NOT NULL though
    
    # Fill NaNs in numeric columns that are NOT NULL in DB with a default (e.g., 0) or handle appropriately
    # Example: if 'benefit_per_order' cannot be null in DB, fillna(0)
    # For now, assuming nullable numeric fields in ORM can handle NaNs from pandas (which become None)
    # If a Numeric column is NOT NULL, pandas NaN will cause issues.
    # Ensure ORM models reflect nullability correctly or clean data here.
    # Example: orders_df['benefit_per_order'] = orders_df['benefit_per_order'].fillna(0.0)

    await batch_insert_ignoring_conflicts(SCMOrders, orders_df.to_dict(orient='records'), ['OrderId'])

    # --- Load SCMOrderItems ---
    logger.info("Processing SCMOrderItems...")
    # ORM attributes: order_item_id, order_id (FK), product_id (FK)
    # DB columns (PK): OrderItemId
    order_item_cols = [
        'order_item_id', 'order_id', 'product_id', 'quantity', 'price_at_order',
        'discount', 'discount_rate', 'profit_ratio', 'sales', 'total_amount'
    ]
    order_items_df = df[order_item_cols].drop_duplicates(subset=['order_item_id']).dropna(subset=['order_item_id', 'order_id', 'product_id', 'quantity', 'price_at_order'])
    # Fill NaNs for numeric fields if they are not nullable in DB
    order_items_df['discount'] = order_items_df['discount'].fillna(0.0)
    order_items_df['discount_rate'] = order_items_df['discount_rate'].fillna(0.0)
    # profit_ratio can be null in model
    order_items_df['sales'] = order_items_df['sales'].fillna(0.0) 
    order_items_df['total_amount'] = order_items_df['total_amount'].fillna(0.0)


    await batch_insert_ignoring_conflicts(SCMOrderItems, order_items_df.to_dict(orient='records'), ['OrderItemId'])

    logger.info("Data loading process finished.")
    return {"status": "success", "message": "Data loading attempted. Check logs for details on inserted/skipped records."}

# Example for standalone execution (for testing the script)
# async def main_loader_script():
#     db = AsyncSessionLocal()
#     try:
#         from app.db.database import create_db_and_tables # Ensure tables exist
#         # await create_db_and_tables() # Usually called on app startup
        
#         # Optional: Clear SCM tables before loading for a fresh start during testing
#         # from sqlalchemy import text
#         # logger.info("Clearing SCM tables for fresh load...")
#         # await db.execute(text("DELETE FROM scm_orderitems; DELETE FROM scm_orders; DELETE FROM scm_products; DELETE FROM scm_customers; DELETE FROM scm_categories; DELETE FROM scm_departments;"))
#         # await db.commit()
#         # logger.info("SCM tables cleared.")

#         result = await load_data_from_csv(db, settings.DATAGO_CSV_FILE_PATH)
#         logger.info(f"load_data_from_csv result: {result}")
#         if result.get("status") == "success":
#             await db.commit() # Commit after successful loading
#             logger.info("Data committed to database.")
#         else:
#             await db.rollback()
#             logger.warning("Data loading had issues, transaction rolled back.")
            
#     except Exception as e:
#         await db.rollback()
#         logger.error(f"Error in main_loader_script: {e}", exc_info=True)
#     finally:
#         await db.close()

# if __name__ == "__main__":
#     import asyncio
#     # Configure logging for standalone script run
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger.info("Running load_dataco_csv.py as a standalone script...")
#     asyncio.run(main_loader_script())
