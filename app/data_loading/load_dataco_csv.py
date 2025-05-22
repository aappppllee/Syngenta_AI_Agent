import pandas as pd
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, DataError
from datetime import datetime
import math # For checking nan/inf for numeric types
from typing import List as PyList, Dict, Any, Optional # Added this import

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
        # Use errors='ignore' for astype to Int64 if NaNs are present,
        # as Int64 cannot store NaN directly (it uses pd.NA)
        try:
            return numeric_series.astype(downcast)
        except (TypeError, ValueError): # Catches if downcast to int fails due to NA
             if pd.api.types.is_integer_dtype(downcast) or (isinstance(downcast, str) and 'int' in downcast.lower()):
                return numeric_series # Return as float or object if int conversion with NA fails
             else:
                raise # Re-raise for other types of conversion errors
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
        if dt_obj.year < 1900 or dt_obj.year > datetime.now().year + 5: 
             logger.warning(f"Date '{date_str}' parsed to an unlikely year: {dt_obj.year}. Treating as invalid.")
             return None
        return dt_obj
    except (ValueError, TypeError) as e:
        logger.warning(f"Robust date parsing failed for: '{date_str}'. Error: {e}. Returning None.")
        return None

async def load_data_from_csv(db: AsyncSession, csv_file_path: str = settings.DATAGO_CSV_FILE_PATH):
    logger.info(f"Starting data loading from CSV: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False)
        logger.info(f"CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {csv_file_path}")
        return {"status": "error", "message": f"CSV file not found: {csv_file_path}"}
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=settings.DEBUG_MODE)
        return {"status": "error", "message": f"Error reading CSV: {str(e)}"}

    # This map renames CSV headers to a more consistent intermediate format,
    # often matching the ORM model attribute names directly or being a step towards them.
    column_rename_map = {
        "Type": "payment_type", # Intermediate, will be mapped to SCMOrders.Type
        "Days for shipping (real)": "days_for_shipping_real", # Intermediate
        "Days for shipment (scheduled)": "days_for_shipment_scheduled", # Intermediate
        "Benefit per order": "benefit_per_order", # Intermediate
        "Sales per customer": "sales_per_customer", # Intermediate
        "Delivery Status": "delivery_status", # Intermediate
        "Late_delivery_risk": "late_delivery_risk", # Intermediate
        "Category Id": "CategoryId", 
        "Category Name": "CategoryName",
        "Customer City": "CustomerCity", 
        "Customer Country": "CustomerCountry",
        "Customer Email": "CustomerEmail",
        "Customer Fname": "CustomerFname",
        "Customer Id": "CustomerId", 
        "Customer Lname": "CustomerLname",
        "Customer Password": "CustomerPassword", 
        "Customer Segment": "CustomerSegment",
        "Customer State": "CustomerState",
        "Customer Street": "CustomerStreet",
        "Customer Zipcode": "CustomerZipcode",
        "Department Id": "DepartmentId", 
        "Department Name": "DepartmentName", 
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Market": "Market",
        "Order City": "OrderCity",
        "Order Country": "OrderCountry",
        "Order Customer Id": "OrderCustomerId", # Intermediate, will be mapped to SCMOrders.CustomerId
        "Order Id": "OrderId", 
        "Order Item Cardprod Id": "ProductCardId", # For SCMOrderItems, matches ORM
        "Order Item Discount": "OrderItemDiscount",
        "Order Item Discount Rate": "OrderItemDiscountRate",
        "Order Item Id": "OrderItemId",
        "Order Item Product Price": "OrderItemProductPrice",
        "Order Item Profit Ratio": "OrderItemProfitRatio",
        "Order Item Quantity": "OrderItemQuantity",
        "Sales": "Sales", 
        "Order Item Total": "OrderItemTotal",
        "Order Profit Per Order": "OrderProfitPerOrder", # Now directly maps to PascalCase
        "Order Region": "OrderRegion",
        "Order State": "OrderState",
        "Order Status": "OrderStatus",
        "Order Zipcode": "OrderZipcode",
        "Product Card Id": "ProductCardId_Products", # Intermediate for SCMProducts
        "Product Category Id": "ProductCategoryId", # Intermediate for SCMProducts
        "Product Description": "ProductDescription", 
        "Product Image": "ProductImage",
        "Product Name": "ProductName",
        "Product Price": "ProductPrice", 
        "Product Status": "ProductStatus",
        "shipping date (DateOrders)": "ShippingDate",
        "Shipping Mode": "ShippingMode",
        "order date (DateOrders)": "OrderDate"
    }
    df.rename(columns=column_rename_map, inplace=True)
    logger.debug(f"DataFrame columns after first rename: {df.columns.tolist()}")

    date_cols_to_convert = ['OrderDate', 'ShippingDate']
    for col in date_cols_to_convert:
        if col in df.columns:
            logger.debug(f"Converting date column: {col}")
            df[col] = df[col].apply(parse_date_robust)
        else: logger.warning(f"Date column '{col}' not found in DataFrame after renaming.")

    # Numeric attributes that are intermediate (snake_case) or already final (PascalCase)
    # These lists should use the names as they exist in `df` after the first rename
    numeric_float_attributes = [
        'benefit_per_order', 'sales_per_customer', # Intermediate snake_case
        'OrderProfitPerOrder', # Now using PascalCase from first rename
        'ProductPrice', 'OrderItemProductPrice', 'OrderItemDiscount', 'OrderItemDiscountRate', 
        'OrderItemProfitRatio', 'Sales', 'OrderItemTotal', 'Latitude', 'Longitude'
    ]
    for col in numeric_float_attributes:
        if col in df.columns: df[col] = to_numeric_safe(df[col])
        else: logger.warning(f"Numeric float column '{col}' not found in df after first rename.")
            
    numeric_int_attributes = [
        'days_for_shipping_real', 'days_for_shipment_scheduled', 'late_delivery_risk', # Intermediate snake_case
        'OrderItemQuantity', 'ProductStatus'
    ]
    for col in numeric_int_attributes:
        if col in df.columns: df[col] = to_numeric_safe(df[col], downcast='integer').astype('Int64') 
        else: logger.warning(f"Numeric int column '{col}' not found in df after first rename.")

    async def batch_insert_ignoring_conflicts(model_cls, records: PyList[Dict], pk_orm_attributes: PyList[str]):
        if not records: return
        # pk_orm_attributes are the ORM attribute names (e.g., 'DepartmentId')
        # We need to check if these attributes exist as keys in the records (which should match ORM attributes)
        valid_records = [r for r in records if all(not pd.isna(r.get(pk_attr)) for pk_attr in pk_orm_attributes)]
        if not valid_records:
            logger.warning(f"No valid records to insert for {model_cls.__tablename__} after PK NaN check (using ORM attributes: {pk_orm_attributes}).")
            return

        stmt = pg_insert(model_cls).values(valid_records)
        # index_elements should be the actual database column names
        # We get them from the ORM model attribute's mapped column
        db_pk_column_names = [getattr(model_cls, pk_attr).expression.key for pk_attr in pk_orm_attributes if hasattr(model_cls, pk_attr)]
        if not db_pk_column_names:
             logger.error(f"Could not determine DB PK column names for {model_cls.__tablename__} from ORM attributes {pk_orm_attributes}")
             return
        stmt = stmt.on_conflict_do_nothing(index_elements=db_pk_column_names)
        try:
            await db.execute(stmt)
            logger.info(f"Executed batch insert/ignore for {len(valid_records)} records into {model_cls.__tablename__}.")
        except DataError as de: 
            logger.error(f"DataError during batch insert for {model_cls.__tablename__}: {de}", exc_info=settings.DEBUG_MODE)
        except IntegrityError as ie: 
            logger.error(f"IntegrityError during batch insert for {model_cls.__tablename__}: {ie}", exc_info=settings.DEBUG_MODE)
        except Exception as e:
            logger.error(f"Generic error during batch insert for {model_cls.__tablename__}: {e}", exc_info=settings.DEBUG_MODE)


    logger.info("Processing SCMDepartments...")
    departments_df = df[['DepartmentId', 'DepartmentName']].drop_duplicates(subset=['DepartmentId']).dropna(subset=['DepartmentId'])
    await batch_insert_ignoring_conflicts(SCMDepartments, departments_df.to_dict(orient='records'), ['DepartmentId'])

    logger.info("Processing SCMCategories...")
    categories_df = df[['CategoryId', 'CategoryName', 'DepartmentId']].drop_duplicates(subset=['CategoryId']).dropna(subset=['CategoryId', 'DepartmentId'])
    await batch_insert_ignoring_conflicts(SCMCategories, categories_df.to_dict(orient='records'), ['CategoryId'])

    logger.info("Processing SCMCustomers...")
    customer_cols_for_df = ['CustomerId', 'CustomerFname', 'CustomerLname', 'CustomerEmail', 'CustomerPassword', 'CustomerSegment', 'CustomerStreet', 'CustomerCity', 'CustomerState', 'CustomerZipcode', 'CustomerCountry']
    customers_df = df[customer_cols_for_df].drop_duplicates(subset=['CustomerId']).dropna(subset=['CustomerId'])
    await batch_insert_ignoring_conflicts(SCMCustomers, customers_df.to_dict(orient='records'), ['CustomerId'])

    logger.info("Processing SCMProducts...")
    # `df` has `ProductCardId_Products` and `ProductCategoryId` from the first rename
    product_cols_from_df = ['ProductCardId_Products', 'ProductName', 'ProductDescription', 'ProductPrice', 'ProductImage', 'ProductStatus', 'ProductCategoryId']
    products_intermediate_df = df[product_cols_from_df].copy()
    # Rename to match ORM attributes for SCMProducts
    products_intermediate_df.rename(columns={'ProductCardId_Products': 'ProductCardId', 'ProductCategoryId': 'CategoryId'}, inplace=True)
    products_df_final = products_intermediate_df.drop_duplicates(subset=['ProductCardId']).dropna(subset=['ProductCardId', 'CategoryId'])
    products_df_final['ProductStatus'] = products_df_final['ProductStatus'].fillna(0).astype(int) 
    products_df_final['ProductPrice'] = products_df_final['ProductPrice'].fillna(0.0) 
    await batch_insert_ignoring_conflicts(SCMProducts, products_df_final.to_dict(orient='records'), ['ProductCardId'])

    logger.info("Processing SCMOrders...")
    # Select columns from `df` using names after the first `column_rename_map`
    order_cols_to_select_from_df = [
        'OrderId', 'OrderCustomerId', 'OrderDate', 'ShippingDate', 'days_for_shipping_real',
        'days_for_shipment_scheduled', 'delivery_status', 'late_delivery_risk', 'benefit_per_order',
        'sales_per_customer', 'OrderProfitPerOrder', # Corrected to PascalCase
        'OrderCity', 'OrderCountry', 'OrderRegion',
        'OrderState', 'OrderZipcode', 'OrderStatus', 'ShippingMode', 'Market', 'payment_type',
        'Latitude', 'Longitude'
    ]
    # Ensure all selected columns exist in df.columns
    order_cols_to_select_from_df = [col for col in order_cols_to_select_from_df if col in df.columns]
    orders_intermediate_df = df[order_cols_to_select_from_df].copy()
    
    # Map these intermediate DataFrame column names to the final ORM attribute names for SCMOrders
    orders_intermediate_df.rename(columns={
        'OrderCustomerId': 'CustomerId', 
        'payment_type': 'Type',
        'days_for_shipping_real': 'DaysForShippingReal',
        'days_for_shipment_scheduled': 'DaysForShipmentScheduled',
        'delivery_status': 'DeliveryStatus',
        'late_delivery_risk': 'LateDeliveryRisk',
        'benefit_per_order': 'BenefitPerOrder',
        'sales_per_customer': 'SalesPerCustomer'
        # OrderId, OrderDate, ShippingDate, OrderProfitPerOrder, OrderCity etc. should already match ORM attributes
        # if column_rename_map mapped them to PascalCase correctly.
    }, inplace=True)

    orders_df_final = orders_intermediate_df.drop_duplicates(subset=['OrderId']).dropna(subset=['OrderId', 'CustomerId', 'OrderDate'])
    orders_df_final['ShippingDate'] = orders_df_final['ShippingDate'].apply(lambda x: x if pd.notnull(x) else None)
    # OrderDate is NOT NULL in model, so dropna handles it.
    
    # Fill NaNs for specific numeric columns in SCMOrders using final ORM attribute names
    # These are the ORM attribute names.
    cols_to_fill_zero_orders = ['DaysForShippingReal', 'DaysForShipmentScheduled', 'LateDeliveryRisk', 
                                'BenefitPerOrder', 'SalesPerCustomer', 'OrderProfitPerOrder']
    for col in cols_to_fill_zero_orders:
        if col in orders_df_final.columns:
            # Ensure correct dtype after fillna
            is_int_col = 'Days' in col or 'Risk' in col
            orders_df_final[col] = orders_df_final[col].fillna(0)
            if is_int_col:
                orders_df_final[col] = orders_df_final[col].astype(int)
            else:
                orders_df_final[col] = orders_df_final[col].astype(float)
        else:
            logger.warning(f"Column {col} not found in orders_df_final for fillna operation.")


    await batch_insert_ignoring_conflicts(SCMOrders, orders_df_final.to_dict(orient='records'), ['OrderId'])

    logger.info("Processing SCMOrderItems...")
    # Columns from df (after first rename) for SCMOrderItems
    # `column_rename_map` maps "Order Item Cardprod Id" to "ProductCardId" (matches ORM for SCMOrderItems)
    order_item_cols_from_df = [
        'OrderItemId', 'OrderId', 'ProductCardId', 'OrderItemQuantity', 'OrderItemProductPrice',
        'OrderItemDiscount', 'OrderItemDiscountRate', 'OrderItemProfitRatio', 'Sales', 'OrderItemTotal'
    ]
    order_item_cols_from_df = [col for col in order_item_cols_from_df if col in df.columns]
    order_items_intermediate_df = df[order_item_cols_from_df].copy()
    # No further renames needed if CSV renames in column_rename_map match ORM fields for SCMOrderItems
    
    order_items_df_final = order_items_intermediate_df.drop_duplicates(subset=['OrderItemId']).dropna(
        subset=['OrderItemId', 'OrderId', 'ProductCardId', 'OrderItemQuantity', 'OrderItemProductPrice']
    )
    
    cols_to_fill_zero_items = ['OrderItemDiscount', 'OrderItemDiscountRate', 'Sales', 'OrderItemTotal']
    for col in cols_to_fill_zero_items:
        if col in order_items_df_final.columns:
            order_items_df_final[col] = order_items_df_final[col].fillna(0.0)
    if 'OrderItemProfitRatio' in order_items_df_final.columns: 
        order_items_df_final['OrderItemProfitRatio'] = order_items_df_final['OrderItemProfitRatio'].fillna(pd.NA)


    await batch_insert_ignoring_conflicts(SCMOrderItems, order_items_df_final.to_dict(orient='records'), ['OrderItemId'])

    logger.info("Data loading process finished.")
    return {"status": "success", "message": "Data loading attempted. Check logs for details on inserted/skipped records."}

