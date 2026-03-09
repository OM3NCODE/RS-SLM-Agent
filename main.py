

import os
from dotenv import load_dotenv
from supabase import create_client, Client  

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def inventory_check_tool(product_name:str)->str:
    """
    Checks the current inventory levels and overhang status for a specific product.
    Call this tool when the user asks about stock, availability, or inventory overhang.

    Args:
    product_name : str   The name of the product to check inventory for (e.g., "Parle-G", "Mustard Oil", etc.)

    """

    search_term = product_name.strip().lower()

    try : 
        response = supabase.table("mock_inventory")\
            .select("quantity, overhang_status, last_restocked")\
            .ilike("product_name", f"%{search_term}%") \
            .execute()
        
        data = response.data

        if data :
            item = data[0]  # Assuming the first match is the most relevant
            answer = f"Found {product_name}: {item['quantity']} units in stock."
            return print(answer)
        else:
            print(f"Product '{product_name}' not found in the Supabase database.")
    except Exception as e:
        print(f"Database error while checking inventory: {str(e)}")
        



inventory_check_tool("Parle-G")