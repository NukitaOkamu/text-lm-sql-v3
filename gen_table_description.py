# gen_table_description.py
# This script generates general Slovenian descriptions (2-4 sentences) for tables by synthesizing 
# routine-specific metadata, using AzureOpenAIClient with gpt-4o-mini. It fetches table DDL and 
# existing descriptions from DB_ROUTINE_TABLE_USAGE and DB_TABLE_METADATA via cx_Oracle, processes 
# them in batches of 50, and updates the description field in DB_TABLE_METADATA. It tracks API costs 
# to enrich table metadata for the text-to-SQL pipeline.
import cx_Oracle
import os
from dotenv import load_dotenv
from azure_openai_client import AzureOpenAIClient  # Import the custom client

load_dotenv()

# Database connection details from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")

if not all([DB_USERNAME, DB_PASSWORD, DB_DSN]):
    raise ValueError("Missing required database credentials in .env file (DB_USERNAME, DB_PASSWORD, DB_DSN).")

# Global variable to track total cost across all generate_general_sql_example calls
TOTAL_COST_ACCUMULATED = 0.0

# Prompt template for generating a general description, including DDL
prompt_template = """You are an expert in Oracle PL/SQL and database design. I have a table called "{table_name}" with the following DDL definition:

{ddl}

Additionally, here are existing descriptions from various procedures/functions:

{existing_descriptions}

Based on the DDL and these descriptions, write a concise general description of what the table "{table_name}" is used for. Summarize its purpose, the type of data it stores, and how it is typically manipulated (e.g., inserted, updated, queried). Keep the description short (2-4 sentences) and technical, suitable for database documentation. Write the description in Slovenian."""

# Initialize the Azure OpenAI client (choose model here)
# azure_client = AzureOpenAIClient(model_type="gpt-4o")  # or "gpt-4o-mini"
azure_client = AzureOpenAIClient(model_type="gpt-4o-mini")  # or "gpt-4o-mini"
client = azure_client.get_client()
model = azure_client.get_model()
pricing = azure_client.get_pricing()

def connect_to_db():
    """Connect to the Oracle database."""
    try:
        connection = cx_Oracle.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN)
        print("Connected to the database successfully.")
        return connection
    except cx_Oracle.Error as error:
        print(f"Failed to connect to database: {error}")
        raise

def fetch_all_tables(connection):
    """Fetch all tables that have SQL examples and DDL."""
    query = """
        select distinct tma.name table_name
        from db_routine_table_usage rte
        , db_table_metadata tma 
        where rte.tma_id = tma.id
        and tma.description is null
        and rte.table_description is not null
        and rte.table_sql_example is not null
        and rte.table_sql_example_description is not null
        and tma.ddl is not null  -- Ensure DDL exists
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    except cx_Oracle.Error as error:
        print(f"Error fetching tables: {error}")
        raise
    finally:
        cursor.close()

def fetch_existing_sql_examples(connection, table_name):
    """Fetch existing SQL examples, descriptions, and DDL for the specified table."""
    query = """
        select tma.name tabela
        , tma.ddl
        , rte.table_description
        , rte.table_sql_example
        , rte.table_sql_example_description
        from db_routine_table_usage rte
        , db_table_metadata tma
        where rte.tma_id = tma.id
        and tma.description is null
        and rte.table_description is not null
        and rte.table_sql_example is not null
        and rte.table_sql_example_description is not null
        and tma.ddl is not null  -- Ensure DDL exists
        and tma.name = :table_name
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query, {"table_name": table_name})
        rows = []
        ddl = None
        for row in cursor:
            table, ddl_data, description, sql_example, sql_description = row
            if not ddl:  # Store DDL only once (assuming it’s the same for all rows)
                ddl = ddl_data
            rows.append({
                "table": table,
                "description": description,
                "sql_example": sql_example,
                "sql_description": sql_description if sql_description else "No description available"
            })
        return rows, ddl
    except cx_Oracle.Error as error:
        print(f"Error fetching SQL examples and DDL: {error}")
        raise
    finally:
        cursor.close()

def generate_general_description(table_name, rows, ddl):
    """Generate a general description for a table by processing in batches, including DDL."""
    global TOTAL_COST_ACCUMULATED
    batch_size = 50  # Adjust this based on experimentation (e.g., 50-100 rows per batch)
    all_descriptions = []

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        existing_descriptions = "\n".join(
            [f"- Table description: {row['description']}\n SQL Example: {row['sql_example']}\n  Description: {row['sql_description']}"
             for row in batch_rows]
        )
        prompt = prompt_template.format(table_name=table_name, ddl=ddl, existing_descriptions=existing_descriptions)

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            description = response.choices[0].message.content.strip()
            all_descriptions.append(description)

            # Extract token usage from the response
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate costs based on the selected model's pricing
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost

            # Accumulate the total cost
            TOTAL_COST_ACCUMULATED += total_cost

            # Print token usage and cost for this batch
            print(f"Token Usage for {table_name} (Batch {i//batch_size + 1}, Model: {model}):")
            print(f"  Input Tokens: {input_tokens}")
            print(f"  Output Tokens: {output_tokens}")
            print(f"  Total Tokens: {total_tokens}")
            print(f"Cost Breakdown (Pricing: Input=${pricing['input']}/M, Output=${pricing['output']}/M):")
            print(f"  Input Cost: ${input_cost:.6f}")
            print(f"  Output Cost: ${output_cost:.6f}")
            print(f"  Total Cost: ${total_cost:.6f}")

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1} for {table_name}: {e}")
            continue

    # Combine all batch descriptions (e.g., take the first one or concatenate)
    if all_descriptions:
        final_description = "\n".join(all_descriptions)  # Simple concatenation; adjust as needed
    else:
        final_description = "Failed to generate description due to errors in all batches."
    
    return final_description

def update_general_sql_example(connection, table_name, description):
    """Update the db_table_metadata table with the new general description."""
    update_query = """
        update db_table_metadata tma
        set tma.description = :description
        where tma.name = :table_name        
    """
    try:
        cursor = connection.cursor()
        cursor.execute(update_query, {
            "description": description[:4000],  # Truncate to fit VARCHAR2(4000)
            "table_name": table_name
        })
        connection.commit()
        print(f"Updated general table description for table {table_name}")
    except cx_Oracle.Error as error:
        print(f"Error updating database: {error}")
        connection.rollback()
        raise
    finally:
        cursor.close()

def process_table(connection, table_name):
    """Process a single table: fetch, generate, and update table description."""
    rows, ddl = fetch_existing_sql_examples(connection, table_name)
    if not rows or not ddl:
        print(f"No table description or DDL found for table {table_name}")
        return
    
    print(f"Found {len(rows)} table descriptions and DDL for table {table_name}")
    description = generate_general_description(table_name, rows, ddl)
    if description:
        print(f"Generated general table description for {table_name}:")
        print(f"Description: {description}")
        update_general_sql_example(connection, table_name, description)
    else:
        print(f"Failed to generate general table description for {table_name}")

def main():
    table_name = ''  # Empty string means process all tables
    
    # Step 1: Connect to the database
    connection = connect_to_db()
    
    try:
        if not table_name:
            # Fetch all tables that need SQL examples
            tables = fetch_all_tables(connection)
            if not tables:
                print("No tables found needing SQL examples.")
                return
            
            print(f"Found {len(tables)} tables to process: {tables}")
            for table in tables:
                print(f"\nProcessing table: {table}")
                process_table(connection, table)
        else:
            # Process a single specified table
            process_table(connection, table_name)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Print the accumulated total cost before closing the connection
        print(f"\nTotal Accumulated Cost for All API Calls: ${TOTAL_COST_ACCUMULATED:.6f}")
        connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()