# gen_table_query_example.py
# This script generates general SELECT-only SQL query examples and Slovenian descriptions (2-3 sentences) 
# for tables, using AzureOpenAIClient with gpt-4o-mini. It fetches table metadata (DDL, alias, description) 
# and routine-specific SQL examples from DB_TABLE_METADATA and DB_ROUTINE_TABLE_USAGE via cx_Oracle, 
# creates implicit-syntax queries in batches of 50, and updates sql_example and sql_example_description 
# in DB_TABLE_METADATA. It tracks API costs to enhance table metadata for text-to-SQL.
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

# Prompt template for generating SELECT-only SQL query examples with descriptions, including DDL and alias
prompt_template = """You are an expert in Oracle PL/SQL and SQL query writing. I have a table called "{table_name}" with the following general description, DDL definition, table alias, and existing SQL examples with their descriptions from various procedures/functions:

General Table Description: {table_description}

Table DDL: 
{ddl}

Table Alias: {table_alias}

Existing SQL Examples:
{existing_sql_examples}

Based on the general table description, the DDL, the table alias, and these SQL examples with their descriptions, generate a single, concise SQL SELECT query that demonstrates the typical use of the table "{table_name}". The query must be a SELECT statement only (no INSERT, UPDATE, or DELETE). Use implicit SQL syntax (e.g., list tables with commas in the FROM clause and specify join conditions in the WHERE clause) instead of explicit JOINs (e.g., INNER JOIN, LEFT JOIN). Use the provided table alias "{table_alias}" for the table "{table_name}" in the query. Include joins with the most commonly associated tables if applicable, based on the provided examples and DDL, using appropriate aliases for those tables if evident from the examples.

Provide the SQL query followed by a delimiter "---" and then a concise description (2-3 sentences) in Slovenian of what the query does, focusing on its purpose and the data it retrieves. Format the output as follows:

```
SQL query text
---
Description in Slovenian
```

Do not include additional explanations or markers like ```sql."""

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
    """Fetch all tables that have SQL examples."""
    query = """
        select distinct tma.name table_name
        from db_routine_table_usage rte
        , db_table_metadata tma 
        where rte.tma_id = tma.id
        and tma.description is not null
        and tma.sql_example is null
        and rte.table_sql_example is not null
        and rte.table_sql_example_description is not null
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
    """Fetch existing SQL examples, their descriptions, table description, DDL, and alias for the specified table."""
    query = """
        select tma.name tabela
        , tma.description
        , tma.ddl
        , tma.alias
        , rte.table_sql_example
        , rte.table_sql_example_description
        from db_routine_table_usage rte
        , db_table_metadata tma
        where rte.tma_id = tma.id
        and tma.description is not null
        and tma.sql_example is null
        and rte.table_sql_example is not null
        and rte.table_sql_example_description is not null
        and tma.name = :table_name
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query, {"table_name": table_name})
        rows = []
        table_description = None
        table_ddl = None
        table_alias = None
        for row in cursor:
            tabela, description, ddl, alias, sql_example, sql_description = row
            if table_description is None:  # Capture table description from the first row
                table_description = description if description else "No general description available"
            if table_ddl is None:  # Capture DDL from the first row
                table_ddl = ddl if ddl else "No DDL available"
            if table_alias is None:  # Capture alias from the first row
                table_alias = alias if alias else "No alias defined"
            rows.append({
                "tabela": tabela,
                "sql_example": sql_example,
                "sql_description": sql_description if sql_description else "No description available"
            })
        return rows, table_description, table_ddl, table_alias
    except cx_Oracle.Error as error:
        print(f"Error fetching SQL examples: {error}")
        raise
    finally:
        cursor.close()

def generate_general_sql_example(table_name, rows, table_description, table_ddl, table_alias):
    """Generate a general SQL example and its description by processing in batches."""
    global TOTAL_COST_ACCUMULATED
    batch_size = 50  # Adjust this based on experimentation (e.g., 50-100 rows per batch)
    all_results = []

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        existing_sql_examples = "\n".join(
            [f"- SQL Example: {row['sql_example']}\n  Description: {row['sql_description']}"
             for row in batch_rows]
        )
        prompt = prompt_template.format(
            table_name=table_name,
            table_description=table_description,
            ddl=table_ddl,
            table_alias=table_alias,
            existing_sql_examples=existing_sql_examples
        )

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip()
            all_results.append(result)

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
            print(f"\nTotal Accumulated Cost for All API Calls until now: ${TOTAL_COST_ACCUMULATED:.6f}")
            print(f"-----------------------------------------------------------------------------------")

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1} for {table_name}: {e}")
            continue

    # Combine all batch results (take the first valid result or handle multiple as needed)
    if all_results:
        # For simplicity, take the first valid result; adjust logic if multiple results need merging
        for result in all_results:
            try:
                sql_example, description = result.split("---", 1)
                sql_example = sql_example.strip()
                description = description.strip()
                return sql_example, description
            except ValueError:
                continue
        # If no valid split is found
        print(f"Error: No valid query-description pair found in batches for {table_name}")
        return all_results[0], "Opis ni na voljo zaradi napake pri generiranju."  # Fallback
    else:
        return None, "Failed to generate SQL example due to errors in all batches."

def update_general_sql_example(connection, table_name, sql_example, description):
    """Update the db_table_metadata table with the new general SQL example and description."""
    update_query = """
        update db_table_metadata tma
        set tma.sql_example = :sql_example
        , tma.sql_example_description = :sql_example_description
        where tma.name = :table_name   
    """
    try:
        cursor = connection.cursor()
        cursor.execute(update_query, {
            "sql_example": sql_example[:4000],  # Truncate to fit VARCHAR2(4000)
            "sql_example_description": description[:4000],  # Truncate to fit VARCHAR2(4000)
            "table_name": table_name
        })
        connection.commit()
        print(f"Updated general SQL example and description for table {table_name}")
    except cx_Oracle.Error as error:
        print(f"Error updating database: {error}")
        connection.rollback()
        raise
    finally:
        cursor.close()

def process_table(connection, table_name):
    """Process a single table: fetch, generate, and update SQL example and description."""
    rows, table_description, table_ddl, table_alias = fetch_existing_sql_examples(connection, table_name)
    if not rows:
        print(f"No SQL examples found for table {table_name}")
        return
    
    print(f"Found {len(rows)} SQL examples for table {table_name}")
    sql_example, description = generate_general_sql_example(table_name, rows, table_description, table_ddl, table_alias)
    if sql_example and description:
        print(f"Generated general SQL example for {table_name}: {sql_example}")
        print(f"Description: {description}")
        update_general_sql_example(connection, table_name, sql_example, description)
    else:
        print(f"Failed to generate general SQL example for {table_name}")

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