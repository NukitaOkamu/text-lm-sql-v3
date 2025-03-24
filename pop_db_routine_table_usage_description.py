# pop_db_routine_table_usage_description.py
# This script generates concise Slovenian descriptions (2-4 sentences) for table usage within PL/SQL 
# routines, using AzureOpenAIClient with gpt-4o-mini. It fetches table DDL and procedure bodies from 
# DB_ROUTINE_TABLE_USAGE, DB_TABLE_METADATA, and DB_ROUTINE_METADATA via cx_Oracle, processes them 
# with a custom prompt, and updates the table_description field in DB_ROUTINE_TABLE_USAGE. It tracks 
# API costs, capping at $5, to enrich metadata for the text-to-SQL pipeline.
import cx_Oracle
import json
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

# Global variable to track total cost across all generate_description calls
TOTAL_COST_ACCUMULATED = 0.0

# Updated Prompt template with DDL
prompt_template = """You are an expert in Oracle PL/SQL and database design. I have a table called "{table_name}" with the following DDL definition:

```sql
{ddl}
```

And a PL/SQL procedure/function with the following code:

```sql
{procedure_body}
```

Based on the provided DDL and PL/SQL code, write a concise description of what the table "{table_name}" is used for in this specific procedure/function. Focus on the table's purpose, the type of data it stores, and how it is manipulated (e.g., inserted, updated, queried). Keep the description short (2-4 sentences) and technical, suitable for database documentation. Write the description in Slovenian."""

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

def fetch_data(connection):
    """Fetch data from the database, including DDL."""
    query = """
        select tma.name
        , rma.PACKAGE
        , rma.PROCEDURE 
        , rte.TABLE_FREQUENCY
        , rma.PROCEDURE_BODY
        , rte.id rte_id
        , tma.ddl
        from DB_ROUTINE_TABLE_USAGE rte
        , DB_TABLE_METADATA tma 
        , DB_ROUTINE_METADATA rma
        where rte.tma_id = tma.id
        and rte.rma_id = rma.id
        and tma.name in ('MET_OSEBA','MET_OSEBA_ZGO')
        and rte.table_description is null
        and tma.description is null
        order by rte.TABLE_FREQUENCY desc
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = []
        for row in cursor:
            table, package, procedure, table_frequency, procedure_body, rte_id, ddl = row
            procedure_body_str = procedure_body.read() if procedure_body else ""
            ddl_str = ddl.read() if ddl else ""
            rows.append({
                "table": table,
                "package": package,
                "procedure": procedure,
                "table_frequency": table_frequency,
                "procedure_body_str": procedure_body_str,
                "rte_id": rte_id,
                "ddl": ddl_str
            })
        return rows
    except cx_Oracle.Error as error:
        print(f"Error fetching data: {error}")
        raise
    finally:
        cursor.close()

def generate_description(table_name, procedure_body, ddl):
    """Call Azure OpenAI API to generate a description with DDL and track token usage/cost."""
    global TOTAL_COST_ACCUMULATED  # Access the global variable
    prompt = prompt_template.format(table_name=table_name, procedure_body=procedure_body, ddl=ddl)
    response = client.chat.completions.create(
        model=model,  # Use the selected model
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )        
    description = response.choices[0].message.content

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

    # Print token usage and cost
    print(f"Token Usage for {table_name} (Model: {model}):")
    print(f"  Input Tokens: {input_tokens}")
    print(f"  Output Tokens: {output_tokens}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"Cost Breakdown (Pricing: Input=${pricing['input']}/M, Output=${pricing['output']}/M):")
    print(f"  Input Cost: ${input_cost:.6f}")
    print(f"  Output Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")

    return description

def update_description(connection, rte_id, description):
    """Update the DB_ROUTINE_TABLE_USAGE table with the generated description."""
    update_query = """
    UPDATE DB_ROUTINE_TABLE_USAGE
    SET table_description = :description
    WHERE id = :rte_id
    """
    try:
        cursor = connection.cursor()
        cursor.execute(update_query, {"description": description, "rte_id": rte_id})
        connection.commit()
        print(f"Updated description for rte_id={rte_id}")
    except cx_Oracle.Error as error:
        print(f"Error updating database: {error}")
        connection.rollback()
        raise
    finally:
        cursor.close()

def main():
    # Step 1: Connect to the database
    connection = connect_to_db()
    
    try:
        # Step 2: Fetch data and generate descriptions
        rows = fetch_data(connection)
        for row in rows:
            print(f"Processing table: {row['table']}, procedure: {row['package']}.{row['procedure']}")
            description = generate_description(row["table"], row["procedure_body_str"], row["ddl"])
            if description:
                # Step 3: Save the description back to the database
                update_description(connection, row["rte_id"], description)
            else:
                print(f"Failed to generate description for rte_id={row['rte_id']}")
            print(f"\nTotal Accumulated Cost for All API Calls so far: ${TOTAL_COST_ACCUMULATED:.6f}")
            
            if TOTAL_COST_ACCUMULATED > 5:
                print(f"\nTotal Accumulated Cost is greater then $1 -> EXIT!: TOTAL_COST_ACCUMULATED: ${TOTAL_COST_ACCUMULATED:.6f}")
                break            
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Print the accumulated total cost before closing the connection
        print(f"\nTotal Accumulated Cost for All API Calls: ${TOTAL_COST_ACCUMULATED:.6f}")
        connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()