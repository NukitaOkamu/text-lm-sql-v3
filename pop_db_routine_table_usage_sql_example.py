# pop_db_routine_table_usage_sql_example.py
# This script generates SELECT-only SQL query examples and Slovenian descriptions (2-3 sentences) 
# for table usage in PL/SQL routines, using AzureOpenAIClient with gpt-4o-mini. It fetches table DDL 
# and procedure bodies from DB_ROUTINE_TABLE_USAGE, DB_TABLE_METADATA, and DB_ROUTINE_METADATA via 
# cx_Oracle, creates implicit-syntax queries, and updates table_sql_example and table_sql_example_description 
# in DB_ROUTINE_TABLE_USAGE. It tracks API costs, capping at $5, to enhance metadata for text-to-SQL.
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

# Global variable to track total cost across all generate_query_example calls
TOTAL_COST_ACCUMULATED = 0.0

# Prompt template for generating SELECT-only SQL query examples with descriptions
prompt_template = """You are an expert in Oracle PL/SQL and SQL query writing. I have a table called "{table_name}" with the following DDL definition:

```sql
{ddl}
```

And a PL/SQL procedure/function with the following code:

```sql
{procedure_body}
```

Based on the provided DDL and PL/SQL code, generate a single, concise SQL SELECT query example that demonstrates how the table "{table_name}" is typically queried in this specific procedure/function. The query must be a SELECT statement only (no INSERT, UPDATE, or DELETE). Use implicit SQL syntax (e.g., list tables with commas in the FROM clause and specify join conditions in the WHERE clause) instead of explicit JOINs (e.g., INNER JOIN, LEFT JOIN).

Provide the SQL query as plain text, followed by a delimiter "---" and then a concise description (2-3 sentences) in Slovenian of what the query does, focusing on its purpose and the data it retrieves. Format the output as follows:

SQL query text
---
Description in Slovenian

Do not wrap the SQL query or description in any code block markers like ``` or ```sql. Do not include additional explanations beyond the required description."""

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
        select tma.name table_name
        , rma.package
        , rma.procedure 
        , rte.table_frequency
        , rma.procedure_body
        , rte.id rte_id
        , tma.ddl
        from db_routine_table_usage rte
        , db_table_metadata tma 
        , db_routine_metadata rma
        where rte.tma_id = tma.id
        and rte.rma_id = rma.id
        --and tma.name like 'PFR_POSTAVKA_FR_DOKUMENTA'
        and rte.table_description is not null
        and (rte.table_sql_example is null or rte.table_sql_example_description is null or rte.table_sql_example_description like 'Description unavailable due to generation error.')
        order by rte.table_frequency desc
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
                "procedure_body": procedure_body_str,
                "rte_id": rte_id,
                "ddl": ddl_str
            })
        return rows
    except cx_Oracle.Error as error:
        print(f"Error fetching data: {error}")
        raise
    finally:
        cursor.close()

def generate_query_example(table_name, procedure_body, ddl):
    """Call Azure OpenAI API to generate a SQL query example and description."""
    global TOTAL_COST_ACCUMULATED  # Access the global variable
    prompt = prompt_template.format(table_name=table_name, procedure_body=procedure_body, ddl=ddl)
    response = client.chat.completions.create(
        model=model,  # Use the selected model
        max_tokens=4096,  # Increased to accommodate query + description
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )        
    response_text = response.choices[0].message.content.strip()
    
    # Split the response into query and description
    try:
        query_example, description = response_text.split("---", 1)
        query_example = query_example.strip()
        description = description.strip()
    except ValueError:
        print(f"Error: Could not split response into query and description for {table_name}")
        query_example = response_text  # Fallback to full text if split fails
        description = "Description unavailable due to generation error."

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

    return query_example, description

def update_query_example(connection, rte_id, query_example, description):
    """Update the met_ntl_npu table with the generated query example and description."""
    update_query = """
        UPDATE DB_ROUTINE_TABLE_USAGE
        SET table_sql_example = :query_example
        ,   table_sql_example_description = :description
        WHERE id = :rte_id
    """ 
    
    # Truncate to avoid exceeding VARCHAR2(4000) limits (adjust as needed)
    max_length = 4000  # Adjust based on actual column length
    query_example = query_example[:max_length] if query_example else ""
    description = description[:max_length] if description else ""    
    
    try:
        cursor = connection.cursor()
        cursor.execute(update_query, {
            "query_example": query_example,
            "description": description,
            "rte_id": rte_id
        })
        connection.commit()
        print(f"Updated query example and description for rte_id={rte_id}")
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
        # Step 2: Fetch data and generate query examples
        rows = fetch_data(connection)
        for row in rows:
            print(f"Processing table: {row['table']}, procedure: {row['package']}.{row['procedure']}")
            query_example, description = generate_query_example(row["table"], row["procedure_body"], row["ddl"])
            if query_example and description:
                # Step 3: Save the query example and description back to the database
                update_query_example(connection, row["rte_id"], query_example, description)
            else:
                print(f"Failed to generate query example or description for rte_id={row['rte_id']}")
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