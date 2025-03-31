# text_to_sql.py
# This script converts natural language requests in Slovenian into Oracle SQL queries using AzureOpenAIClient 
# with gpt-4o and text-embedding-ada-002. It refactors input, identifies relevant tables via embeddings and 
# cosine similarity, selects a driver table, generates an initial implicit-syntax SELECT query, refines it 
# iteratively (up to 3 levels) with foreign key joins via cx_Oracle, and logs the final query to sql_logs/. 
# It tracks API costs, capping usage, to complete the text-to-SQL pipeline.
import cx_Oracle
import os
import numpy as np
from scipy.spatial.distance import cosine
from azure_openai_client import AzureOpenAIClient  # Import the custom client
from dotenv import load_dotenv

load_dotenv()

# Database connection details from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")

if not all([DB_USERNAME, DB_PASSWORD, DB_DSN]):
    raise ValueError("Missing required database credentials in .env file (DB_USERNAME, DB_PASSWORD, DB_DSN).")

# Global variable to track total cost across all API calls
TOTAL_COST_ACCUMULATED = 0.0

# Initialize the Azure OpenAI clients
embedding_client = AzureOpenAIClient(model_type="text-embedding-ada-002")
completion_client = AzureOpenAIClient(model_type="gpt-4o")  # Can be changed to "gpt-4o-mini" for cost savings

# Get clients and pricing info
embedding_azure_client = embedding_client.get_client()
embedding_model = embedding_client.get_model()
embedding_pricing = embedding_client.get_pricing()

completion_azure_client = completion_client.get_client()
completion_model = completion_client.get_model()
completion_pricing = completion_client.get_pricing()

def connect_to_db():
    """Connect to the Oracle database."""
    try:
        connection = cx_Oracle.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN)
        print("Connected to the database successfully.")
        return connection
    except cx_Oracle.Error as error:
        print(f"Failed to connect to database: {error}")
        raise

def get_embedding(text):
    """Get embedding for the given text using Azure OpenAI API."""
    global TOTAL_COST_ACCUMULATED
    
    response = embedding_azure_client.embeddings.create(
        model=embedding_model,
        input=text
    )
    
    embedding = response.data[0].embedding
    
    # Calculate and track costs
    input_tokens = response.usage.prompt_tokens
    input_cost = (input_tokens / 1_000_000) * embedding_pricing["input"]
    TOTAL_COST_ACCUMULATED += input_cost
    
    print(f"Embedding generated. Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    
    return embedding

def refactor_user_input(raw_input):
    """Refactor raw user input into a concise, intent-focused version using Azure OpenAI."""
    global TOTAL_COST_ACCUMULATED
    
    prompt = f"""You are an expert in natural language processing and database querying. Below is a user request in Slovenian:

    **Raw Input**: "{raw_input}"

    Analyze the request and refactor it into a concise, clear statement optimized for generating an Oracle SQL query. Focus on:
    - Identifying key entities (e.g., names of people, objects).
    - Interpreting the intent (e.g., financial, informational).
    - Rephrasing it to align with database concepts (e.g., tables like 'persons', 'transactions').
    - Preserving any specific conditions or table names mentioned.

    Return your response in the following format:
    ```
    Refined Input: <your refactored statement>
    Explanation: <brief reasoning>
    ```"""

    response = completion_azure_client.chat.completions.create(
        model=completion_model,
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content.strip()
    
    # Extract token usage and calculate costs
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    
    input_cost = (input_tokens / 1_000_000) * completion_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * completion_pricing["output"]
    total_cost = input_cost + output_cost
    
    TOTAL_COST_ACCUMULATED += total_cost
    
    print(f"\nInput Refactoring:")
    print(f"  Input Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    print(f"  Output Tokens: {output_tokens}, Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    # Parse the response
    lines = content.splitlines()
    refined_input = None
    explanation = ""
    
    for line in lines:
        if line.startswith("Refined Input:"):
            refined_input = line.replace("Refined Input:", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()
    
    if not refined_input:
        raise ValueError(f"Could not refactor input from response: {content}")
    
    print(f"Raw Input: {raw_input}")
    print(f"Refined Input: {refined_input}")
    print(f"Explanation: {explanation}")
    
    return refined_input

def find_top_3_relevant_tables(request, table_metadata):
    """Find the top 3 most relevant tables for the given request using embeddings."""
    request_embedding = get_embedding(request)
    
    # Store similarity scores with table indices
    similarities = []
    
    for i, table in enumerate(table_metadata):
        table_embedding = table['embeddings']
        
        # Ensure table_embedding is in the correct format (e.g., list or NumPy array)
        if isinstance(table_embedding, str):
            import json
            table_embedding = json.loads(table_embedding)  # Adjust parsing based on actual format
        elif isinstance(table_embedding, bytes):
            import pickle
            table_embedding = pickle.loads(table_embedding)
        
        # Calculate cosine similarity (1 - cosine distance for similarity)
        similarity = 1 - cosine(request_embedding, table_embedding)
        similarities.append((similarity, i))
    
    # Sort by similarity in descending order and take top 3
    similarities.sort(reverse=True)
    top_3_indices = [index for _, index in similarities[:3]]
    
    top_3_tables = [table_metadata[i] for i in top_3_indices]
    top_3_similarities = [similarities[i][0] for i in range(min(3, len(similarities)))]
    
    print("\nTop 3 most relevant tables:")
    for table, sim in zip(top_3_tables, top_3_similarities):
        print(f"- {table['table_name']} (Similarity: {sim:.4f})")
    
    return top_3_tables, top_3_similarities

def determine_driver_table(request, top_3_tables):
    """Use Azure OpenAI to determine the driver table from the top 3 tables."""
    global TOTAL_COST_ACCUMULATED
    
    formatted_metadata = format_table_metadata(top_3_tables)
    
    prompt = f"""You are an expert in Oracle SQL and database design. Below is metadata for the top 3 tables identified as potentially relevant to a user request:

{formatted_metadata}

Based on this metadata, determine which of these tables should be the driver table (the primary table) for generating an Oracle SQL query to answer the following request in Slovenian:
**Request**: "{request}"

- The driver table should be the central table that most directly relates to the request and likely contains the primary data needed.
- Provide a brief explanation for your choice.

Return your response in the following format:
```
Driver Table: <table_name>
Explanation: <your reasoning>
```"""

    response = completion_azure_client.chat.completions.create(
        model=completion_model,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content.strip()
    
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    
    input_cost = (input_tokens / 1_000_000) * completion_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * completion_pricing["output"]
    total_cost = input_cost + output_cost
    
    TOTAL_COST_ACCUMULATED += total_cost
    
    print(f"\nDriver Table Determination:")
    print(f"  Input Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    print(f"  Output Tokens: {output_tokens}, Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    lines = content.splitlines()
    driver_table_name = None
    explanation = ""
    
    for line in lines:
        if line.startswith("Driver Table:"):
            driver_table_name = line.replace("Driver Table:", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()
    
    for table in top_3_tables:
        if table["table_name"].upper() == driver_table_name.upper():
            print(f"\nSelected Driver Table: {table['table_name']}")
            print(f"Explanation: {explanation}")
            return table
    
    raise ValueError(f"Could not determine driver table from response: {content}")

def fetch_table_metadata(connection):
    """Fetch table metadata from the database."""
    query = """        
        select tma.name tabela
        , tma.alias
        , tma.ddl
        , tma.description
        , tma.sql_example
        , tma.sql_example_description
        , tma.embeddings
        from db_table_metadata tma
        where tma.description is not null
        and tma.sql_example is not null
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = []
        for row in cursor:
            table, alias, ddl, description, sql_example, sql_example_desc, embeddings = row
            ddl_str = ddl.read() if ddl else ""
            embeddings_str = embeddings.read() if embeddings else ""
            rows.append({
                "table_name": table,
                "table_alias": alias,
                "ddl": ddl_str,
                "table_description": description,
                "sql_example": sql_example,
                "sql_example_desc": sql_example_desc,
                "embeddings": embeddings_str
            })
        return rows
    except cx_Oracle.Error as error:
        print(f"Error fetching table metadata: {error}")
        raise
    finally:
        cursor.close()

def fetch_foreign_keys(connection, table_name):
    """Fetch foreign key relationships for a given table."""
    query = """
        SELECT
            a.table_name as parent_table,
            a.column_name as parent_column,
            a.constraint_name,
            c.table_name as child_table,
            c.column_name as child_column
        FROM
            all_cons_columns a
        JOIN
            all_constraints b ON a.constraint_name = b.constraint_name
        JOIN
            all_cons_columns c ON b.r_constraint_name = c.constraint_name
        WHERE
            b.constraint_type = 'R'
            AND a.table_name = :table_name
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query, table_name=table_name)
        
        fk_relationships = []
        for row in cursor:
            parent_table, parent_column, constraint_name, child_table, child_column = row
            fk_relationships.append({
                "parent_table": parent_table,
                "parent_column": parent_column,
                "constraint_name": constraint_name,
                "child_table": child_table,
                "child_column": child_column
            })
        return fk_relationships
    except cx_Oracle.Error as error:
        print(f"Error fetching foreign keys: {error}")
        raise
    finally:
        cursor.close()

def get_table_metadata_by_name(connection, table_name, all_tables):
    """Get metadata for a specific table by name."""
    for table in all_tables:
        if table["table_name"].upper() == table_name.upper():
            return table
    
    query = """        
        select tma.name tabela
        , tma.alias
        , tma.ddl
        , tma.description
        , tma.sql_example
        , tma.sql_example_description
        from db_table_metadata tma
        where upper(tma.name) = upper(:table_name)
        and tma.description is not null
        and tma.sql_example is not null
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query, table_name=table_name)
        row = cursor.fetchone()
        
        if row:
            table, alias, ddl, description, sql_example, sql_example_desc = row
            ddl_str = ddl.read() if ddl else ""
            return {
                "table_name": table,
                "table_alias": alias,
                "ddl": ddl_str,
                "table_description": description,
                "sql_example": sql_example,
                "sql_example_desc": sql_example_desc
            }
        return None
    except cx_Oracle.Error as error:
        print(f"Error fetching table metadata by name: {error}")
        raise
    finally:
        cursor.close()

def format_table_metadata(metadata_list):
    """Format a list of table metadata into a readable string for the LLM."""
    formatted = ""
    for metadata in metadata_list:
        formatted += f"Table: {metadata['table_name']}\n"
        formatted += f"Table alias: {metadata['table_alias']}\n"
        formatted += f"DDL:\n{metadata['ddl']}\n"
        formatted += f"General Description: {metadata['table_description']}\n"
        if metadata.get('sql_example'):
            formatted += f"SQL Example:\n{metadata['sql_example']}\n"
        if metadata.get('sql_example_desc'):
            formatted += f"SQL Example Description: {metadata['sql_example_desc']}\n"
        formatted += "\n"
    return formatted

def generate_initial_sql(request, driving_table):
    """Generate initial SQL query based on the driving table."""
    global TOTAL_COST_ACCUMULATED
    
    prompt = f"""You are an expert in Oracle SQL and database design. Below is metadata about a table that is likely the main table needed for the query:

Table: {driving_table['table_name']}
Table alias: {driving_table['table_alias']}
DDL:
{driving_table['ddl']}

General Description: {driving_table['table_description']}

SQL Example:
{driving_table['sql_example']}

SQL Example Description: {driving_table['sql_example_desc']}

Based on this metadata, generate an Oracle SQL query to answer the following request in Slovenian:
**Request**: "{request}"

Additionally, analyze if this table alone is sufficient to answer the request, or if we need to join with other tables through foreign keys.

Return your response in the following format:
```sql
-- Your generated SQL query here
```

```analysis
-- Your analysis of whether additional tables are needed
-- If additional tables are needed, specify which foreign key relationships should be explored
```"""

    response = completion_azure_client.chat.completions.create(
        model=completion_model,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content.strip()
    
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    
    input_cost = (input_tokens / 1_000_000) * completion_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * completion_pricing["output"]
    total_cost = input_cost + output_cost
    
    TOTAL_COST_ACCUMULATED += total_cost
    
    print(f"Initial SQL Generation:")
    print(f"  Input Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    print(f"  Output Tokens: {output_tokens}, Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    sql_parts = content.split("```sql")
    if len(sql_parts) > 1:
        sql_query = sql_parts[1].split("```")[0].strip()
    else:
        sql_query = ""
    
    analysis_parts = content.split("```analysis")
    if len(analysis_parts) > 1:
        analysis = analysis_parts[1].split("```")[0].strip()
    else:
        analysis = ""
    
    needs_fk = "additional tables" in analysis.lower() or "foreign key" in analysis.lower() or "join" in analysis.lower()
    
    return sql_query, analysis, needs_fk

def refine_sql_with_fk(request, current_sql, driving_table, fk_tables_metadata, depth=0, max_depth=3):
    """Refine SQL query with foreign key table information."""
    global TOTAL_COST_ACCUMULATED
    
    if depth >= max_depth:
        print(f"Reached maximum FK depth of {max_depth}")
        return current_sql, False
    
    all_tables = [driving_table] + fk_tables_metadata
    formatted_metadata = format_table_metadata(all_tables)
    
    prompt = f"""You are an expert in Oracle SQL and database design. Below is metadata about available tables including the driving table and related tables through foreign keys:

{formatted_metadata}

Current SQL query:
```sql
{current_sql}
```

Based on this expanded metadata, refine the Oracle SQL query to better answer the following request in Slovenian:
**Request**: "{request}"

- Ensure the query is syntactically correct and uses the tables and columns available in the metadata.
- Use implicit SQL syntax (e.g., list tables with commas in the FROM clause and specify join conditions in the WHERE clause) instead of explicit JOINs.
- For date types use only <, >, =, >=, <=, != and =
- Date format always 'dd.mm.yyyy' and for time 'dd.mm.yyyy hh24:mi:ss'
- Do not include schema names () before table names.
- Include appropriate JOINs through WHERE clauses based on the foreign key relationships.
- Do not hallucinate non-existing columns; use only columns available in the DDL.

Additionally, analyze if we need to expand further to include more FK tables:

Return your response in the following format:
```sql
-- Your refined SQL query here
```

```analysis
-- Your analysis of whether even more tables are needed
-- If more tables are needed, specify which foreign key relationships should be explored
```"""

    response = completion_azure_client.chat.completions.create(
        model=completion_model,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content.strip()
    
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    
    input_cost = (input_tokens / 1_000_000) * completion_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * completion_pricing["output"]
    total_cost = input_cost + output_cost
    
    TOTAL_COST_ACCUMULATED += total_cost
    
    print(f"SQL Refinement (Depth {depth}):")
    print(f"  Input Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    print(f"  Output Tokens: {output_tokens}, Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    sql_parts = content.split("```sql")
    if len(sql_parts) > 1:
        refined_sql = sql_parts[1].split("```")[0].strip()
    else:
        refined_sql = current_sql
    
    analysis_parts = content.split("```analysis")
    if len(analysis_parts) > 1:
        analysis = analysis_parts[1].split("```")[0].strip()
    else:
        analysis = ""
    
    needs_more_fk = "more tables" in analysis.lower() or "additional foreign key" in analysis.lower()
    
    return refined_sql, needs_more_fk

def generate_final_sql(request, current_sql, all_tables):
    """Generate the final, optimized SQL query."""
    global TOTAL_COST_ACCUMULATED
    
    formatted_metadata = format_table_metadata(all_tables)
    
    prompt = f"""You are an expert in Oracle SQL and database design. Below is metadata about all available tables needed for the query:

{formatted_metadata}

Current SQL query:
```sql
{current_sql}
```

Based on all the available metadata, generate the final, optimized Oracle SQL query to answer the following request in Slovenian:
**Request**: "{request}"

- Ensure the query is syntactically correct and uses the tables and columns available in the metadata.
- Use implicit SQL syntax (e.g., list tables with commas in the FROM clause and specify join conditions in the WHERE clause) instead of explicit JOINs.
- For date types use only <, >, =, >=, <=, != and =
- Date format always 'dd.mm.yyyy' and for time 'dd.mm.yyyy hh24:mi:ss'
- Do not include schema names (e.g., "INSURANCE2.") before table names.
- Include appropriate JOINs through WHERE clauses based on the foreign key relationships.
- Optimize the query for performance with appropriate indexing hints if necessary.
- Do not hallucinate non-existing columns; use only columns available in the DDL.
- Audit columns: UPOVNO is user insert, UPOSPR is user last update, DATVNO is date insert, DATSPR is date last update.
- Return only the SQL query as the output, without additional explanation.
"""

    response = completion_azure_client.chat.completions.create(
        model=completion_model,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    final_sql = response.choices[0].message.content.strip()
    
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    
    input_cost = (input_tokens / 1_000_000) * completion_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * completion_pricing["output"]
    total_cost = input_cost + output_cost
    
    TOTAL_COST_ACCUMULATED += total_cost
    
    print(f"Final SQL Generation:")
    print(f"  Input Tokens: {input_tokens}, Cost: ${input_cost:.6f}")
    print(f"  Output Tokens: {output_tokens}, Cost: ${output_cost:.6f}")
    print(f"  Total Cost: ${total_cost:.6f}")
    
    return final_sql

def process_request(connection, request, all_tables_metadata):
    """Process user request to generate SQL query with FK handling."""
    print(f"\n=== Processing Request: '{request}' ===\n")
    
    # Step 1: Refactor the raw user input
    refined_request = refactor_user_input(request)
    
    # Step 2: Find the top 3 most relevant tables using refined request
    top_3_tables, similarities = find_top_3_relevant_tables(refined_request, all_tables_metadata)
    
    # Step 3: Determine the driver table using refined request
    driving_table = determine_driver_table(refined_request, top_3_tables)
    print(f"Selected driving table: {driving_table['table_name']}")
    
    # Step 4: Generate initial SQL and analyze if FKs are needed
    initial_sql, analysis, needs_fk = generate_initial_sql(refined_request, driving_table)
    print("\nInitial SQL Query:")
    print(initial_sql)
    print("\nAnalysis:")
    print(analysis)
    
    current_sql = initial_sql
    all_tables = [driving_table]
    
    # Step 5: Recursive FK handling if needed
    if needs_fk:
        print("\nForeign key handling required. Starting recursive process...")
        
        depth = 0
        max_depth = 3
        processed_tables = {driving_table['table_name']}
        
        while needs_fk and depth < max_depth:
            new_fk_tables = []
            for table in all_tables:
                fk_relationships = fetch_foreign_keys(connection, table['table_name'])
                
                for fk in fk_relationships:
                    child_table_name = fk['child_table']
                    if child_table_name not in processed_tables:
                        child_metadata = get_table_metadata_by_name(connection, child_table_name, all_tables_metadata)
                        if child_metadata:
                            new_fk_tables.append(child_metadata)
                            processed_tables.add(child_table_name)
            
            if not new_fk_tables:
                print("No new FK tables found. Ending recursive process.")
                break
                
            print(f"\nDepth {depth + 1}: Adding {len(new_fk_tables)} new FK tables")
            for table in new_fk_tables:
                print(f"- {table['table_name']}")
                
            all_tables.extend(new_fk_tables)
            
            current_sql, needs_fk = refine_sql_with_fk(
                refined_request,
                current_sql,
                driving_table,
                all_tables[1:],
                depth,
                max_depth
            )
            
            depth += 1
            
            print(f"\nRefined SQL (Depth {depth}):")
            print(current_sql)
    
    # Step 6: Generate final optimized SQL
    final_sql = generate_final_sql(refined_request, current_sql, all_tables)
    
    return final_sql, all_tables

def log_generated_query(request, sql_query, tables_used):
    """Log the generated query for future reference and analysis."""
    table_names = [table['table_name'] for table in tables_used]
    timestamp = os.popen('date +%Y-%m-%d_%H:%M:%S').read().strip()
    
    log_dir = os.getenv("SQL_LOG_DIR")
    if not log_dir:
        raise ValueError("Missing SQL_LOG_DIR in .env file.")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/query_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Request: {request}\n")
        f.write(f"Tables Used: {', '.join(table_names)}\n")
        f.write(f"Generated SQL Query:\n{sql_query}\n")
        f.write(f"Total Cost: ${TOTAL_COST_ACCUMULATED:.6f}\n")
    
    print(f"\nQuery logged to {log_file}")

def main():
    global TOTAL_COST_ACCUMULATED
    TOTAL_COST_ACCUMULATED = 0.0
    
    connection = connect_to_db()
    
    try:
        print("Fetching all table metadata...")
        all_tables_metadata = fetch_table_metadata(connection)
        print(f"Found {len(all_tables_metadata)} tables with metadata")
        request = ""
        
        # Raw user input for testing
        # request = "Ali ima oseba Miha Novak prijavljeno kakšno škodo?"
        
        sql_query, tables_used = process_request(connection, request, all_tables_metadata)
        
        print("\n==== FINAL SQL QUERY ====")
        print(sql_query)
        print("=========================")
        
        log_generated_query(request, sql_query, tables_used)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTotal Accumulated Cost for All API Calls: ${TOTAL_COST_ACCUMULATED:.6f}")
        connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()