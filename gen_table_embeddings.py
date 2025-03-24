# gen_table_embeddings.py
# This script generates vector embeddings for tables using AzureOpenAIClient with text-embedding-ada-002. 
# It fetches table metadata (name, alias, description) from DB_TABLE_METADATA via cx_Oracle, combines 
# them into text, creates embeddings, and stores them as JSON in the embeddings field. It tracks API 
# costs and enhances table metadata for similarity-based matching in the text-to-SQL pipeline.
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

# Initialize the Azure OpenAI client for embeddings
embedding_client = AzureOpenAIClient(model_type="text-embedding-ada-002")

def connect_to_db():
    """Connect to the Oracle database."""
    try:
        connection = cx_Oracle.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN)
        print("Connected to the database successfully.")
        return connection
    except cx_Oracle.Error as error:
        print(f"Failed to connect to database: {error}")
        raise

def fetch_table_metadata(connection):
    """Fetch table metadata for embedding generation."""
    query = """
        select tma.name
        , tma.alias
        , tma.ddl
        , tma.description
        , tma.sql_example
        , tma.sql_example_description
        from db_table_metadata tma
        where tma.ddl is not null
        and tma.description is not null
        and tma.sql_example is not null
        and tma.sql_example_description is not null
        and tma.embeddings is null
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = [
            {
                "name": row[0],
                "alias": row[1],
                "ddl": row[2],
                "description": row[3],
                "sql_example": row[4],
                "sql_example_description": row[5]
            }
            for row in cursor.fetchall()
        ]
        return rows
    except cx_Oracle.Error as error:
        print(f"Error fetching table metadata: {error}")
        raise
    finally:
        cursor.close()

def generate_embedding(text):
    """Generate an embedding for the provided text using Azure OpenAI."""
    try:
        embedding = embedding_client.get_embedding(text)
        cost = embedding_client.calculate_cost()
        print(f"Embedding generated. Length: {len(embedding)}, Cost: ${cost:.6f}")
        return embedding, cost
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None, 0.0

def update_embedding(connection, table_name, embedding):
    """Update the db_table_metadata table with the embedding."""
    update_query = """
        update db_table_metadata
        set embeddings = :embedding
        where name = :table_name
    """
    try:
        cursor = connection.cursor()
        # Convert embedding list to JSON string for storage in CLOB
        embedding_json = json.dumps(embedding)
        cursor.execute(update_query, {"embedding": embedding_json, "table_name": table_name})
        connection.commit()
        print(f"Updated embedding for table {table_name}")
    except cx_Oracle.Error as error:
        print(f"Error updating embedding: {error}")
        connection.rollback()
        raise
    finally:
        cursor.close()

def process_tables(connection):
    """Process all tables: fetch metadata, generate embeddings, and update."""
    total_cost = 0.0
    rows = fetch_table_metadata(connection)
    
    if not rows:
        print("No tables found matching the criteria.")
        return total_cost

    print(f"Found {len(rows)} tables to process.")
    for row in rows:
        table_name = row["name"]
        print(f"\nProcessing table: {table_name}")
        
        # Combine fields to create the text for embedding
        combined_text = (
            f"{row['name']}\n"
            f"{row['alias']}\n"
            f"{row['description']}"
        )
        
        # Generate embedding
        embedding, cost = generate_embedding(combined_text)
        total_cost += cost
        
        if embedding:
            # Update the database with the embedding
            update_embedding(connection, table_name, embedding)
        else:
            print(f"Skipping update for {table_name} due to embedding generation failure.")

    return total_cost

def main():
    # Connect to the database
    connection = connect_to_db()
    
    try:
        # Process all tables and track total cost
        total_cost = process_tables(connection)
        print(f"\nTotal Cost for All Embeddings: ${total_cost:.6f}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()