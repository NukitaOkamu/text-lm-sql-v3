# populate_pne.py
# This script populates the DB_ROUTINE_METADATA table with PL/SQL routine metadata. It traverses 
# the 'out' directory structure created by split_save_pkg.py, extracting schema, package, and 
# procedure names from the folder hierarchy and .SQL files. It connects to an Oracle database 
# using cx_Oracle and inserts each routine’s details, including its body, to centralize metadata 
# for subsequent analysis in the text-to-SQL pipeline. 
import os
import cx_Oracle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Database connection details from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")

if not all([DB_USERNAME, DB_PASSWORD, DB_DSN]):
    raise ValueError("Missing required database credentials in .env file (DB_USERNAME, DB_PASSWORD, DB_DSN).")

# Path to your 'out' folder
BASE_PATH = os.getenv("BASE_PATH")
if not BASE_PATH:
    raise ValueError("Missing BASE_PATH in .env file.")

def get_db_connection():
    try:
        connection = cx_Oracle.connect(
            user=DB_USERNAME,
            password=DB_PASSWORD,
            dsn=DB_DSN
        )
        return connection
    except cx_Oracle.Error as error:
        print(f"Error connecting to database: {error}")
        raise

def insert_procedure(connection, id_num, schema, package, procedure, body):
    try:
        cursor = connection.cursor()
        # Using bind variables for security and performance
        sql = """
            INSERT INTO DB_ROUTINE_METADATA(ID, SCHEMA, PACKAGE, PROCEDURE, PROCEDURE_BODY)
            VALUES (:1, :2, :3, :4, :5)
        """
        cursor.execute(sql, (id_num, schema, package, procedure, body))
        connection.commit()
        cursor.close()
    except cx_Oracle.Error as error:
        print(f"Error inserting data: {error}")
        raise

def process_files():
    connection = get_db_connection()
    id_counter = 1  # Starting ID
    
    try:
        # Walk through the directory structure
        for root, dirs, files in os.walk(BASE_PATH):
            # Split the path to get schema and package
            path_parts = Path(root).parts
            
            # Skip the base 'out' directory
            if len(path_parts) < 2:
                continue
                
            schema = path_parts[1]  # First level after 'out' (e.g., INSURANCE2)
            package = path_parts[2] if len(path_parts) > 2 else None  # Second level (e.g., ADMINSTR)
            
            # Process each .sql file
            for file in files:
                if file.endswith('.SQL'):
                    procedure = file[:-4]  # Remove .sql extension
                    
                    # Read the file content
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        procedure_body = f.read()
                    
                    print(f"Inserting: ID={id_counter}, Schema={schema}, Package={package}, Procedure={procedure}")
                    insert_procedure(connection, id_counter, schema, package, procedure, procedure_body)
                    id_counter += 1
                    
        print(f"Successfully inserted {id_counter - 1} records")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    # Verify the base path exists
    if not os.path.exists(BASE_PATH):
        print(f"Directory {BASE_PATH} does not exist")
    else:
        process_files()