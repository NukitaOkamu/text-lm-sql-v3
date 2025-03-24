# populate_link_table_rte.py
# This script links tables to PL/SQL routines by analyzing table usage frequency in procedure bodies. 
# It connects to an Oracle database via cx_Oracle, fetches table names from DB_TABLE_METADATA and 
# routine bodies from DB_ROUTINE_METADATA, and uses regex to count occurrences of table references 
# (e.g., FROM, JOIN). The results are inserted into DB_ROUTINE_TABLE_USAGE, quantifying table 
# importance for later SQL generation in the text-to-SQL pipeline.
import cx_Oracle
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection details from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")
LIB_DIR = os.getenv("ORACLE_CLIENT_LIB_DIR")

if not all([DB_USERNAME, DB_PASSWORD, DB_DSN]):
    raise ValueError("Missing required database credentials in .env file (DB_USERNAME, DB_PASSWORD, DB_DSN).")

if not LIB_DIR:
    raise ValueError("Missing ORACLE_CLIENT_LIB_DIR in .env file.")

def get_db_connection():
    try:
        cx_Oracle.init_oracle_client(lib_dir=LIB_DIR)
        connection = cx_Oracle.connect(
            user=DB_USERNAME,
            password=DB_PASSWORD,
            dsn=DB_DSN
        )
        return connection
    except cx_Oracle.Error as error:
        print(f"Error connecting to database: {error}")
        raise

def fetch_tables(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, name FROM DB_TABLE_METADATA")
        tables = {row[1].upper(): row[0] for row in cursor.fetchall()}  # {table_name: id}
        cursor.close()
        return tables
    except cx_Oracle.Error as error:
        print(f"Error fetching tables: {error}")
        raise

def fetch_procedures(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, PROCEDURE_BODY FROM DB_ROUTINE_METADATA")
        procedures = []
        for row in cursor.fetchall():
            proc_id, proc_body = row
            # Convert CLOB to string, handle None
            proc_body_str = proc_body.read() if proc_body is not None else ''
            procedures.append((proc_id, proc_body_str))
        cursor.close()
        return procedures
    except cx_Oracle.Error as error:
        print(f"Error fetching procedures: {error}")
        raise

def count_table_frequency(procedure_body, table_name):
    try:
        # Ensure procedure_body is a string
        if not isinstance(procedure_body, str):
            print(f"Warning: procedure_body is not a string, type={type(procedure_body)}, value={procedure_body}")
            return 0
        # Simple regex to find table name in SQL (case-insensitive)
        pattern = r'\b(FROM|JOIN|INTO|UPDATE|DELETE|INSERT INTO)\s+' + re.escape(table_name) + r'\b'
        matches = re.findall(pattern, procedure_body, re.IGNORECASE)
        return len(matches)
    except TypeError as e:
        print(f"TypeError in count_table_frequency: {e}, procedure_body={procedure_body}")
        return 0

def insert_frequency(connection, id_num, ntl_id, npu_id, frequency):
    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO DB_ROUTINE_TABLE_USAGE(ID, TMA_ID, RMA_ID, TABLE_FREQUENCY)
            VALUES (:1, :2, :3, :4)
        """
        cursor.execute(sql, (id_num, ntl_id, npu_id, frequency))
        connection.commit()
        cursor.close()
    except cx_Oracle.Error as error:
        print(f"Error inserting frequency: {error}")
        raise

def populate_met_ntl_npu():
    connection = get_db_connection()
    id_counter = 1  # Starting ID for met_ntl_npu
    
    try:
        # Fetch tables and procedures
        tables = fetch_tables(connection)
        procedures = fetch_procedures(connection)
        
        if not tables:
            print("No tables found in DB_TABLE_METADATA")
            return
        if not procedures:
            print("No procedures found in DB_ROUTINE_METADATA")
            return

        # Process each procedure
        for proc_id, proc_body in procedures:
            print(f"Processing procedure ID={proc_id}, body length={len(proc_body)}")
            for table_name, table_id in tables.items():
                frequency = count_table_frequency(proc_body, table_name)
                if frequency > 0:  # Only insert if the table is used
                    print(f"Inserting: ID={id_counter}, Table={table_name} (ID={table_id}), Procedure ID={proc_id}, Frequency={frequency}")
                    insert_frequency(connection, id_counter, table_id, proc_id, frequency)
                    id_counter += 1
        
        print(f"Successfully inserted {id_counter - 1} records into DB_ROUTINE_TABLE_USAGE")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    populate_met_ntl_npu()