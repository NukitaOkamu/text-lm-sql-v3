conda create --name text-lm-sql-v3 python=3.9

conda activate text-lm-sql-v3

pytonh -m pip install -r requirements.txt

export LD_LIBRARY_PATH=/opt/oracle/instantclient_23_7:$LD_LIBRARY_PATH

# database command for createng tables 
tables_ddl_2.sql

# extract all procedures and functions into subfolders
python split_save_pkg.py razvoj_db_18022025.sql out

# fill the PROCEDURE_NAME table
# TABLE_NAME table is filled in table.ddl_2.sql
python populate_rma.py

# link tables and procedues
python populate_link_table_rte.py

# fill db_routine_table_usage.description
python pop_db_routine_table_usage_description.py

# fill db_routine_table_usage.table_sql_example and db_routine_table_usage.table_sql_example_description
python pop_db_routine_table_usage_sql_example.py

# generate general table description from db_routine_table_usage for each table
python gen_table_description.py

# generate general table sql example and description of the example
python gen_table_query_example.py

python gen_table_embeddings.py

python sql_maker_flow_2.py

--

naredim embeddinge za description + sql example + sql example description za vsako tabelo

user: poda vprašanje
ai: poiščem driving table preko similarity search tma.embeddings( vektoriziran string table + ddl + description + sql example + sql example description)
ai: driving table podam + metapodatke. kličem azureopenaiapi da sestavi sql in določi če je ta tabela dovolj ali potrebuje še dodatne podatke iz FK tabel
ai: če NE kličem azureopenaiapi da sestavi končni sql, če DA poišče FK tabele
ai: poiščem FK tabele od driving tabele + metapodatke(ddl + description + sql example + sql example description). kličem azureopenaiapi da sestavi sql (ker ima zdaj dodatne FK tabele) ali potrebuje še dodatne podatke iz FK tabel (še globje FK od FK tabel)
loop
ai: če NE kličem azureopenaiapi da sestavi končni sql, če DA poišče FK tabele še globje
ai: poiščem FK tabele od FK tabele + metapodatke(ddl + description + sql example + sql example description). kličem azureopenaiapi da sestavi sql (ker ima zdaj dodatne FK tabele) ali potrebuje še dodatne podatke iz FK tabel (še globje FK od FK tabel)
end loop
ai: podan return končni SQL


---

AI Flow for SQL Generation with Embeddings and Recursive FK Handling
User Input:

The user submits a query.
Driving Table Search with Embeddings:

The AI performs a similarity search using tma.embeddings to find the most relevant driving table based on a vectorized representation of:
Table name
DDL (Data Definition Language)
Table description
SQL example
SQL example description
Initial SQL Generation Attempt:

The AI provides the driving table along with its metadata.
Calls Azure OpenAI API to:
Generate an SQL query.
Determine if the driving table alone is sufficient or if additional foreign key (FK) tables are needed.
Foreign Key Handling (If Needed):

If additional data is required:
The AI retrieves the FK tables of the driving table, including metadata:
DDL
Description
SQL example
SQL example description
Calls Azure OpenAI API again to:
Refine the SQL query with FK data.
Check if even deeper FK tables are necessary.
Recursive FK Table Processing (If Needed):

If deeper FK tables are required:
The AI retrieves FK tables of the FK tables along with metadata.
Calls Azure OpenAI API again to further refine the SQL query.
The process continues in a loop until all required data is gathered.
Final SQL Generation:

Once no further FK tables are needed, the AI generates the final SQL query using Azure OpenAI API.
Return Output:

The final SQL query is returned to the user.