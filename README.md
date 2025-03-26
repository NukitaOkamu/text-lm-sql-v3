# Text-to-SQL: Python and LLM-Powered Query Generation

Transform natural language into precise Oracle SQL queries using Python, PL/SQL parsing, and the Azure OpenAI API. This project automates table analysis, generates rich metadata, and leverages embeddings for a robust text-to-SQL pipeline.

## Overview

This repository converts Slovenian text requests into executable SQL statements by:
- Parsing PL/SQL routines to extract metadata.
- Using Azure OpenAI (`gpt-4o`, `gpt-4o-mini`, `text-embedding-ada-002`) for descriptions, embeddings, and query generation.
- Iteratively refining queries with foreign key relationships.
- Logging outputs for analysis.

Designed for developers, it offers a framework adaptable to any Oracle database with PL/SQL routines.

## Features

- **PL/SQL Parsing**: Extracts procedures/functions from package bodies.
- **Metadata Generation**: Stores table usage, descriptions, and query examples in a database.
- **Embeddings**: Enhances table selection with vector similarity.
- **SQL Generation**: Produces implicit-syntax SELECT queries from natural language.
- **Cost Management**: Caps Azure OpenAI API usage at $5 per script.

## Prerequisites

- Python 3.9
- Oracle Database (tested with `DEVDB`)
- Azure OpenAI API key and endpoint
- Oracle Instant Client (e.g., `instantclient_23_7`)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/text-to-sql.git
   cd text-to-sql
   ```

2. **Set Up Environment**
   ```bash
   conda create --name text-to-sql python=3.9
   conda activate text-to-sql
   python -m pip install cx_Oracle openai python-dotenv tiktoken numpy scipy
   ```

3. **Configure Oracle Instant Client**
   - Download and extract [Oracle Instant Client](https://www.oracle.com/database/technologies/instant-client.html) (e.g., to `/opt/oracle/instantclient_23_7`).
   - Set the library path:
     ```bash
     export LD_LIBRARY_PATH=/opt/oracle/instantclient_23_7:$LD_LIBRARY_PATH
     ```

4. **Set Environment Variables**
   - Create a `.env` file in the root directory:
     ```
     AZURE_OPENAI_API_KEY=your-api-key
     AZURE_OPENAI_ENDPOINT=your-endpoint
     ```

5. **Database Setup**
   - Run `tables_ddl.sql` to create required tables (`DB_ROUTINE_METADATA`, `DB_TABLE_METADATA`, `DB_ROUTINE_TABLE_USAGE`).

## Usage

Run the pipeline sequentially to process PL/SQL files and generate SQL from text queries:

```bash
# Extract PL/SQL routines
python split_save_pkg.py your_plsql_file.sql out

# Populate routine metadata
python populate_rma.py

# Link tables to routines
python populate_link_table_rte.py

# Generate routine-specific descriptions
python pop_db_routine_table_usage_description.py

# Add routine-specific SQL examples
python pop_db_routine_table_usage_sql_example.py

# Synthesize general table descriptions
python gen_table_description.py

# Create general SQL query examples
python gen_table_query_example.py

# Generate table embeddings
python gen_table_embeddings.py

# Convert text to SQL
python text_to_sql.py
```

Example request in `text_to_sql.py`:
```python
request = "Ali je oseba Miha Novak kaj dolžna?"
```

Output SQL is logged to `sql_logs/`.

## Project Structure

- `split_save_pkg.py`: Parses PL/SQL into individual `.SQL` files.
- `populate_rma.py`: Loads routine metadata into the database.
- `populate_link_table_rte.py`: Links tables to routines with usage frequency.
- `pop_db_routine_table_usage_description.py`: Generates routine-specific table descriptions.
- `pop_db_routine_table_usage_sql_example.py`: Adds routine-specific SQL examples.
- `gen_table_description.py`: Synthesizes general table descriptions.
- `gen_table_query_example.py`: Creates general SQL query examples.
- `gen_table_embeddings.py`: Generates table embeddings.
- `text_to_sql.py`: Converts text to SQL with iterative refinement.
- `azure_openai_client.py`: Manages Azure OpenAI API interactions.
- `tables_ddl.sql`: Database schema setup.

## Blog Post

Learn more about the project’s design and implementation in [this blog post](https://nukitaokamu.github.io/blog/posts/2025-03-21-turning-text-into-sql.html).

## Challenges and Solutions

- **PL/SQL Parsing**: Solved with robust regex and token counting.
- **Cost Control**: Enforced $5 cap per script via `AzureOpenAIClient`.
- **Multi-Table Queries**: Handled with embeddings and foreign key refinement.

## Future Enhancements

- Support for subqueries and complex SQL constructs.
- Real-time query execution against the database.
- Multi-language request handling.

## Contributing

Contributions are welcome! Fork the repo, create a branch, and submit a pull request with your improvements. :()
