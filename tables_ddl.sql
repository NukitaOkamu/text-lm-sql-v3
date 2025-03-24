-- create table DB_TABLE_METADATA
-- tables from schema
INSERT INTO DB_TABLE_METADATA (
  ID,
  NAME,
  CHANGE_DATE,
  ENTRY_DATE,
  USER_MODIFIED,
  USER_ENTERED,
  CHECK_DATE,
  MIN_DATE,
  MAX_DATE,
  ALIAS
);

SELECT 
  ROWNUM AS ID,                          -- Generates a unique ID starting from 1
  TABLE_NAME AS NAME,                    -- Table name from USER_TABLES
  SYSDATE AS CHANGE_DATE,                -- Current date as change date
  SYSDATE AS ENTRY_DATE,                 -- Current date as entry date
  USER AS USER_MODIFIED,                 -- Current user who modified the record
  USER AS USER_ENTERED,                  -- Current user who entered the record
  'Y' AS CHECK_DATE,                     -- Default value 'Y'
  TO_DATE('01-JAN-2000', 'DD-MON-YYYY') AS MIN_DATE, -- Default min date (adjust as needed)
  SYSDATE AS MAX_DATE,                   -- Current date as max date (adjust as needed)
  SUBSTR(TABLE_NAME, 1, 10) AS ALIAS     -- First 10 characters of table name as alias
--FROM USER_TABLES;
from all_tables where owner = 'MY_SCHEMA';

-- check data in new table DB_TABLE_METADATA
select *
from DB_TABLE_METADATA;

-- table for procedure and functions
-- populate it with populate_pne.py
-- alias rma
CREATE TABLE DB_ROUTINE_METADATA
(
  ID              NUMBER,
  SCHEMA          VARCHAR2(50),
  PACKAGE         VARCHAR2(50),
  PROCEDURE       VARCHAR2(50),
  PROCEDURE_BODY  CLOB
);

-- check data in new table DB_ROUTINE_METADATA
select *
from DB_ROUTINE_METADATA rma
order by id desc;

-- link table for procures/functions and tables within them
-- populate it with populate_link_table_rte
-- alias rte
CREATE TABLE DB_ROUTINE_TABLE_USAGE (
    ID               NUMBER PRIMARY KEY,            -- Unique identifier
    TMA_ID          NUMBER NOT NULL,               -- FK to DB_TABLE_METADATA (Table)
    RMA_ID          NUMBER NOT NULL,               -- FK to DB_ROUTINE_METADATA (Procedure/Function)
    TABLE_FREQUENCY NUMBER DEFAULT 1              -- Frequency of table usage in the routine
);

-- check data in new table DB_ROUTINEDB_ROUTINE_TABLE_USAGE_METADATA
select *
from DB_ROUTINE_TABLE_USAGE
order by TABLE_FREQUENCY desc;

-- check tables in which procedures/functions are used
select tma.name tabela
, rma.PACKAGE
, rma.PROCEDURE 
, rte.TABLE_FREQUENCY
, rma.PROCEDURE_BODY
, rte.id rte_id
, rte.TABLE_DESCRIPTION
from DB_ROUTINE_TABLE_USAGE rte
, DB_TABLE_METADATA tma 
, DB_ROUTINE_METADATA rma
where rte.tma_id = tma.id
and rte.rma_id = rma.id
order by rte.TABLE_FREQUENCY desc;

-- store ddl of tables
alter table DB_TABLE_METADATA
add ddl clob;

-- populate ddl tables 
update DB_TABLE_METADATA
set ddl = DBMS_METADATA.GET_DDL('TABLE', name,'MY_SCHEMA');
where ddl is null;

-- check data in DB_TABLE_METADATA
select *
from DB_TABLE_METADATA;

-- add column table DB_TABLE_METADATA.description
alter table DB_TABLE_METADATA
add DESCRIPTION varchar2(4000);

-- add column table TABLE_DESCRIPTION.description
alter table DB_ROUTINE_TABLE_USAGE
add TABLE_DESCRIPTION varchar2(4000);

-- add column table DB_ROUTINE_TABLE_USAGE.TABLE_SQL_EXAMPLE
alter table DB_ROUTINE_TABLE_USAGE
add TABLE_SQL_EXAMPLE varchar2(4000);

-- add column table DB_ROUTINE_TABLE_USAGE.TABLE_SQL_EXAMPLE_DESCRIPTION
alter table DB_ROUTINE_TABLE_USAGE
add TABLE_SQL_EXAMPLE_DESCRIPTION varchar2(4000);

-- add column table DB_TABLE_METADATA.sql_example
alter table DB_TABLE_METADATA
add sql_example varchar2(4000);

-- add column table DB_TABLE_METADATA.sql_example_description
alter table DB_TABLE_METADATA
add sql_example_description varchar2(4000);

-- add column table db_table_metadata.embeddings
alter table db_table_metadata
add embeddings clob;

-- after running pop_db_routine_table_usage_description.py check which rte.table_description was not successfuly filled
select tma.name table_name
, rma.package
, rma.procedure 
, rte.table_frequency
, rma.procedure_body
, rte.id rte_id
, tma.ddl
, rte.table_description
, rte.table_sql_example
, rte.table_sql_example_description
from db_routine_table_usage rte
, db_table_metadata tma 
, db_routine_metadata rma
where rte.tma_id = tma.id
and rte.rma_id = rma.id
and rte.table_description is null
and (rte.table_sql_example is null or rte.table_sql_example_description is null or rte.table_sql_example_description like 'Description unavailable due to generation error.')
order by rte.table_frequency desc;

-- after running pop_db_routine_table_usage_sql_example.py check which rte.table_sql_example was not successfully filled
select tma.name table_name
, rma.package
, rma.procedure 
, rte.table_frequency
, rma.procedure_body
, rte.id rte_id
, tma.ddl
, rte.table_description
, rte.table_sql_example
, rte.table_sql_example_description
from db_routine_table_usage rte
, db_table_metadata tma 
, db_routine_metadata rma
where rte.tma_id = tma.id
and rte.rma_id = rma.id
and rte.table_description is not null
and rte.table_sql_example is not null
--and rte.table_sql_example like '```%'
order by rte.table_frequency desc;

-- check which tables have all the datat so the tma.description can be filled with running gen_table_description.py
select distinct tma.name table_name
from db_routine_table_usage rte
, db_table_metadata tma 
where rte.tma_id = tma.id
and tma.description is null
and rte.table_description is not null
and rte.table_sql_example is not null
and rte.table_sql_example_description is not null;

-- after running gen_table_description.py and gen_table_query_example.py
-- check tables that were successfully processed (tma.description and tma.sql_example is willed)
select distinct tma.name tabela
, tma.alias
, tma.description
, tma.sql_example
, tma.sql_example_description
from db_routine_table_usage rte
, db_table_metadata tma
where rte.tma_id = tma.id
and tma.description is not null
and tma.sql_example is not null
and rte.table_description is not null
and rte.table_sql_example is not null
and rte.table_sql_example_description is not null
;

-- update db_table_metadata.sql_example so we clear the blank spaces.
update db_table_metadata
set sql_example = LTRIM(sql_example, CHR(10));
commit;

-- check all the tables with embeddings filled
select tma.name tabela
, tma.alias
, tma.description
, tma.sql_example
, tma.sql_example_description
, tma.embeddings
from db_routine_table_usage rte
, db_table_metadata tma
where rte.tma_id = tma.id
and tma.description is not null
and tma.sql_example is not null
and tma.embeddings is not null
;

