DO $$
DECLARE
    current_schema TEXT;
BEGIN
    FOR current_schema IN
        SELECT schema_name FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
    LOOP
        EXECUTE format('GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA %I TO cmoore', current_schema);
    END LOOP;
END $$;
