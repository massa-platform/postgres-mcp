"""Regression tests: the function allowlist must apply everywhere a FuncCall can hide."""

from unittest.mock import MagicMock

import pytest

from postgres_mcp.sql.safe_sql import SafeSqlDriver


@pytest.fixture
def validator():
    return SafeSqlDriver(sql_driver=MagicMock())


@pytest.mark.parametrize(
    "sql",
    [
        # FROM-clause functions: RangeFunction.functions is a tuple of (FuncCall, None) pairs
        "SELECT * FROM pg_sleep(30)",
        "SELECT * FROM pg_terminate_backend(123)",
        "SELECT * FROM pg_cancel_backend(123)",
        "SELECT * FROM pg_advisory_lock(42)",
        "SELECT * FROM set_config('statement_timeout', '0', false)",
        "SELECT * FROM query_to_xml('select 1', true, false, '')",
        "SELECT * FROM dblink_exec('x', 'y')",
        "SELECT * FROM pg_read_file('/etc/passwd')",
        "SELECT * FROM lo_import('/etc/passwd')",
        "SELECT * FROM unnest(ARRAY[pg_sleep(30)])",
        "SELECT * FROM generate_series(1, 3) g, LATERAL pg_sleep(1)",
        # VALUES lists: SelectStmt.valuesLists is a tuple of tuples
        "VALUES (pg_notify('c', 'p'))",
        "SELECT * FROM (VALUES (pg_sleep(1))) v(x)",
        # WITH ORDINALITY / ROWS FROM
        "SELECT * FROM ROWS FROM (pg_sleep(1))",
    ],
)
def test_disallowed_functions_are_rejected_in_from_and_values(validator, sql):
    with pytest.raises(ValueError):
        validator._validate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM unnest(ARRAY[1, 2, 3])",
        "SELECT * FROM generate_series(1, 3)",
        "SELECT g FROM generate_series(now() - interval '1 day', now(), interval '1 hour') g",
        "SELECT * FROM jsonb_each('{\"a\": 1}'::jsonb)",
        "SELECT date_bin('1 hour', now(), timestamp '2001-01-01') AS b",
        "SELECT * FROM regexp_split_to_table('a,b', ',')",
        "VALUES (1), (2)",
        "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS v(n, s)",
        "SELECT * FROM analytics.fact_gl f, LATERAL (SELECT count(*) FROM analytics.dim_source) d",
    ],
)
def test_allowed_functions_still_pass_in_from_and_values(validator, sql):
    validator._validate(sql)
