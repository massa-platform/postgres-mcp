"""Tests for the non-MCP POST /internal/query route used by the gateway.

The route is exercised through FastMCP's Starlette app with an in-process
ASGI transport, so the secret check, connection resolution, SafeSqlDriver
validation and JSON serialization are all tested as the gateway sees them.
Only the very bottom - SqlDriver.execute_query, the actual Postgres call -
is mocked.
"""

import base64
import uuid
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from decimal import Decimal
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

import postgres_mcp.server as server
from postgres_mcp.server import AccessMode
from postgres_mcp.server import json_safe
from postgres_mcp.sql.sql_driver import DbConnPool
from postgres_mcp.sql.sql_driver import SqlDriver

SECRET = "test-gateway-secret"
URL = "/internal/query"

Row = SqlDriver.RowResult


def rows(*cells: dict) -> list:
    return [Row(cells=c) for c in cells]


@pytest.fixture
def pools():
    return {
        "zambia_inv": MagicMock(spec=DbConnPool),
        "all_fin_inv": MagicMock(spec=DbConnPool),
    }


@pytest.fixture
def execute(pools, monkeypatch):
    """Patch the environment and the Postgres call; yield the execute mock."""
    monkeypatch.setenv(server.GATEWAY_SECRET_ENV, SECRET)
    monkeypatch.delenv(server.MAX_ROWS_ENV, raising=False)
    with (
        patch.dict("postgres_mcp.server.db_connections", pools, clear=True),
        patch("postgres_mcp.server.default_connection_name", None),
        # Even in UNRESTRICTED mode the route must stay on SafeSqlDriver.
        patch("postgres_mcp.server.current_access_mode", AccessMode.UNRESTRICTED),
        # autospec keeps the method signature, so the mock receives `self`
        # (the SqlDriver built by the route) as its first positional argument.
        patch.object(SqlDriver, "execute_query", autospec=True) as mock_exec,
    ):
        mock_exec.return_value = rows({"n": 1})
        yield mock_exec


@pytest_asyncio.fixture
async def client():
    app = server.mcp.http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://postgres-mcp") as c:
        yield c


def headers(secret=SECRET, connection="zambia_inv"):
    h = {}
    if secret is not None:
        h["X-Gateway-Secret"] = secret
    if connection is not None:
        h["X-Postgres-Connection"] = connection
    return h


# ---------------------------------------------------------------------------
# Gateway secret
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_secret_is_401(client, execute):
    r = await client.post(URL, headers=headers(secret=None), json={"sql": "SELECT 1"})
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_secret_is_401(client, execute):
    r = await client.post(URL, headers=headers(secret="nope"), json={"sql": "SELECT 1"})
    assert r.status_code == 401
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_unset_gateway_secret_env_rejects_everything(client, execute, monkeypatch):
    monkeypatch.delenv(server.GATEWAY_SECRET_ENV, raising=False)
    for provided in ("", "anything", None):
        r = await client.post(URL, headers=headers(secret=provided), json={"sql": "SELECT 1"})
        assert r.status_code == 401, provided
    execute.assert_not_called()


# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_connection_is_400_without_leaking_names(client, execute):
    r = await client.post(URL, headers=headers(connection="tanzania_inv"), json={"sql": "SELECT 1"})
    assert r.status_code == 400
    body = r.json()
    assert body == {"error": "unknown or unselected connection"}
    assert "zambia_inv" not in r.text and "all_fin_inv" not in r.text
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_missing_connection_with_several_configured_is_400(client, execute):
    r = await client.post(URL, headers=headers(connection=None), json={"sql": "SELECT 1"})
    assert r.status_code == 400
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_missing_connection_falls_back_to_unambiguous_default(client, execute):
    with (
        patch.dict("postgres_mcp.server.db_connections", {"only": MagicMock(spec=DbConnPool)}, clear=True),
        patch("postgres_mcp.server.default_connection_name", "only"),
    ):
        r = await client.post(URL, headers=headers(connection=None), json={"sql": "SELECT 1"})
    assert r.status_code == 200
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_runs_on_the_named_connection(client, execute, pools):
    r = await client.post(URL, headers=headers(connection="all_fin_inv"), json={"sql": "SELECT 1 AS n"})
    assert r.status_code == 200
    # SqlDriver.execute_query is patched on the class, so `self` is the first positional arg.
    driver_self = execute.await_args.args[0]
    assert driver_self.conn is pools["all_fin_inv"]


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,content_type",
    [
        (b"not json", "application/json"),
        (b'{"query": "SELECT 1"}', "application/json"),
        (b'{"sql": ""}', "application/json"),
        (b'{"sql": "   "}', "application/json"),
        (b'{"sql": 42}', "application/json"),
        (b"[1,2]", "application/json"),
    ],
)
async def test_bad_body_is_400(client, execute, content, content_type):
    r = await client.post(URL, headers={**headers(), "Content-Type": content_type}, content=content)
    assert r.status_code == 400
    assert "error" in r.json()
    execute.assert_not_called()


# ---------------------------------------------------------------------------
# SafeSqlDriver always applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE t (a int)",
        "SELECT 1; INSERT INTO t VALUES (1)",
        "SELECT pg_sleep(10)",
        "COPY t TO '/tmp/x'",
    ],
)
async def test_write_and_dangerous_statements_are_400_even_in_unrestricted_mode(client, execute, sql):
    r = await client.post(URL, headers=headers(), json={"sql": sql})
    assert r.status_code == 400, sql
    assert "error" in r.json()
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_select_reaches_the_driver_read_only_with_timeout(client, execute):
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT n FROM t WHERE x = 1"})
    assert r.status_code == 200
    kwargs = execute.await_args.kwargs
    assert kwargs["force_readonly"] is True
    # SafeSqlDriver prefixes the statement it forwards.
    assert "SELECT n FROM t WHERE x = 1" in execute.await_args.args[1]
    # Not part of the mock, but the driver built by the route must carry the 30 s timeout.
    driver_self = execute.await_args.args[0]
    assert isinstance(driver_self, SqlDriver)


# ---------------------------------------------------------------------------
# Response shape and serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_shape(client, execute):
    execute.return_value = rows(
        {"source_name": "ZMB", "amount": Decimal("12.5"), "posted_at": datetime(2026, 9, 1, tzinfo=timezone.utc)},
        {"source_name": "ZMB", "amount": Decimal("3"), "posted_at": None},
    )
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT 1"})
    assert r.status_code == 200
    assert r.json() == {
        "columns": ["source_name", "amount", "posted_at"],
        "rows": [
            {"source_name": "ZMB", "amount": 12.5, "posted_at": "2026-09-01T00:00:00Z"},
            {"source_name": "ZMB", "amount": 3.0, "posted_at": None},
        ],
    }
    assert "truncated" not in r.json()


@pytest.mark.asyncio
async def test_no_result_set_returns_empty(client, execute):
    execute.return_value = None
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT 1"})
    assert r.status_code == 200
    assert r.json() == {"columns": [], "rows": []}


def test_json_safe_decimal():
    assert json_safe(Decimal("12.5")) == 12.5
    assert json_safe(Decimal("0.1")) == 0.1
    assert json_safe(Decimal("1234567.89")) == 1234567.89
    assert json_safe(Decimal("-0.000001")) == -0.000001
    assert json_safe(Decimal("3")) == 3.0
    # 30 significant digits do not survive a float: keep the exact text.
    assert json_safe(Decimal("123456789012345678901234567890.123")) == "123456789012345678901234567890.123"
    assert json_safe(Decimal("0.12345678901234567890123")) == "0.12345678901234567890123"
    assert json_safe(Decimal("NaN")) == "NaN"
    assert json_safe(Decimal("Infinity")) == "Infinity"


def test_json_safe_temporal():
    assert json_safe(datetime(2026, 9, 1, 12, 30, 15, tzinfo=timezone.utc)) == "2026-09-01T12:30:15Z"
    # Offset-aware values are normalised to UTC.
    plus_two = timezone(timedelta(hours=2))
    assert json_safe(datetime(2026, 9, 1, 14, 30, tzinfo=plus_two)) == "2026-09-01T12:30:00Z"
    # Naive values are taken as UTC.
    assert json_safe(datetime(2026, 9, 1, 12, 30)) == "2026-09-01T12:30:00Z"
    assert json_safe(datetime(2026, 9, 1, 12, 30, 0, 250000, tzinfo=timezone.utc)) == "2026-09-01T12:30:00.250000Z"
    assert json_safe(date(2026, 9, 1)) == "2026-09-01T00:00:00Z"
    assert json_safe(timedelta(minutes=1, seconds=30)) == 90.0


def test_json_safe_binary_and_misc():
    assert json_safe(b"\x00\x01hello") == base64.b64encode(b"\x00\x01hello").decode()
    assert json_safe(memoryview(b"abc")) == "YWJj"
    u = uuid.uuid4()
    assert json_safe(u) == str(u)
    assert json_safe(None) is None
    assert json_safe(True) is True
    assert json_safe(float("nan")) == "nan"
    assert json_safe({"a": Decimal("1.5"), "b": [date(2026, 1, 1)]}) == {"a": 1.5, "b": ["2026-01-01T00:00:00Z"]}
    assert json_safe(("x", 1)) == ["x", 1]


# ---------------------------------------------------------------------------
# Row cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_cap_truncates_and_flags(client, execute, monkeypatch):
    monkeypatch.setenv(server.MAX_ROWS_ENV, "2")
    execute.return_value = rows({"n": 1}, {"n": 2}, {"n": 3})
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT 1"})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == [{"n": 1}, {"n": 2}]
    assert body["truncated"] is True


@pytest.mark.asyncio
async def test_row_cap_not_flagged_at_exactly_the_cap(client, execute, monkeypatch):
    monkeypatch.setenv(server.MAX_ROWS_ENV, "2")
    execute.return_value = rows({"n": 1}, {"n": 2})
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT 1"})
    assert r.json()["rows"] == [{"n": 1}, {"n": 2}]
    assert "truncated" not in r.json()


def test_row_cap_env_parsing(monkeypatch):
    monkeypatch.delenv(server.MAX_ROWS_ENV, raising=False)
    assert server.internal_query_max_rows() == server.DEFAULT_MAX_ROWS
    monkeypatch.setenv(server.MAX_ROWS_ENV, "50")
    assert server.internal_query_max_rows() == 50
    monkeypatch.setenv(server.MAX_ROWS_ENV, "0")
    assert server.internal_query_max_rows() == server.DEFAULT_MAX_ROWS
    monkeypatch.setenv(server.MAX_ROWS_ENV, "lots")
    assert server.internal_query_max_rows() == server.DEFAULT_MAX_ROWS


# ---------------------------------------------------------------------------
# Errors from Postgres
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_error_is_400_with_obfuscated_message(client, execute):
    execute.side_effect = RuntimeError('connection to postgresql://p_zambia_inv:s3cr3t@db:5432/meta failed: relation "x" does not exist')
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT * FROM x"})
    assert r.status_code == 400
    body = r.json()
    assert "s3cr3t" not in body["error"]
    assert "****" in body["error"]
    assert 'relation "x" does not exist' in body["error"]


@pytest.mark.asyncio
async def test_permission_denied_surfaces_as_400(client, execute):
    execute.side_effect = Exception("permission denied for table fact_gl")
    r = await client.post(URL, headers=headers(), json={"sql": "SELECT * FROM analytics.fact_gl"})
    assert r.status_code == 400
    assert r.json() == {"error": "permission denied for table fact_gl"}


# ---------------------------------------------------------------------------
# The route is not an MCP tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_query_is_not_exposed_as_a_tool():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names, "tool listing should not be empty"
    assert "internal_query" not in names
