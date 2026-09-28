"""A statement error must not tear down the tenant's pool; a connection error must."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import psycopg
import pytest

from postgres_mcp.sql.sql_driver import DbConnPool
from postgres_mcp.sql.sql_driver import SqlDriver


def make_pool(execute_side_effect):
    pool = DbConnPool("postgresql://u:p@h/db")
    pool._is_valid = True
    fake_pool = MagicMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=execute_side_effect)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.rollback = AsyncMock()

    @asynccontextmanager
    async def connection():
        yield conn

    fake_pool.connection = connection
    pool.pool = fake_pool
    return pool


@pytest.mark.asyncio
async def test_statement_error_keeps_pool_valid():
    pool = make_pool(psycopg.errors.InsufficientPrivilege("permission denied for table x"))
    driver = SqlDriver(conn=pool)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await driver.execute_query("SELECT * FROM x", force_readonly=True)
    assert pool.is_valid is True


@pytest.mark.asyncio
async def test_connection_error_invalidates_pool():
    pool = make_pool(psycopg.OperationalError("server closed the connection unexpectedly"))
    driver = SqlDriver(conn=pool)
    with pytest.raises(psycopg.OperationalError):
        await driver.execute_query("SELECT 1", force_readonly=True)
    assert pool.is_valid is False


@pytest.mark.asyncio
async def test_concurrent_reconnects_build_one_pool(monkeypatch):
    pool = DbConnPool("postgresql://u:p@h/db")
    built = []

    class FakeAsyncPool:
        def __init__(self, **kwargs):
            built.append(self)
            self.closed = False

        async def open(self):
            await asyncio.sleep(0.01)

        async def close(self):
            self.closed = True

        @asynccontextmanager
        async def connection(self):
            cur = MagicMock()
            cur.execute = AsyncMock()
            cur.__aenter__ = AsyncMock(return_value=cur)
            cur.__aexit__ = AsyncMock(return_value=False)
            conn = MagicMock()
            conn.cursor = MagicMock(return_value=cur)
            yield conn

    monkeypatch.setattr("postgres_mcp.sql.sql_driver.AsyncConnectionPool", FakeAsyncPool)
    results = await asyncio.gather(*(pool.pool_connect() for _ in range(5)))
    assert len(built) == 1
    assert all(r is built[0] for r in results)
    assert not built[0].closed
