import sys
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
async def test_transport_argument_parsing(transport):
    """Test that all transport options are parsed and passed to run_async."""
    from postgres_mcp.server import main

    original_argv = sys.argv
    try:
        sys.argv = [
            "postgres_mcp",
            "postgresql://user:password@localhost/db",
            f"--transport={transport}",
        ]

        with (
            patch("postgres_mcp.server.DbConnPool.pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_async", AsyncMock()) as mock_run,
        ):
            await main()

            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["transport"] == transport
    finally:
        sys.argv = original_argv


@pytest.mark.asyncio
async def test_streamable_http_host_port_arguments():
    """Test that streamable-http host and port arguments are passed to run_async."""
    from postgres_mcp.server import main

    original_argv = sys.argv
    try:
        sys.argv = [
            "postgres_mcp",
            "postgresql://user:password@localhost/db",
            "--transport=streamable-http",
            "--streamable-http-host=0.0.0.0",
            "--streamable-http-port=9000",
        ]

        with (
            patch("postgres_mcp.server.DbConnPool.pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_async", AsyncMock()) as mock_run,
        ):
            await main()

            kwargs = mock_run.call_args.kwargs
            assert kwargs["transport"] == "streamable-http"
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 9000
    finally:
        sys.argv = original_argv


@pytest.mark.asyncio
async def test_sse_host_port_arguments():
    """Test that SSE host and port arguments are passed to run_async."""
    from postgres_mcp.server import main

    original_argv = sys.argv
    try:
        sys.argv = [
            "postgres_mcp",
            "postgresql://user:password@localhost/db",
            "--transport=sse",
            "--sse-host=0.0.0.0",
            "--sse-port=8080",
        ]

        with (
            patch("postgres_mcp.server.DbConnPool.pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_async", AsyncMock()) as mock_run,
        ):
            await main()

            kwargs = mock_run.call_args.kwargs
            assert kwargs["transport"] == "sse"
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 8080
    finally:
        sys.argv = original_argv


@pytest.mark.asyncio
async def test_default_transport_is_stdio():
    """Test that the default transport is stdio when not specified."""
    from postgres_mcp.server import main

    original_argv = sys.argv
    try:
        sys.argv = [
            "postgres_mcp",
            "postgresql://user:password@localhost/db",
        ]

        with (
            patch("postgres_mcp.server.DbConnPool.pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_async", AsyncMock()) as mock_run,
        ):
            await main()

            assert mock_run.call_args.kwargs["transport"] == "stdio"
    finally:
        sys.argv = original_argv