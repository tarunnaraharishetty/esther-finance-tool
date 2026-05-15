"""Tests for the ``/api/stream`` SSE endpoint.

We boot a real uvicorn server on a background thread and hit it with
a normal HTTP client. The synchronous ``TestClient`` and the
async ``httpx.ASGITransport`` both buffer streaming responses until
the body completes — for an open-ended SSE stream that never returns,
both deadlock indefinitely. A real uvicorn flushes chunks like
production, which is what we actually want to verify.

The uvicorn-in-thread overhead is ~100ms per fixture invocation —
acceptable for the four streaming-specific tests below. The REST
endpoints continue to be tested via ``TestClient`` in ``test_api.py``
where the buffering quirk doesn't apply.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController


def _free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then close.

    There's a tiny race between "we closed the socket" and "the test
    binds to it" — fine in single-test runs, fine in serial pytest.
    """
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@contextmanager
def _live_server(*, stream_interval: float | None) -> Iterator[str]:
    """Boot uvicorn on a free port, yield the base URL, tear down.

    Uses ``uvicorn.Server`` directly with ``log_level="warning"`` to
    keep test output uncluttered. The server thread is daemon so a
    test crash can't leave a stuck thread behind.
    """
    controller = MockDashboardController(
        watchlist=["AAPL", "MSFT", "NVDA"], seed=7
    )
    app = create_app(controller, stream_interval=stream_interval)
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # Disable uvicorn's signal handling — we control shutdown.
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    # Wait until the health endpoint is reachable, max ~5 seconds.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=0.3) as client:
                if client.get(f"{base_url}/api/health").status_code == 200:
                    break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=2.0)
        raise RuntimeError("uvicorn never came up within the test deadline")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Opt-in semantics — these don't require the broker, so TestClient is fine
# --------------------------------------------------------------------------- #


def test_stream_returns_503_when_broker_disabled() -> None:
    """With ``stream_interval=None`` no broker is started; the endpoint
    must refuse cleanly with 503 + actionable error body."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, stream_interval=None)
    with TestClient(app) as client:
        res = client.get("/api/stream")
    assert res.status_code == 503
    assert "streaming is disabled" in res.json()["detail"]


def test_snapshot_endpoint_unchanged_by_streaming() -> None:
    """Even with the broker running, the REST snapshot endpoint must
    still produce its own tick and return a normal payload — the
    streaming opt-in shouldn't alter request/response semantics."""
    controller = MockDashboardController(watchlist=["AAPL", "MSFT", "NVDA"], seed=7)
    app = create_app(controller, stream_interval=0.05)
    with TestClient(app) as client:
        body: dict[str, Any] = client.get("/api/snapshot").json()
    assert body["tick"] >= 1
    assert "rows" in body and len(body["rows"]) == 3


# --------------------------------------------------------------------------- #
# Wire format / live streaming — real uvicorn in a background thread
# --------------------------------------------------------------------------- #


def _collect_lines_until_data(base_url: str, *, max_lines: int = 40) -> list[str]:
    """Stream from ``/api/stream``, collecting raw lines until we have
    at least one ``data:`` line (or hit the line cap)."""
    lines: list[str] = []
    with httpx.Client(timeout=10.0) as client, client.stream(
        "GET", f"{base_url}/api/stream"
    ) as res:
        assert res.status_code == 200, res.text
        for line in res.iter_lines():
            lines.append(line)
            if line.startswith("data: "):
                break
            if len(lines) >= max_lines:
                break
    return lines


def test_stream_content_type_and_cache_headers() -> None:
    """text/event-stream is non-negotiable — EventSource clients
    refuse the response otherwise. Cache-Control: no-cache prevents
    intermediary proxies from holding events back."""
    with (
        _live_server(stream_interval=10.0) as url,
        httpx.Client(timeout=5.0) as client,
        client.stream("GET", f"{url}/api/stream") as res,
    ):
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert res.headers["cache-control"] == "no-cache"


def test_stream_first_line_is_connected_comment() -> None:
    """The very first non-blank line should be the ``: connected``
    comment — this flushes headers through reverse proxies and gives
    EventSource clients an open signal before any tick lands."""
    with (
        _live_server(stream_interval=10.0) as url,
        httpx.Client(timeout=5.0) as client,
        client.stream("GET", f"{url}/api/stream") as res,
    ):
        for raw in res.iter_lines():
            if raw == "":
                continue
            assert raw == ": connected"
            return
    pytest.fail("stream closed before any line arrived")


def test_stream_delivers_a_snapshot_event() -> None:
    """A live tick must land within a couple of broker intervals,
    formatted as ``id: <tick>\\ndata: <json>\\n\\n``. The JSON payload
    must include tick + rows fields so the client can render it.
    """
    with _live_server(stream_interval=0.05) as url:
        lines = _collect_lines_until_data(url)

    # Find the id and data lines (id precedes data in our format).
    data_lines = [line for line in lines if line.startswith("data: ")]
    id_lines = [line for line in lines if line.startswith("id: ")]
    assert data_lines, f"no data: line received; got: {lines}"
    assert id_lines, f"no id: line received; got: {lines}"

    payload = json.loads(data_lines[0][len("data: ") :])
    assert "tick" in payload
    assert "rows" in payload and isinstance(payload["rows"], list)
    assert len(payload["rows"]) == 3
    # id line numbering should match the tick on the payload.
    assert id_lines[0] == f"id: {payload['tick']}"


def test_stream_payload_shape_matches_rest_snapshot() -> None:
    """The streamed payload should be structurally identical to the
    REST ``/api/snapshot`` response — same field names, same enum
    encoding. A future web client should share one type definition
    across both surfaces.
    """
    with _live_server(stream_interval=0.05) as url, httpx.Client(timeout=5.0) as client:
        rest_keys = set(client.get(f"{url}/api/snapshot").json().keys())
        lines = _collect_lines_until_data(url)

    data_line = next(line for line in lines if line.startswith("data: "))
    stream_payload = json.loads(data_line[len("data: ") :])
    assert set(stream_payload.keys()) == rest_keys
