"""End-to-end tests for dynamic request batching (@cog.batch).

Each test starts a real coglet server with a batched runner and sends
independent HTTP predictions. The runner records the size of every batch it
receives so tests can assert how requests were grouped.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
from test_coglet import CogletServer, _wait_for_health_status

# =============================================================================
# Helpers
# =============================================================================


def write_runner(tmp_path: Path, body: str, decorator: str) -> Path:
    """Write a batched runner. `body` is the run() body; `inputs` is the batch."""
    record = tmp_path / "batches.jsonl"
    predictor = tmp_path / "predict.py"
    predictor.write_text(f"""
import asyncio
import json
import time

import cog
from cog import BaseModel, BaseRunner, Input

RECORD = {str(record)!r}


class Item(BaseModel):
    prompt: str
    delay: float = Input(default=0.0)
    fail: bool = False


def record(inputs):
    with open(RECORD, "a") as f:
        f.write(json.dumps([i.prompt for i in inputs]) + "\\n")


class Predictor(BaseRunner):
    {decorator}
{body}
""")
    (tmp_path / "cog.yaml").write_text('predict: "predict.py:Predictor"\n')
    return predictor


SYNC_ECHO = """    def run(self, inputs: list[Item]) -> list[str]:
        record(inputs)
        print("running batch of", len(inputs))
        time.sleep(max(i.delay for i in inputs))
        if any(i.fail for i in inputs):
            raise ValueError("batch exploded")
        return [f"{i.prompt}|n={len(inputs)}" for i in inputs]
"""

ASYNC_ECHO = """    async def run(self, inputs: list[Item]) -> list[str]:
        record(inputs)
        await asyncio.sleep(max(i.delay for i in inputs))
        return [f"{i.prompt}|n={len(inputs)}" for i in inputs]
"""


def recorded_batches(tmp_path: Path) -> list[list[str]]:
    record = tmp_path / "batches.jsonl"
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines()]


def batch_env(slots: int) -> dict[str, str]:
    return {"COG_MAX_CONCURRENCY": str(slots)}


def post(server: CogletServer, input_data: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{server.base_url}/predictions", json={"input": input_data}, timeout=30
    )
    return resp.json()


def post_many(
    server: CogletServer, inputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        return list(pool.map(lambda i: post(server, i), inputs))


class WebhookCollector:
    """Minimal HTTP server collecting webhook payloads per prediction ID."""

    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.lock = threading.Lock()
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                with collector.lock:
                    collector.events.setdefault(payload["id"], []).append(payload)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "WebhookCollector":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/webhook"

    def wait_terminal(
        self, prediction_id: str, timeout: float = 15.0
    ) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                for event in self.events.get(prediction_id, []):
                    if event["status"] in ("succeeded", "failed", "canceled"):
                        return event
            time.sleep(0.05)
        pytest.fail(f"no terminal webhook for {prediction_id}: {self.events}")


# =============================================================================
# Batch formation
# =============================================================================


class TestBatchFormation:
    def test_single_request_runs_as_batch_of_one_after_timeout(
        self, tmp_path: Path
    ) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=8, max_wait_ms=200)"
        )
        with CogletServer(predictor, env=batch_env(16)) as server:
            result = post(server, {"prompt": "solo"})
        assert result["status"] == "succeeded", result
        assert result["output"] == "solo|n=1"
        metrics = result["metrics"]
        assert metrics["batch_size"] == 1
        # The lone request waited for max_wait before dispatch.
        assert metrics["batch_queue_time"] >= 0.15
        assert metrics["batch_run_time"] >= 0
        assert metrics["predict_time"] >= metrics["batch_queue_time"]
        assert recorded_batches(tmp_path) == [["solo"]]

    def test_full_batch_dispatches_before_timeout(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=4, max_wait_ms=10000)"
        )
        with CogletServer(predictor, env=batch_env(8)) as server:
            start = time.monotonic()
            results = post_many(server, [{"prompt": f"p{i}"} for i in range(4)])
            elapsed = time.monotonic() - start
        assert elapsed < 5, "a full batch must not wait for max_wait"
        for i, result in enumerate(results):
            assert result["status"] == "succeeded", result
            assert result["output"] == f"p{i}|n=4"
            assert result["metrics"]["batch_size"] == 4
        assert len(recorded_batches(tmp_path)) == 1

    def test_partial_batch_dispatches_after_timeout(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=8, max_wait_ms=500)"
        )
        with CogletServer(predictor, env=batch_env(16)) as server:
            results = post_many(server, [{"prompt": "a"}, {"prompt": "b"}])
        assert all(r.get("status") == "succeeded" for r in results), results
        assert sorted(r["output"] for r in results) == ["a|n=2", "b|n=2"]
        assert [sorted(b) for b in recorded_batches(tmp_path)] == [["a", "b"]]

    def test_overflow_forms_multiple_batches(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=3, max_wait_ms=200)"
        )
        with CogletServer(predictor, env=batch_env(8)) as server:
            results = post_many(
                server, [{"prompt": f"p{i}", "delay": 0.2} for i in range(7)]
            )
        batches = recorded_batches(tmp_path)
        assert all(len(b) <= 3 for b in batches)
        assert len(batches) >= 3
        assert sorted(p for b in batches for p in b) == sorted(
            f"p{i}" for i in range(7)
        )
        for i, result in enumerate(results):
            assert result["status"] == "succeeded", result
            # Each output is routed back to the request that sent the input.
            assert result["output"].startswith(f"p{i}|n=")

    def test_many_concurrent_requests_route_outputs_correctly(
        self, tmp_path: Path
    ) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=8, max_wait_ms=20)"
        )
        with CogletServer(predictor, env=batch_env(16)) as server:
            results = post_many(server, [{"prompt": f"req-{i}"} for i in range(16)])
        assert len(results) == 16
        ids = {r["id"] for r in results}
        assert len(ids) == 16, "every request keeps its own prediction ID"
        for i, result in enumerate(results):
            assert result["status"] == "succeeded", result
            assert result["output"].split("|")[0] == f"req-{i}"

    def test_async_batched_run(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, ASYNC_ECHO, "@cog.batch(max_batch_size=4, max_wait_ms=300)"
        )
        with CogletServer(predictor, env=batch_env(8)) as server:
            results = post_many(server, [{"prompt": f"a{i}"} for i in range(4)])
        for i, result in enumerate(results):
            assert result["status"] == "succeeded", result
            assert result["output"] == f"a{i}|n=4"

    def test_batch_logs_reach_every_member(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=2, max_wait_ms=2000)"
        )
        with CogletServer(predictor, env=batch_env(4)) as server:
            results = post_many(server, [{"prompt": "x"}, {"prompt": "y"}])
        for result in results:
            assert "running batch of 2" in result["logs"]


# =============================================================================
# Request lifecycle semantics
# =============================================================================


class TestRequestSemantics:
    def test_async_requests_with_webhooks(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=3, max_wait_ms=2000)"
        )
        with (
            WebhookCollector() as hooks,
            CogletServer(predictor, env=batch_env(6)) as server,
        ):
            for i in range(3):
                resp = requests.put(
                    f"{server.base_url}/predictions/async-{i}",
                    json={"input": {"prompt": f"w{i}"}, "webhook": hooks.url},
                    headers={"Prefer": "respond-async"},
                    timeout=10,
                )
                assert resp.status_code == 202
                assert resp.json()["id"] == f"async-{i}"
            for i in range(3):
                event = hooks.wait_terminal(f"async-{i}")
                assert event["status"] == "succeeded"
                assert event["output"] == f"w{i}|n=3"
                assert event["metrics"]["batch_size"] == 3

    def test_idempotent_put_does_not_run_twice(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=4, max_wait_ms=1000)"
        )
        with (
            WebhookCollector() as hooks,
            CogletServer(predictor, env=batch_env(8)) as server,
        ):
            url = f"{server.base_url}/predictions/same-id"
            body = {"input": {"prompt": "once"}, "webhook": hooks.url}
            first = requests.put(
                url, json=body, headers={"Prefer": "respond-async"}, timeout=10
            )
            assert first.status_code == 202
            # While the first is queued, a repeat PUT returns its state.
            second = requests.put(
                url, json=body, headers={"Prefer": "respond-async"}, timeout=10
            )
            assert second.status_code == 202
            assert second.json()["id"] == "same-id"
            assert second.json()["status"] == "processing"
            event = hooks.wait_terminal("same-id")
            assert event["output"] == "once|n=1"
        assert recorded_batches(tmp_path) == [["once"]]

    def test_sync_put_returns_output(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=4, max_wait_ms=50)"
        )
        with CogletServer(predictor, env=batch_env(8)) as server:
            resp = requests.put(
                f"{server.base_url}/predictions/sync-put",
                json={"input": {"prompt": "hi"}},
                timeout=10,
            )
        assert resp.status_code == 200
        assert resp.json()["output"] == "hi|n=1"
        assert resp.json()["id"] == "sync-put"

    def test_cancel_before_dispatch(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=4, max_wait_ms=3000)"
        )
        with (
            WebhookCollector() as hooks,
            CogletServer(predictor, env=batch_env(8)) as server,
        ):
            for pid, prompt in [("keep", "k"), ("drop", "d")]:
                resp = requests.put(
                    f"{server.base_url}/predictions/{pid}",
                    json={"input": {"prompt": prompt}, "webhook": hooks.url},
                    headers={"Prefer": "respond-async"},
                    timeout=10,
                )
                assert resp.status_code == 202
            time.sleep(0.3)  # both queued, batch still waiting for max_wait
            cancel = requests.post(f"{server.base_url}/predictions/drop/cancel")
            assert cancel.status_code == 200
            assert hooks.wait_terminal("drop")["status"] == "canceled"
            kept = hooks.wait_terminal("keep")
            assert kept["status"] == "succeeded"
            assert kept["output"] == "k|n=1"
        assert recorded_batches(tmp_path) == [["k"]]

    def test_cancel_member_of_running_batch(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=2, max_wait_ms=2000)"
        )
        with (
            WebhookCollector() as hooks,
            CogletServer(predictor, env=batch_env(4)) as server,
        ):
            for pid in ("slow-a", "slow-b"):
                requests.put(
                    f"{server.base_url}/predictions/{pid}",
                    json={"input": {"prompt": pid, "delay": 2.0}, "webhook": hooks.url},
                    headers={"Prefer": "respond-async"},
                    timeout=10,
                )
            deadline = time.time() + 5
            while not recorded_batches(tmp_path) and time.time() < deadline:
                time.sleep(0.05)
            assert recorded_batches(tmp_path), "batch should be running"
            start = time.monotonic()
            requests.post(f"{server.base_url}/predictions/slow-a/cancel")
            assert hooks.wait_terminal("slow-a")["status"] == "canceled"
            # The cancelled member resolves without waiting for the batch.
            assert time.monotonic() - start < 1.5
            other = hooks.wait_terminal("slow-b")
            assert other["status"] == "succeeded"
            assert other["output"] == "slow-b|n=2"


# =============================================================================
# Errors
# =============================================================================


class TestBatchErrors:
    def test_exception_fails_every_member(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=2, max_wait_ms=2000)"
        )
        with CogletServer(predictor, env=batch_env(4)) as server:
            results = post_many(
                server, [{"prompt": "ok"}, {"prompt": "bad", "fail": True}]
            )
            # The scheduler keeps serving after a failed batch.
            after = post(server, {"prompt": "again"})
        for result in results:
            assert result["status"] == "failed", result
            assert "batch exploded" in result["error"]
        assert after["status"] == "succeeded"

    def test_wrong_output_count_fails_every_member(self, tmp_path: Path) -> None:
        body = """    def run(self, inputs: list[Item]) -> list[str]:
        return ["only-one"]
"""
        predictor = write_runner(
            tmp_path, body, "@cog.batch(max_batch_size=2, max_wait_ms=2000)"
        )
        with CogletServer(predictor, env=batch_env(4)) as server:
            results = post_many(server, [{"prompt": "a"}, {"prompt": "b"}])
        for result in results:
            assert result["status"] == "failed", result
            assert "returned 1 outputs for a batch of 2 inputs" in result["error"]

    def test_invalid_item_fails_only_that_request(self, tmp_path: Path) -> None:
        predictor = write_runner(
            tmp_path, SYNC_ECHO, "@cog.batch(max_batch_size=2, max_wait_ms=300)"
        )
        with CogletServer(predictor, env=batch_env(4)) as server:
            # No bundled schema in tests, so the item is validated in the worker.
            results = post_many(server, [{"prompt": "good"}, {"nope": 1}])
        assert results[0]["status"] == "succeeded"
        assert results[0]["output"] == "good|n=1"
        assert results[1]["status"] == "failed"
        assert "unexpected input field" in results[1]["error"]

    def test_invalid_batch_signature_fails_setup(self, tmp_path: Path) -> None:
        body = """    def run(self, prompt: str) -> str:
        return prompt
"""
        predictor = write_runner(tmp_path, body, "@cog.batch(max_batch_size=2)")
        with CogletServer(predictor, env=batch_env(4), wait_ready=False) as server:
            _wait_for_health_status(server, "SETUP_FAILED", timeout=15)
            logs = server.health_check()["setup"]["logs"] + "".join(server.stderr_lines)
        assert "list[Item]" in logs


# =============================================================================
# Backward compatibility
# =============================================================================


def test_non_batched_runner_is_unaffected(tmp_path: Path) -> None:
    predictor = tmp_path / "predict.py"
    predictor.write_text("""
from cog import BaseRunner

class Predictor(BaseRunner):
    def run(self, prompt: str) -> str:
        return prompt.upper()
""")
    (tmp_path / "cog.yaml").write_text('predict: "predict.py:Predictor"\n')
    with CogletServer(predictor) as server:
        result = post(server, {"prompt": "hi"})
    assert result["output"] == "HI"
    assert "batch_size" not in result["metrics"]
