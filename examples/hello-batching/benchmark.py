"""Measure throughput and latency of a Cog server under concurrent load.

Point it at a running server (``cog serve`` / ``docker run``)::

    python benchmark.py --url http://localhost:5000 --requests 400 --clients 16

or let it start local coglet servers for the batched and unbatched runners in
``run.py`` and compare them (requires the ``coglet`` and ``cog`` packages and
numpy in the current environment)::

    python benchmark.py --compare --requests 400 --clients 16

Every number printed is measured from the responses of this run. Requests
rejected with 409 (all slots busy) are retried after a short pause; the
latency of a request includes its retries.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


@dataclass
class Result:
    latency: float
    batch_size: int
    retries: int
    ok: bool


@dataclass
class Report:
    label: str
    requests: int
    failures: int
    retries: int
    elapsed: float
    latencies: list[float]
    batch_sizes: list[int]

    def print(self) -> None:
        ordered = sorted(self.latencies)

        def pct(p: float) -> float:
            return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

        print(f"== {self.label}")
        print(f"   requests:         {self.requests} ({self.failures} failed)")
        print(f"   throughput:       {self.requests / self.elapsed:.1f} req/s")
        print(f"   latency p50:      {pct(0.50) * 1000:.1f} ms")
        print(f"   latency p95:      {pct(0.95) * 1000:.1f} ms")
        print(f"   avg batch size:   {statistics.mean(self.batch_sizes):.2f}")
        print(f"   409 retries:      {self.retries}")


def send(session: requests.Session, url: str, seed: int) -> Result:
    start = time.perf_counter()
    retries = 0
    while True:
        resp = session.post(
            f"{url}/predictions", json={"input": {"seed": seed}}, timeout=120
        )
        if resp.status_code == 409:
            retries += 1
            time.sleep(0.002)
            continue
        latency = time.perf_counter() - start
        body = resp.json()
        ok = resp.status_code == 200 and body.get("status") == "succeeded"
        batch_size = int(body.get("metrics", {}).get("batch_size", 1))
        return Result(latency=latency, batch_size=batch_size, retries=retries, ok=ok)


def run_load(url: str, label: str, total: int, clients: int) -> Report:
    local = threading.local()

    def task(seed: int) -> Result:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        return send(local.session, url, seed)

    # Warm up connections and the model before timing.
    with ThreadPoolExecutor(max_workers=clients) as pool:
        list(pool.map(task, range(clients)))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        results = list(pool.map(task, range(total)))
    elapsed = time.perf_counter() - start

    return Report(
        label=label,
        requests=total,
        failures=sum(not r.ok for r in results),
        retries=sum(r.retries for r in results),
        elapsed=elapsed,
        latencies=[r.latency for r in results],
        batch_sizes=[r.batch_size for r in results],
    )


class LocalServer:
    """Start a coglet server for a runner in run.py."""

    def __init__(self, runner: str, slots: int) -> None:
        self.runner = runner
        self.slots = slots
        self.process: subprocess.Popen[str] | None = None
        self.url = ""

    def __enter__(self) -> "LocalServer":
        ref = f"{HERE / 'run.py'}:{self.runner}"
        # Runs this interpreter with a fixed command; no untrusted input.
        self.process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                f"import coglet; coglet.server.serve({ref!r}, port=0)",
            ],
            cwd=HERE,
            env={**os.environ, "COG_MAX_CONCURRENCY": str(self.slots)},
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
        )
        assert self.process.stderr is not None
        deadline = time.time() + 30
        for line in self.process.stderr:
            match = re.search(r"Starting coglet server on [\d.]+:(\d+)", line)
            if match:
                self.url = f"http://localhost:{match.group(1)}"
                break
            if time.time() > deadline:
                break
        if not self.url:
            raise RuntimeError(f"server for {self.runner} did not start")
        threading.Thread(target=self._drain, daemon=True).start()
        while time.time() < deadline:
            try:
                if (
                    requests.get(f"{self.url}/health-check", timeout=1).json()["status"]
                    == "READY"
                ):
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"server for {self.runner} did not become ready")

    def _drain(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for _ in self.process.stderr:
            pass

    def __exit__(self, *args: object) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="URL of a running Cog server")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="start and compare local batched/unbatched servers",
    )
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument(
        "--clients", type=int, default=16, help="concurrent client connections"
    )
    args = parser.parse_args()

    if args.url:
        run_load(args.url.rstrip("/"), args.url, args.requests, args.clients).print()
        return
    if not args.compare:
        parser.error("pass --url or --compare")

    # The unbatched runner is sync, so it serves one request at a time.
    with LocalServer("UnbatchedRunner", slots=1) as server:
        run_load(
            server.url, "unbatched run() (1 slot)", args.requests, args.clients
        ).print()
    # Batched runners get 2 x max_batch_size slots, as `cog build` configures.
    with LocalServer("Runner", slots=16) as server:
        run_load(
            server.url,
            "@batch(max_batch_size=8, max_wait_ms=5) (16 slots)",
            args.requests,
            args.clients,
        ).print()


if __name__ == "__main__":
    main()
