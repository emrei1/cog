# hello-batching

This example shows dynamic request batching with `@cog.batch`.

The model is a two-layer MLP written with numpy. Every prediction sends one input (a `seed` used to generate an input vector). Without batching, each prediction runs a matrix-vector product that reads all of the model's weights from memory. With `@cog.batch`, predictions that arrive together share one matrix-matrix product, so the weights are read once per batch instead of once per request. GPU models benefit from batching for the same reason.

```py
from cog import BaseModel, BaseRunner, Input, batch


class Item(BaseModel):
    seed: int = Input(description="Seed used to generate this request's input vector")


class Runner(BaseRunner):
    @batch(max_batch_size=8, max_wait_ms=5)
    def run(self, inputs: list[Item]) -> list[Output]:
        ...
```

Clients send ordinary, independent predictions:

```console
$ cog serve
$ curl -s localhost:8393/predictions -H 'Content-Type: application/json' \
    -d '{"input": {"seed": 42}}'
```

Each response contains one output and the batch metrics `batch_size`, `batch_queue_time`, and `batch_run_time`.

`run.py` also defines `UnbatchedRunner`: the same model without batching, used as the baseline in the benchmark.

## Benchmark

`benchmark.py` sends concurrent requests and reports throughput, p50/p95 latency, the average batch size (from each response's `batch_size` metric), and how many requests were retried after a `409 Conflict`. It only prints values measured during the run.

Against a running server:

```console
$ python benchmark.py --url http://localhost:8393 --requests 400 --clients 16
```

To compare the batched and unbatched runners side by side, run it in an environment with `cog`, `coglet`, `numpy`, and `requests` installed. It starts a local coglet server for each runner:

```console
$ python benchmark.py --compare --requests 600 --clients 16
```

The unbatched runner is sync, so it serves one prediction at a time; the batched runner gets `2 × max_batch_size` slots, which is what `cog build` configures. Results depend heavily on the hardware, the model, and the load, so measure your own model.

`run.py` pins BLAS to one thread. On the machine we tested, multi-threaded OpenBLAS made small matrix products slower than a single thread, which hid the effect of batching.
