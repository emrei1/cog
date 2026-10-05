# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
#
# A two-layer MLP served with dynamic request batching. Each request carries one
# input vector. Without batching every request is a matrix-vector product that
# streams all weights from memory; with @batch, concurrent requests share one
# matrix-matrix product, which reuses the weights across the whole batch. GPU
# models benefit from batching for the same reason.

import os

# Pin BLAS to one thread so the comparison measures batching rather than BLAS
# thread scheduling (multi-threaded OpenBLAS on small matrices can be slower
# than a single thread). Set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from cog import BaseModel, BaseRunner, Input, batch

HIDDEN = 2048


class Item(BaseModel):
    seed: int = Input(description="Seed used to generate this request's input vector")


class Output(BaseModel):
    seed: int
    score: float


def make_weights() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    scale = np.float32(1.0 / np.sqrt(HIDDEN))
    w1 = rng.standard_normal((HIDDEN, HIDDEN), dtype=np.float32) * scale
    w2 = rng.standard_normal((HIDDEN, HIDDEN), dtype=np.float32) * scale
    return w1, w2


def make_input(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(HIDDEN, dtype=np.float32)


def forward(w1: np.ndarray, w2: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x: (batch, HIDDEN) -> one score per row."""
    hidden = np.maximum(x @ w1, 0.0)
    return (hidden @ w2).mean(axis=1)


class Runner(BaseRunner):
    """Batched: up to 8 concurrent requests share one forward pass."""

    def setup(self) -> None:
        self.w1, self.w2 = make_weights()

    @batch(max_batch_size=8, max_wait_ms=5)
    def run(self, inputs: list[Item]) -> list[Output]:
        x = np.stack([make_input(item.seed) for item in inputs])
        scores = forward(self.w1, self.w2, x)
        return [
            Output(seed=item.seed, score=float(score))
            for item, score in zip(inputs, scores, strict=True)
        ]


class UnbatchedRunner(BaseRunner):
    """Baseline: the same model, one request per forward pass."""

    def setup(self) -> None:
        self.w1, self.w2 = make_weights()

    def run(self, seed: int = Input(description="Seed")) -> Output:
        score = forward(self.w1, self.w2, make_input(seed)[None, :])[0]
        return Output(seed=seed, score=float(score))
