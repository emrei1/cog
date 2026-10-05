"""
Dynamic request batching support.

``@cog.batch`` marks a ``run()`` method as batch-capable. The Cog runtime then
collects independent prediction requests into batches and calls ``run()`` once
per batch with a list of input objects. Each request keeps its own prediction
ID, status, output, webhook, and cancellation state -- batching is purely an
execution optimization, not a bulk API.

The helpers in this module are used by the coglet worker at startup (to
validate the batch signature) and per request (to build input objects).
"""

import dataclasses
import inspect
import math
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .input import FieldInfo
from .model import BaseModel

_F = TypeVar("_F", bound=Callable[..., object])

#: Upper bound on ``max_batch_size``. Each queued request occupies a slot, so
#: very large values would create an unreasonable number of slot sockets.
MAX_BATCH_SIZE_LIMIT = 1024

#: Upper bound on ``max_wait_ms`` (one minute).
MAX_WAIT_MS_LIMIT = 60_000


@dataclass(frozen=True)
class BatchConfig:
    """Batching parameters attached to a ``run()`` method by ``@cog.batch``."""

    max_batch_size: int
    max_wait_ms: float


@dataclass(frozen=True)
class BatchSignature:
    """Validated shape of a batched ``run()`` method."""

    config: BatchConfig
    #: The ``cog.BaseModel`` subclass each request is converted into.
    item_type: type[BaseModel]
    #: The per-item output type (the ``X`` in ``-> list[X]``).
    output_type: Any


def batch(
    fn: _F | None = None,
    *,
    max_batch_size: int | None = None,
    max_wait_ms: float = 5,
) -> Callable[[_F], _F]:
    """Opt a ``run()`` method into dynamic request batching.

    The decorated method receives a list of input objects and must return a
    list of outputs of the same length, in the same order::

        class Item(BaseModel):
            prompt: str
            steps: int = Input(default=20, ge=1)

        class Runner(BaseRunner):
            @cog.batch(max_batch_size=8, max_wait_ms=5)
            def run(self, inputs: list[Item]) -> list[str]:
                return self.model.generate([i.prompt for i in inputs])

    Args:
        max_batch_size: Maximum number of requests executed together.
        max_wait_ms: Maximum time the oldest queued request waits for the
            batch to fill before a partial batch is dispatched.
    """
    if fn is not None:
        raise TypeError(
            "batch must be used as @batch(max_batch_size=..., max_wait_ms=...)"
        )
    if max_batch_size is None:
        raise TypeError("batch requires max_batch_size=...")
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
        raise TypeError("batch max_batch_size must be an integer")
    if max_batch_size < 1:
        raise ValueError("batch max_batch_size must be at least 1")
    if max_batch_size > MAX_BATCH_SIZE_LIMIT:
        raise ValueError(f"batch max_batch_size must be at most {MAX_BATCH_SIZE_LIMIT}")
    if isinstance(max_wait_ms, bool) or not isinstance(max_wait_ms, (int, float)):
        raise TypeError("batch max_wait_ms must be a number")
    if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
        raise ValueError("batch max_wait_ms must be a finite number >= 0")
    if max_wait_ms > MAX_WAIT_MS_LIMIT:
        raise ValueError(f"batch max_wait_ms must be at most {MAX_WAIT_MS_LIMIT}")

    config = BatchConfig(max_batch_size=max_batch_size, max_wait_ms=float(max_wait_ms))

    def decorate(inner: _F) -> _F:
        if inspect.isgeneratorfunction(inner) or inspect.isasyncgenfunction(inner):
            raise TypeError(
                "@batch does not support generator run() methods; "
                "return a list of outputs instead"
            )
        if not callable(inner):
            raise TypeError("@batch must decorate a function")
        inner.__cog_batch__ = config  # type: ignore[attr-defined]
        return inner

    return decorate


def get_batch_config(func: object) -> BatchConfig | None:
    """Return the ``@batch`` configuration of a (bound) function, if any."""
    target = getattr(func, "__func__", func)
    config = getattr(target, "__cog_batch__", None)
    return config if isinstance(config, BatchConfig) else None


def _list_item_type(annotation: Any) -> Any:
    """Return ``X`` for ``list[X]`` / ``List[X]`` annotations, else None."""
    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        if len(args) == 1:
            return args[0]
    return None


def inspect_batch_run(func: Callable[..., Any]) -> BatchSignature | None:
    """Validate a batched ``run()`` method and describe its signature.

    Returns None when ``func`` is not decorated with ``@batch``. Raises
    ``TypeError`` with a descriptive message when the signature cannot be
    batched; the worker calls this during startup so invalid batch
    definitions fail before serving traffic.
    """
    config = get_batch_config(func)
    if config is None:
        return None

    target = getattr(func, "__func__", func)
    if getattr(target, "__cog_concurrent_max__", 1) > 1:
        raise TypeError(
            "@batch cannot be combined with @concurrent(max > 1); "
            "batches already execute one at a time"
        )
    if inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target):
        raise TypeError("@batch does not support generator run() methods")

    sig = inspect.signature(func)
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    if len(params) != 1 or params[0].kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    ):
        raise TypeError(
            "@batch run() must take exactly one parameter, e.g. "
            "run(self, inputs: list[Item]) -> list[Output]"
        )
    param = params[0]

    try:
        hints = typing.get_type_hints(target)
    except Exception as e:  # noqa: BLE001 - surface any annotation resolution error
        raise TypeError(f"@batch run() annotations could not be resolved: {e}") from e

    item_type = _list_item_type(hints.get(param.name))
    if not (inspect.isclass(item_type) and issubclass(item_type, BaseModel)):
        raise TypeError(
            f"@batch run() parameter '{param.name}' must be annotated as "
            "list[Item] where Item is a cog.BaseModel subclass"
        )
    if not dataclasses.is_dataclass(item_type):
        raise TypeError(
            f"@batch input type {item_type.__name__} must be a dataclass "
            "(do not use auto_dataclass=False)"
        )

    output_type = _list_item_type(hints.get("return"))
    if output_type is None:
        raise TypeError("@batch run() must be annotated to return list[Output]")

    return BatchSignature(config=config, item_type=item_type, output_type=output_type)


def build_batch_item(item_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    """Construct one batch input object from a prepared request payload.

    Fields omitted from the payload fall back to their declared defaults;
    ``Input(default=...)`` defaults are unwrapped so the object never holds
    ``FieldInfo`` placeholders.
    """
    kwargs = dict(payload)
    known: set[str] = set()
    for field in dataclasses.fields(item_type):  # type: ignore[arg-type]
        if not field.init:
            continue
        known.add(field.name)
        if field.name in kwargs:
            continue
        if isinstance(field.default, FieldInfo):
            kwargs[field.name] = field.default.default

    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise ValueError(f"unexpected input field(s): {', '.join(unknown)}")

    try:
        return item_type(**kwargs)
    except TypeError as e:
        raise ValueError(f"invalid input for {item_type.__name__}: {e}") from e


def check_batch_outputs(outputs: Any, expected: int) -> list[Any]:
    """Validate a batched ``run()`` return value.

    Returns the outputs as a list. Raises ``ValueError`` when the value is not
    a list/tuple or when its length differs from the number of inputs.
    """
    if not isinstance(outputs, (list, tuple)):
        raise ValueError(
            f"@batch run() must return a list of outputs, got {type(outputs).__name__}"
        )
    if len(outputs) != expected:
        raise ValueError(
            f"@batch run() returned {len(outputs)} outputs for a batch of "
            f"{expected} inputs"
        )
    return list(outputs)
