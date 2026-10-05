"""Tests for the cog.batch decorator and batch signature helpers."""

import sys
import types
from collections.abc import AsyncIterator, Iterator

import pytest

if "coglet" not in sys.modules:
    coglet = types.ModuleType("coglet")
    coglet.CancelationException = type("CancelationException", (BaseException,), {})
    sys.modules["coglet"] = coglet

from cog import BaseModel, BaseRunner, Input, Path, batch, concurrent
from cog._batch import (
    BatchConfig,
    build_batch_item,
    check_batch_outputs,
    get_batch_config,
    inspect_batch_run,
)


class Item(BaseModel):
    prompt: str
    steps: int = Input(default=20, ge=1)
    image: Path | None = None


class Out(BaseModel):
    text: str


# =============================================================================
# Decorator argument validation
# =============================================================================


def test_batch_sets_config() -> None:
    @batch(max_batch_size=8, max_wait_ms=5)
    def run(self: object, inputs: list[Item]) -> list[str]:
        return []

    assert get_batch_config(run) == BatchConfig(max_batch_size=8, max_wait_ms=5.0)


def test_batch_default_max_wait() -> None:
    @batch(max_batch_size=4)
    def run(self: object, inputs: list[Item]) -> list[str]:
        return []

    assert get_batch_config(run) == BatchConfig(max_batch_size=4, max_wait_ms=5.0)


def test_batch_accepts_float_wait() -> None:
    @batch(max_batch_size=2, max_wait_ms=0.5)
    def run(self: object, inputs: list[Item]) -> list[str]:
        return []

    config = get_batch_config(run)
    assert config is not None
    assert config.max_wait_ms == 0.5


def test_batch_returns_same_function() -> None:
    def run(self: object, inputs: list[Item]) -> list[str]:
        return []

    assert batch(max_batch_size=2)(run) is run


def test_bare_batch_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be used as @batch"):

        @batch  # type: ignore[arg-type]
        def run(self: object, inputs: list[Item]) -> list[str]:
            return []


def test_batch_requires_max_batch_size() -> None:
    with pytest.raises(TypeError, match="requires max_batch_size"):
        batch()


@pytest.mark.parametrize("value", [0, -1])
def test_batch_rejects_small_max_batch_size(value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        batch(max_batch_size=value)


def test_batch_rejects_huge_max_batch_size() -> None:
    with pytest.raises(ValueError, match="at most"):
        batch(max_batch_size=100_000)


@pytest.mark.parametrize("value", [True, 2.0, "8"])
def test_batch_rejects_non_integer_max_batch_size(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        batch(max_batch_size=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_batch_rejects_invalid_max_wait(value: float) -> None:
    with pytest.raises(ValueError, match="max_wait_ms"):
        batch(max_batch_size=2, max_wait_ms=value)


def test_batch_rejects_non_number_max_wait() -> None:
    with pytest.raises(TypeError, match="max_wait_ms"):
        batch(max_batch_size=2, max_wait_ms="5")  # type: ignore[arg-type]


def test_batch_rejects_generator() -> None:
    with pytest.raises(TypeError, match="generator"):

        @batch(max_batch_size=2)
        def run(self: object, inputs: list[Item]) -> Iterator[str]:
            yield "x"


def test_batch_rejects_async_generator() -> None:
    with pytest.raises(TypeError, match="generator"):

        @batch(max_batch_size=2)
        async def run(self: object, inputs: list[Item]) -> AsyncIterator[str]:
            yield "x"


def test_undecorated_function_has_no_config() -> None:
    def run(self: object, prompt: str) -> str:
        return prompt

    assert get_batch_config(run) is None
    assert inspect_batch_run(run) is None


# =============================================================================
# Signature inspection (startup validation)
# =============================================================================


def test_inspect_batch_run_on_bound_method() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=8, max_wait_ms=5)
        def run(self, inputs: list[Item]) -> list[Out]:
            return [Out(text=i.prompt) for i in inputs]

    sig = inspect_batch_run(Runner().run)
    assert sig is not None
    assert sig.item_type is Item
    assert sig.output_type is Out
    assert sig.config.max_batch_size == 8


def test_inspect_batch_run_async() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=4)
        async def run(self, inputs: list[Item]) -> list[str]:
            return [i.prompt for i in inputs]

    sig = inspect_batch_run(Runner().run)
    assert sig is not None
    assert sig.output_type is str


def test_inspect_batch_run_rejects_multiple_params() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=4)
        def run(self, inputs: list[Item], extra: int = 1) -> list[str]:
            return []

    with pytest.raises(TypeError, match="exactly one parameter"):
        inspect_batch_run(Runner().run)


def test_inspect_batch_run_rejects_non_list_input() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=4)
        def run(self, inputs: Item) -> list[str]:
            return []

    with pytest.raises(TypeError, match="list\\[Item\\]"):
        inspect_batch_run(Runner().run)


def test_inspect_batch_run_rejects_non_model_items() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=4)
        def run(self, inputs: list[str]) -> list[str]:
            return []

    with pytest.raises(TypeError, match="cog.BaseModel"):
        inspect_batch_run(Runner().run)


def test_inspect_batch_run_rejects_non_list_output() -> None:
    class Runner(BaseRunner):
        @batch(max_batch_size=4)
        def run(self, inputs: list[Item]) -> str:
            return ""

    with pytest.raises(TypeError, match="return list\\[Output\\]"):
        inspect_batch_run(Runner().run)


def test_inspect_batch_run_rejects_concurrent() -> None:
    class Runner(BaseRunner):
        @concurrent(max=4)
        @batch(max_batch_size=4)
        async def run(self, inputs: list[Item]) -> list[str]:
            return []

    with pytest.raises(TypeError, match="@concurrent"):
        inspect_batch_run(Runner().run)


# =============================================================================
# Item construction and output checks
# =============================================================================


def test_build_batch_item_uses_payload() -> None:
    item = build_batch_item(Item, {"prompt": "hi", "steps": 3})
    assert isinstance(item, Item)
    assert item.prompt == "hi"
    assert item.steps == 3
    assert item.image is None


def test_build_batch_item_unwraps_input_defaults() -> None:
    item = build_batch_item(Item, {"prompt": "hi"})
    assert item.steps == 20


def test_build_batch_item_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unexpected input field"):
        build_batch_item(Item, {"prompt": "hi", "bogus": 1})


def test_build_batch_item_rejects_missing_required() -> None:
    with pytest.raises(ValueError, match="invalid input for Item"):
        build_batch_item(Item, {"steps": 1})


def test_check_batch_outputs_accepts_matching_list() -> None:
    assert check_batch_outputs(("a", "b"), 2) == ["a", "b"]


def test_check_batch_outputs_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="returned 1 outputs for a batch of 2"):
        check_batch_outputs(["a"], 2)


def test_check_batch_outputs_rejects_non_list() -> None:
    with pytest.raises(ValueError, match="must return a list"):
        check_batch_outputs("ab", 2)
