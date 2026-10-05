package python

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/replicate/cog/pkg/schema"
)

func TestBatchDecoratorDerivesPerRequestSchema(t *testing.T) {
	source := `
import cog
from cog import BaseModel, BaseRunner, Input, Path


class Item(BaseModel):
    prompt: str = Input(description="Text prompt")
    steps: int = Input(default=20, ge=1, le=100)
    image: Path | None = None
    scale: float = 1.5


class Output(BaseModel):
    text: str
    score: float


class Runner(BaseRunner):
    @cog.batch(max_batch_size=8, max_wait_ms=5)
    def run(self, inputs: list[Item]) -> list[Output]:
        return []
`
	info := parse(t, source, "Runner")
	require.NotNil(t, info.Batch)
	require.Equal(t, 8, info.Batch.MaxBatchSize)
	require.InDelta(t, 5.0, info.Batch.MaxWaitMs, 1e-9)
	require.False(t, info.IsAsync)

	require.Equal(t, []string{"prompt", "steps", "image", "scale"}, info.Inputs.Keys())

	prompt, ok := info.Inputs.Get("prompt")
	require.True(t, ok)
	require.Equal(t, schema.TypeString, prompt.FieldType.Primitive)
	require.Nil(t, prompt.Default)
	require.Equal(t, "Text prompt", *prompt.Description)

	steps, _ := info.Inputs.Get("steps")
	require.Equal(t, schema.TypeInteger, steps.FieldType.Primitive)
	require.Equal(t, int64(20), steps.Default.Int)
	require.InDelta(t, 1.0, *steps.GE, 1e-9)
	require.InDelta(t, 100.0, *steps.LE, 1e-9)

	image, _ := info.Inputs.Get("image")
	require.Equal(t, schema.TypePath, image.FieldType.Primitive)

	scale, _ := info.Inputs.Get("scale")
	require.InDelta(t, 1.5, scale.Default.Float, 1e-9)

	// The output schema is the element type, not the list.
	require.Equal(t, schema.SchemaObject, info.Output.Kind)
}

func TestBatchDecoratorImportedName(t *testing.T) {
	source := `
from cog import BaseModel, BaseRunner, batch


class Item(BaseModel):
    prompt: str


class Runner(BaseRunner):
    @batch(max_batch_size=4, max_wait_ms=2.5)
    async def run(self, inputs: list[Item]) -> list[str]:
        return []
`
	info := parse(t, source, "Runner")
	require.NotNil(t, info.Batch)
	require.Equal(t, 4, info.Batch.MaxBatchSize)
	require.InDelta(t, 2.5, info.Batch.MaxWaitMs, 1e-9)
	require.True(t, info.IsAsync)
	require.Equal(t, schema.SchemaPrimitive, info.Output.Kind)
}

func TestBatchDecoratorDefaultsMaxWait(t *testing.T) {
	source := `
import cog
from cog import BaseModel, BaseRunner


class Item(BaseModel):
    prompt: str


class Runner(BaseRunner):
    @cog.batch(max_batch_size=2)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`
	info := parse(t, source, "Runner")
	require.InDelta(t, 5.0, info.Batch.MaxWaitMs, 1e-9)
}

func TestBatchDecoratorMetadataParse(t *testing.T) {
	source := `
import cog
from cog import BaseModel, BaseRunner


class Item(BaseModel):
    prompt: str


class Runner(BaseRunner):
    @cog.batch(max_batch_size=3)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`
	info, err := ParsePredictorMetadataWithSourcePath([]byte(source), "Runner", schema.ModePredict, "", "")
	require.NoError(t, err)
	require.NotNil(t, info.Batch)
	require.Equal(t, 3, info.Batch.MaxBatchSize)
}

func TestBatchDecoratorIgnoredWhenNotFromCog(t *testing.T) {
	source := `
from other import batch
from cog import BaseRunner


class Runner(BaseRunner):
    @batch(max_batch_size=4)
    def run(self, prompt: str) -> str:
        return prompt
`
	info := parse(t, source, "Runner")
	require.Nil(t, info.Batch)
	_, ok := info.Inputs.Get("prompt")
	require.True(t, ok)
}

func TestNonBatchedRunnerHasNoBatchInfo(t *testing.T) {
	source := `
from cog import BaseRunner


class Runner(BaseRunner):
    def run(self, prompt: str) -> str:
        return prompt
`
	require.Nil(t, parse(t, source, "Runner").Batch)
}

func TestBatchDecoratorErrors(t *testing.T) {
	header := `
import cog
from typing import Iterator
from cog import BaseModel, BaseRunner


class Item(BaseModel):
    prompt: str

`
	tests := []struct {
		name    string
		runner  string
		message string
	}{
		{
			name: "bare decorator",
			runner: `class Runner(BaseRunner):
    @cog.batch
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "must be called with arguments",
		},
		{
			name: "missing max_batch_size",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_wait_ms=5)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "requires max_batch_size",
		},
		{
			name: "zero max_batch_size",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=0)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "max_batch_size must be between",
		},
		{
			name: "non-literal max_batch_size",
			runner: `SIZE = 4
class Runner(BaseRunner):
    @cog.batch(max_batch_size=SIZE)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "integer literal",
		},
		{
			name: "negative max_wait_ms",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2, max_wait_ms=-1)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "max_wait_ms",
		},
		{
			name: "unknown argument",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2, timeout=3)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "unexpected argument",
		},
		{
			name: "per-item parameters",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2)
    def run(self, prompt: str) -> list[str]:
        return []
`,
			message: "list[Item]",
		},
		{
			name: "two parameters",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2)
    def run(self, inputs: list[Item], extra: int = 1) -> list[str]:
        return []
`,
			message: "exactly one parameter",
		},
		{
			name: "non-model item",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2)
    def run(self, inputs: list[str]) -> list[str]:
        return []
`,
			message: "cog.BaseModel subclass",
		},
		{
			name: "non-list output",
			runner: `class Runner(BaseRunner):
    @cog.batch(max_batch_size=2)
    def run(self, inputs: list[Item]) -> str:
        return ""
`,
			message: "return list[Output]",
		},
		{
			name: "streaming",
			runner: `class Runner(BaseRunner):
    @cog.streaming
    @cog.batch(max_batch_size=2)
    def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "@streaming",
		},
		{
			name: "concurrent",
			runner: `class Runner(BaseRunner):
    @cog.concurrent(max=4)
    @cog.batch(max_batch_size=2)
    async def run(self, inputs: list[Item]) -> list[str]:
        return []
`,
			message: "@concurrent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			se := parseErr(t, header+tt.runner, "Runner", schema.ModePredict)
			require.Contains(t, se.Error(), tt.message)
		})
	}
}

func TestBatchDecoratorRejectedOnTrain(t *testing.T) {
	source := `
import cog
from cog import BaseModel


class Item(BaseModel):
    prompt: str


@cog.batch(max_batch_size=2)
def train(inputs: list[Item]) -> list[str]:
    return []
`
	se := parseErr(t, source, "train", schema.ModeTrain)
	require.Contains(t, se.Error(), "not train()")
}
