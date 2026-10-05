package dockerfile

import (
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/replicate/cog/pkg/config"
	"github.com/replicate/cog/pkg/docker/dockertest"
	"github.com/replicate/cog/pkg/registry/registrytest"
)

func newTestGenerator(t *testing.T, yaml string) *StandardGenerator {
	t.Helper()
	conf, err := config.FromYAML([]byte(yaml))
	require.NoError(t, err)
	require.NoError(t, conf.Complete(""))
	gen, err := NewStandardGenerator(conf, t.TempDir(), t.TempDir(), "", dockertest.NewMockCommand(), registrytest.NewMockRegistryClient(), true)
	require.NoError(t, err)
	pypiWheels(gen)
	return gen
}

func TestGenerateEmptyAMD(t *testing.T) {
	gen := newTestGenerator(t, `
build:
  accelerator: amd
  rocm: "6.2"
  python_version: "3.12"
predict: predict.py:Predictor
`)
	// AMD has no cog-base images, so the default (unset) is the generated path.
	require.False(t, gen.IsUsingCogBaseImage())

	_, actual, _, err := gen.GenerateModelBaseWithSeparateWeights(t.Context(), "r8.im/replicate/cog-test")
	require.NoError(t, err)

	expected := `#syntax=docker/dockerfile:1.4
FROM r8.im/replicate/cog-test-weights AS weights
FROM rocm/dev-ubuntu-22.04:6.2
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
ENV PATH=$PATH:/opt/rocm/bin
` + testTini() + testInstallPython("3.12") + testInstallCogGPU(gen.strip) + `
RUN find / -type f -name "*python*.so" -printf "%h\n" | sort -u > /etc/ld.so.conf.d/cog.conf && ldconfig
WORKDIR /src
EXPOSE 5000
ENV COG_PREDICT_TYPE_STUB="predict.py:Predictor"
CMD ["python", "-m", "cog.server.http"]
COPY . /src`

	require.Equal(t, expected, actual)
}

func TestGenerateFullAMD(t *testing.T) {
	gen := newTestGenerator(t, `
build:
  accelerator: amd
  python_version: "3.12"
  system_packages:
    - ffmpeg
  python_packages:
    - torch==2.5.1
    - torchvision==0.20.1
    - pandas==2.0.3
predict: predict.py:Predictor
`)
	_, actual, _, err := gen.GenerateModelBaseWithSeparateWeights(t.Context(), "r8.im/replicate/cog-test")
	require.NoError(t, err)

	assert.Contains(t, actual, "FROM rocm/dev-ubuntu-22.04:6.2\n")
	assert.NotContains(t, actual, "nvidia")
	assert.NotContains(t, actual, "NVIDIA")
	assert.Contains(t, actual, "apt-get install -qqy ffmpeg")
	assert.Contains(t, actual, "uv run pip install --break-system-packages --cache-dir /root/.cache/pip -r /tmp/requirements.txt")

	requirements, err := os.ReadFile(path.Join(gen.tmpDir, "requirements.txt"))
	require.NoError(t, err)
	assert.Equal(t, `--extra-index-url https://download.pytorch.org/whl/rocm6.2/
torch==2.5.1
torchvision==0.20.1
pandas==2.0.3`, string(requirements))
}

func TestGenerateAMDWithoutVendorBaseImage(t *testing.T) {
	gen := newTestGenerator(t, `
build:
  accelerator: amd
  python_version: "3.12"
  python_packages:
    - torch==2.5.1
predict: predict.py:Predictor
`)
	gen.SetUseCudaBaseImage("false")
	_, actual, _, err := gen.GenerateModelBaseWithSeparateWeights(t.Context(), "r8.im/replicate/cog-test")
	require.NoError(t, err)

	// The PyTorch ROCm wheels bundle the ROCm runtime, so the Python image works too.
	assert.Contains(t, actual, "FROM python:3.12-slim\n")
	assert.Contains(t, actual, "ENV PATH=$PATH:/opt/rocm/bin")
	assert.NotContains(t, actual, "--break-system-packages")

	requirements, err := os.ReadFile(path.Join(gen.tmpDir, "requirements.txt"))
	require.NoError(t, err)
	assert.Contains(t, string(requirements), "--extra-index-url https://download.pytorch.org/whl/rocm6.2/")
}

func TestGenerateAMDRejectsCogBaseImage(t *testing.T) {
	gen := newTestGenerator(t, `
build:
  accelerator: amd
  python_version: "3.12"
predict: predict.py:Predictor
`)
	gen.SetUseCogBaseImage(true)
	_, _, _, err := gen.GenerateModelBaseWithSeparateWeights(t.Context(), "r8.im/replicate/cog-test")
	require.ErrorContains(t, err, "cog base images are not available for AMD ROCm")
}

// The same project built for each accelerator differs only in the
// accelerator-specific parts of the Dockerfile.
func TestGenerateSameProjectForEachAccelerator(t *testing.T) {
	const project = `
  python_version: "3.12"
  python_packages:
    - torch==2.5.1
predict: predict.py:Predictor
`
	tests := []struct {
		build     string
		baseImage string
		index     string
	}{
		{build: "build:\n  gpu: false", baseImage: "python:3.12-slim", index: "https://download.pytorch.org/whl/cpu"},
		{build: "build:\n  gpu: true", baseImage: "nvidia/cuda:", index: "https://download.pytorch.org/whl/cu"},
		{build: "build:\n  accelerator: amd", baseImage: "rocm/dev-ubuntu-22.04:6.2", index: "https://download.pytorch.org/whl/rocm6.2/"},
	}
	for _, tt := range tests {
		t.Run(tt.build, func(t *testing.T) {
			gen := newTestGenerator(t, tt.build+project)
			gen.SetUseCogBaseImage(false)
			_, actual, _, err := gen.GenerateModelBaseWithSeparateWeights(t.Context(), "r8.im/replicate/cog-test")
			require.NoError(t, err)
			assert.Contains(t, actual, "FROM "+tt.baseImage)

			requirements, err := os.ReadFile(path.Join(gen.tmpDir, "requirements.txt"))
			require.NoError(t, err)
			assert.Contains(t, string(requirements), "--extra-index-url "+tt.index)
		})
	}
}
