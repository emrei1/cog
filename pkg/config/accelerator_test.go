package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/replicate/cog/pkg/accelerator"
)

// fakeProbe simulates the build machine's GPUs for --accelerator auto.
func fakeProbe(kind accelerator.Kind) accelerator.HostProbe {
	return accelerator.HostProbe{
		Exists: func(path string) bool {
			switch kind {
			case accelerator.AMD:
				return path == "/dev/kfd"
			case accelerator.NVIDIA:
				return path == "/proc/driver/nvidia/version"
			}
			return false
		},
		LookPath: func(string) (string, error) { return "", os.ErrNotExist },
	}
}

// loadTestConfig writes cog.yaml (and requirements.txt when given) to a
// temporary project and loads it.
func loadTestConfig(t *testing.T, yaml string, requirementsTxt string, opts ...LoadOption) (*Config, error) {
	t.Helper()
	dir := t.TempDir()
	if requirementsTxt != "" {
		require.NoError(t, os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte(requirementsTxt), 0o644))
	}
	// Never consult the real machine's hardware in tests.
	opts = append([]LoadOption{withHostProbe(fakeProbe(accelerator.CPU))}, opts...)
	result, err := Load(strings.NewReader(yaml), dir, opts...)
	if err != nil {
		return nil, err
	}
	return result.Config, nil
}

func TestLegacyGPUConfigTargetsNVIDIA(t *testing.T) {
	cfg, err := loadTestConfig(t, `
build:
  gpu: true
  python_version: "3.12"
  python_requirements: requirements.txt
`, "torch==2.5.1\n")
	require.NoError(t, err)

	assert.Equal(t, accelerator.NVIDIA, cfg.Accelerator())
	assert.True(t, cfg.Build.GPU)
	assert.Empty(t, cfg.Build.Accelerator, "gpu: true must keep its legacy representation")
	assert.Empty(t, cfg.Build.ROCm)
	assert.NotEmpty(t, cfg.Build.CUDA)
	assert.NotEmpty(t, cfg.Build.CuDNN)
	assert.Equal(t, cfg.Build.CUDA, cfg.AcceleratorVersion())

	baseImage, err := cfg.AcceleratorBaseImage()
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(baseImage, "nvidia/cuda:"), baseImage)
	cudaImage, err := cfg.CUDABaseImageTag()
	require.NoError(t, err)
	assert.Equal(t, cudaImage, baseImage)

	reqs, err := cfg.PythonRequirementsForArch("linux", "amd64", nil)
	require.NoError(t, err)
	assert.Contains(t, reqs, "--extra-index-url https://download.pytorch.org/whl/cu")
	assert.NotContains(t, reqs, "rocm")
}

func TestLegacyGPUConfigMatchesExplicitNVIDIA(t *testing.T) {
	legacy, err := loadTestConfig(t, "build:\n  gpu: true\n  python_version: \"3.12\"\n  python_packages: [\"torch==2.5.1\"]\n", "")
	require.NoError(t, err)
	explicit, err := loadTestConfig(t, "build:\n  accelerator: nvidia\n  python_version: \"3.12\"\n  python_packages: [\"torch==2.5.1\"]\n", "")
	require.NoError(t, err)
	viaFlag, err := loadTestConfig(t, "build:\n  python_version: \"3.12\"\n  python_packages: [\"torch==2.5.1\"]\n", "",
		WithAcceleratorSelection(AcceleratorSelection{Accelerator: "nvidia"}))
	require.NoError(t, err)

	legacyJSON, err := json.Marshal(legacy)
	require.NoError(t, err)
	for _, other := range []*Config{explicit, viaFlag} {
		otherJSON, err := json.Marshal(other)
		require.NoError(t, err)
		assert.JSONEq(t, string(legacyJSON), string(otherJSON))
	}
	assert.NotContains(t, string(legacyJSON), `"accelerator"`)
}

func TestCPUConfigIsUnchanged(t *testing.T) {
	for _, yaml := range []string{
		"build:\n  python_version: \"3.12\"\n",
		"build:\n  gpu: false\n  python_version: \"3.12\"\n",
		"build:\n  accelerator: cpu\n  python_version: \"3.12\"\n",
	} {
		cfg, err := loadTestConfig(t, yaml, "")
		require.NoError(t, err)
		assert.Equal(t, accelerator.CPU, cfg.Accelerator())
		assert.False(t, cfg.Build.GPU)
		assert.Empty(t, cfg.Build.Accelerator)
		assert.Empty(t, cfg.Build.CUDA)
		assert.Empty(t, cfg.AcceleratorVersion())
		baseImage, err := cfg.AcceleratorBaseImage()
		require.NoError(t, err)
		assert.Empty(t, baseImage)
	}
}

func TestCPUConfigStillToleratesCUDAFields(t *testing.T) {
	cfg, err := loadTestConfig(t, "build:\n  python_version: \"3.12\"\n  cuda: \"12.1\"\n", "")
	require.NoError(t, err)
	assert.Equal(t, accelerator.CPU, cfg.Accelerator())
}

func TestAMDConfigDefaults(t *testing.T) {
	cfg, err := loadTestConfig(t, "build:\n  accelerator: amd\n  python_version: \"3.12\"\n", "")
	require.NoError(t, err)

	assert.Equal(t, accelerator.AMD, cfg.Accelerator())
	assert.True(t, cfg.Build.GPU, "AMD images need a GPU")
	assert.Equal(t, "amd", cfg.Build.Accelerator)
	assert.Equal(t, accelerator.DefaultROCmRelease().Version, cfg.Build.ROCm)
	assert.Equal(t, cfg.Build.ROCm, cfg.AcceleratorVersion())
	assert.Empty(t, cfg.Build.CUDA)
	assert.Empty(t, cfg.Build.CuDNN)

	baseImage, err := cfg.AcceleratorBaseImage()
	require.NoError(t, err)
	assert.Equal(t, accelerator.DefaultROCmRelease().BaseImage, baseImage)

	data, err := json.Marshal(cfg)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"accelerator":"amd"`)
	assert.Contains(t, string(data), `"gpu":true`)
	assert.Contains(t, string(data), `"rocm":"`+cfg.Build.ROCm+`"`)
}

func TestAMDConfigRoundTripsThroughImageLabel(t *testing.T) {
	cfg, err := loadTestConfig(t, "build:\n  accelerator: amd\n  rocm: \"6.2\"\n  python_version: \"3.12\"\n", "")
	require.NoError(t, err)

	// `cog predict <image>` reads the config back from the image label.
	data, err := json.Marshal(cfg)
	require.NoError(t, err)
	var fromLabel Config
	require.NoError(t, json.Unmarshal(data, &fromLabel))
	assert.Equal(t, accelerator.AMD, fromLabel.Accelerator())
	assert.Equal(t, "6.2", fromLabel.Build.ROCm)
}

func TestAMDPicksROCmForPinnedTorch(t *testing.T) {
	cfg, err := loadTestConfig(t, `
build:
  accelerator: amd
  python_version: "3.12"
  python_requirements: requirements.txt
`, "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\nnumpy==2.1.0\n")
	require.NoError(t, err)
	assert.Equal(t, "6.2", cfg.Build.ROCm, "newest ROCm that ships torch 2.5.1 and torchvision 0.20.1")

	reqs, err := cfg.PythonRequirementsForArch("linux", "amd64", nil)
	require.NoError(t, err)
	assert.Equal(t, `--extra-index-url https://download.pytorch.org/whl/rocm6.2/
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy==2.1.0`, reqs)
}

func TestAMDUnpinnedTorchUsesROCmIndex(t *testing.T) {
	cfg, err := loadTestConfig(t, `
build:
  accelerator: amd
  rocm: "6.4"
  python_version: "3.12"
  python_requirements: requirements.txt
`, "torch>=2.8\npillow\n")
	require.NoError(t, err)

	reqs, err := cfg.PythonRequirementsForArch("linux", "amd64", nil)
	require.NoError(t, err)
	assert.Equal(t, "--extra-index-url https://download.pytorch.org/whl/rocm6.4/\ntorch>=2.8\npillow", reqs)
}

func TestAMDRespectsUserPackageIndex(t *testing.T) {
	cfg, err := loadTestConfig(t, `
build:
  accelerator: amd
  rocm: "6.0"
  python_version: "3.12"
  python_requirements: requirements.txt
`, "torch==2.5.1 --extra-index-url=https://example.com/whl\n")
	require.NoError(t, err, "a requirement with its own index is not checked against the ROCm catalog")

	reqs, err := cfg.PythonRequirementsForArch("linux", "amd64", nil)
	require.NoError(t, err)
	assert.Equal(t, "--extra-index-url https://example.com/whl\ntorch==2.5.1", reqs)
}

func TestAMDConfigErrors(t *testing.T) {
	tests := []struct {
		name         string
		yaml         string
		requirements string
		wantErr      string
	}{
		{
			name:    "unsupported rocm version",
			yaml:    "build:\n  accelerator: amd\n  rocm: \"6.9\"\n",
			wantErr: `unsupported ROCm version "6.9"`,
		},
		{
			name:    "rocm version below minimum",
			yaml:    "build:\n  accelerator: amd\n  rocm: \"5.7\"\n",
			wantErr: "minimum supported ROCm version is 6.0",
		},
		{
			name:         "torch not built for the requested rocm",
			yaml:         "build:\n  accelerator: amd\n  rocm: \"6.3\"\n  python_requirements: requirements.txt\n",
			requirements: "torch==2.5.1\n",
			wantErr:      "torch==2.5.1 has no ROCm 6.3 build. It is available for ROCm 6.1, 6.2.",
		},
		{
			name:         "torchvision not built for the requested rocm",
			yaml:         "build:\n  accelerator: amd\n  rocm: \"6.2\"\n  python_requirements: requirements.txt\n",
			requirements: "torch==2.5.1\ntorchvision==0.21.0\n",
			wantErr:      "torchvision==0.21.0 has no ROCm 6.2 build",
		},
		{
			name:         "torch without any rocm build",
			yaml:         "build:\n  accelerator: amd\n  python_requirements: requirements.txt\n",
			requirements: "torch==1.13.1\n",
			wantErr:      "cog doesn't know of a ROCm build of torch==1.13.1",
		},
		{
			name:         "torch and torchvision without a common rocm",
			yaml:         "build:\n  accelerator: amd\n  python_requirements: requirements.txt\n",
			requirements: "torch==2.5.1\ntorchvision==0.29.1\n",
			wantErr:      "no ROCm version has builds of both torch==2.5.1 and torchvision==0.29.1",
		},
		{
			name:         "cuda build of torch",
			yaml:         "build:\n  accelerator: amd\n  python_requirements: requirements.txt\n",
			requirements: "torch==2.5.1+cu121\n",
			wantErr:      "torch==2.5.1+cu121 is a CUDA/CPU build of PyTorch",
		},
		{
			name:    "cuda with amd",
			yaml:    "build:\n  accelerator: amd\n  cuda: \"12.1\"\n",
			wantErr: `invalid build.cuda "12.1": only applies to accelerator nvidia, but this build targets amd`,
		},
		{
			name:    "cudnn with amd",
			yaml:    "build:\n  accelerator: amd\n  cudnn: \"9\"\n",
			wantErr: "build.cudnn",
		},
		{
			name:    "rocm with nvidia",
			yaml:    "build:\n  gpu: true\n  rocm: \"6.2\"\n",
			wantErr: `invalid build.rocm "6.2": only applies to accelerator amd, but this build targets nvidia`,
		},
		{
			name:    "rocm with cpu",
			yaml:    "build:\n  rocm: \"6.2\"\n",
			wantErr: "only applies to accelerator amd, but this build targets cpu",
		},
		{
			name:    "gpu false with amd",
			yaml:    "build:\n  gpu: false\n  accelerator: amd\n",
			wantErr: "conflicts with gpu: false",
		},
		{
			name:    "gpu true with cpu",
			yaml:    "build:\n  gpu: true\n  accelerator: cpu\n",
			wantErr: "conflicts with gpu: true",
		},
		{
			name:    "auto in cog.yaml",
			yaml:    "build:\n  accelerator: auto\n",
			wantErr: "only accepted on the command line",
		},
		{
			name:    "unknown accelerator",
			yaml:    "build:\n  accelerator: intel\n",
			wantErr: "accelerator",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yaml := strings.Replace(tt.yaml, "build:\n", "build:\n  python_version: \"3.13\"\n", 1)
			_, err := loadTestConfig(t, yaml, tt.requirements)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestAMDGPUTrueIsConsistent(t *testing.T) {
	cfg, err := loadTestConfig(t, "build:\n  gpu: true\n  accelerator: amd\n  python_version: \"3.13\"\n", "")
	require.NoError(t, err)
	assert.Equal(t, accelerator.AMD, cfg.Accelerator())
}

func TestCompleteValidatesAcceleratorFieldsOnDirectConfigs(t *testing.T) {
	cfg := &Config{Build: &Build{GPU: true, Accelerator: "amd", CUDA: "12.1", PythonVersion: "3.12"}}
	require.ErrorContains(t, cfg.Complete(""), "only applies to accelerator nvidia")

	cfg = &Config{Build: &Build{GPU: true, Accelerator: "intel", PythonVersion: "3.12"}}
	require.ErrorContains(t, cfg.Complete(""), `unsupported accelerator "intel"`)
}

func TestAcceleratorSelectionPrecedence(t *testing.T) {
	const nvidiaYAML = "build:\n  gpu: true\n  cuda: \"11.8\"\n  python_version: \"3.12\"\n"
	const cpuYAML = "build:\n  python_version: \"3.12\"\n"
	const undeclaredYAML = "predict: predict.py:Predictor\n"

	tests := []struct {
		name        string
		yaml        string
		sel         AcceleratorSelection
		probe       accelerator.Kind
		want        accelerator.Kind
		wantVersion string
		wantErr     string
	}{
		{name: "no flags keeps cog.yaml", yaml: nvidiaYAML, want: accelerator.NVIDIA, wantVersion: "11.8"},
		{name: "explicit flag overrides gpu: true", yaml: nvidiaYAML, sel: AcceleratorSelection{Accelerator: "amd"}, want: accelerator.AMD, wantVersion: accelerator.DefaultROCmRelease().Version},
		{name: "explicit flag overrides cpu", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "nvidia"}, want: accelerator.NVIDIA},
		{name: "explicit cpu overrides gpu: true", yaml: nvidiaYAML, sel: AcceleratorSelection{Accelerator: "cpu"}, want: accelerator.CPU},
		{name: "explicit amd needs no amd hardware", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "amd"}, probe: accelerator.NVIDIA, want: accelerator.AMD},
		{name: "version flag with amd", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "amd", Version: "6.2"}, want: accelerator.AMD, wantVersion: "6.2"},
		{name: "version flag with nvidia", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "nvidia", Version: "12.1"}, want: accelerator.NVIDIA, wantVersion: "12.1"},
		{name: "version flag overrides cog.yaml cuda", yaml: nvidiaYAML, sel: AcceleratorSelection{Version: "12.1"}, want: accelerator.NVIDIA, wantVersion: "12.1"},
		{name: "auto prefers cog.yaml over hardware", yaml: nvidiaYAML, sel: AcceleratorSelection{Accelerator: "auto"}, probe: accelerator.AMD, want: accelerator.NVIDIA, wantVersion: "11.8"},
		{name: "auto respects explicit gpu: false", yaml: cpuYAML + "  gpu: false\n", sel: AcceleratorSelection{Accelerator: "auto"}, probe: accelerator.AMD, want: accelerator.CPU},
		{name: "auto detects amd", yaml: undeclaredYAML, sel: AcceleratorSelection{Accelerator: "auto"}, probe: accelerator.AMD, want: accelerator.AMD},
		{name: "auto detects nvidia", yaml: undeclaredYAML, sel: AcceleratorSelection{Accelerator: "auto"}, probe: accelerator.NVIDIA, want: accelerator.NVIDIA},
		{name: "auto falls back to cpu", yaml: undeclaredYAML, sel: AcceleratorSelection{Accelerator: "auto"}, probe: accelerator.CPU, want: accelerator.CPU},
		{name: "invalid accelerator flag", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "tpu"}, wantErr: `invalid --accelerator: unsupported accelerator "tpu"`},
		{name: "version flag on cpu", yaml: cpuYAML, sel: AcceleratorSelection{Version: "12.1"}, wantErr: "--accelerator-version requires a GPU accelerator"},
		{name: "version flag with explicit cpu", yaml: nvidiaYAML, sel: AcceleratorSelection{Accelerator: "cpu", Version: "12.1"}, wantErr: "--accelerator-version requires a GPU accelerator"},
		{name: "unsupported rocm version flag", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "amd", Version: "5.0"}, wantErr: "minimum supported ROCm version is 6.0"},
		{name: "invalid cuda version flag", yaml: cpuYAML, sel: AcceleratorSelection{Accelerator: "nvidia", Version: "10.2"}, wantErr: "minimum supported CUDA version is 11"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := tt.probe
			if probe == "" {
				probe = accelerator.CPU
			}
			cfg, err := loadTestConfig(t, tt.yaml, "", WithAcceleratorSelection(tt.sel), withHostProbe(fakeProbe(probe)))
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, cfg.Accelerator())
			assert.Equal(t, tt.want.IsGPU(), cfg.Build.GPU)
			if tt.wantVersion != "" {
				assert.Equal(t, tt.wantVersion, cfg.AcceleratorVersion())
			}
			if tt.want == accelerator.AMD {
				assert.Empty(t, cfg.Build.CUDA, "CUDA pins from cog.yaml don't apply to AMD")
				assert.Empty(t, cfg.Build.CuDNN)
			}
		})
	}
}

func TestAutoWithBothVendorsIsAnError(t *testing.T) {
	probe := accelerator.HostProbe{
		Exists:   func(path string) bool { return path == "/dev/kfd" },
		LookPath: func(file string) (string, error) { return "/usr/bin/" + file, nil },
	}
	_, err := loadTestConfig(t, "predict: predict.py:Predictor\n", "",
		WithAcceleratorSelection(AcceleratorSelection{Accelerator: "auto"}), withHostProbe(probe))
	require.ErrorContains(t, err, "found both NVIDIA and AMD GPUs")
}
