package accelerator

import (
	"errors"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParse(t *testing.T) {
	tests := []struct {
		input     string
		allowAuto bool
		want      Kind
		wantErr   string
	}{
		{input: "cpu", want: CPU},
		{input: "nvidia", want: NVIDIA},
		{input: "amd", want: AMD},
		{input: " AMD ", want: AMD},
		{input: "auto", allowAuto: true, want: Auto},
		{input: "auto", wantErr: "must be one of cpu, nvidia, amd"},
		{input: "intel", allowAuto: true, wantErr: `unsupported accelerator "intel": must be one of auto, cpu, nvidia, amd`},
		{input: "cuda", wantErr: `unsupported accelerator "cuda"`},
		{input: "", wantErr: `unsupported accelerator ""`},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := Parse(tt.input, tt.allowAuto)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestIsGPU(t *testing.T) {
	assert.False(t, CPU.IsGPU())
	assert.True(t, NVIDIA.IsGPU())
	assert.True(t, AMD.IsGPU())
	assert.False(t, Auto.IsGPU())
}

func TestFor(t *testing.T) {
	for _, k := range Targets {
		b, err := For(k)
		require.NoError(t, err)
		assert.Equal(t, k, b.Kind())
	}

	_, err := For(Auto)
	require.ErrorContains(t, err, "must be resolved")

	_, err = For("intel")
	require.ErrorContains(t, err, "newer version of Cog")
}

func TestForRuntimeKeepsNVIDIASemanticsForNonGPUImages(t *testing.T) {
	for _, k := range []Kind{"", CPU} {
		b, err := ForRuntime(k)
		require.NoError(t, err)
		assert.Equal(t, NVIDIA, b.Kind())
	}
	b, err := ForRuntime(AMD)
	require.NoError(t, err)
	assert.Equal(t, AMD, b.Kind())
}

func TestBackendBuildProperties(t *testing.T) {
	legacyEnv := []string{
		"ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin",
		"ENV NVIDIA_DRIVER_CAPABILITIES=all",
	}
	assert.Equal(t, legacyEnv, MustFor(CPU).ImageEnv())
	assert.Equal(t, legacyEnv, MustFor(NVIDIA).ImageEnv())

	amdEnv := MustFor(AMD).ImageEnv()
	for _, line := range amdEnv {
		assert.NotContains(t, line, "nvidia")
		assert.NotContains(t, line, "/opt/rocm/lib")
	}

	assert.True(t, MustFor(CPU).SupportsCogBaseImage())
	assert.True(t, MustFor(NVIDIA).SupportsCogBaseImage())
	assert.False(t, MustFor(AMD).SupportsCogBaseImage())

	assert.Equal(t, "", MustFor(CPU).ToolkitName())
	assert.Equal(t, "CUDA", MustFor(NVIDIA).ToolkitName())
	assert.Equal(t, "ROCm", MustFor(AMD).ToolkitName())
}

func TestNVIDIARunConfig(t *testing.T) {
	tests := []struct {
		gpus    string
		want    RunConfig
		wantErr string
	}{
		{gpus: "", want: RunConfig{}},
		{gpus: "all", want: RunConfig{DeviceRequests: []DeviceRequest{{Driver: "nvidia", Count: -1, Capabilities: [][]string{{"gpu"}}}}}},
		{gpus: "2", want: RunConfig{DeviceRequests: []DeviceRequest{{Driver: "nvidia", Count: 2, Capabilities: [][]string{{"gpu"}}}}}},
		{gpus: "device=0,1", want: RunConfig{DeviceRequests: []DeviceRequest{{Driver: "nvidia", DeviceIDs: []string{"0", "1"}, Capabilities: [][]string{{"gpu"}}}}}},
		{gpus: "bogus", wantErr: `invalid GPU specification: "bogus"`},
	}
	for _, tt := range tests {
		t.Run(tt.gpus, func(t *testing.T) {
			got, err := MustFor(NVIDIA).RunConfig(tt.gpus)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)

			// An explicit --gpus on a CPU image has always meant NVIDIA.
			cpu, err := MustFor(CPU).RunConfig(tt.gpus)
			require.NoError(t, err)
			assert.Equal(t, tt.want, cpu)
		})
	}
}

type fakeAMDHost struct {
	groups []string
	err    error
}

func (h fakeAMDHost) DeviceGroups() ([]string, error) { return h.groups, h.err }

func TestAMDRunConfig(t *testing.T) {
	backend := amdBackend{host: fakeAMDHost{groups: []string{"44", "109"}}}
	devices := []string{"/dev/kfd", "/dev/dri"}
	groups := []string{"44", "109"}

	tests := []struct {
		gpus    string
		want    RunConfig
		wantErr string
	}{
		{gpus: "", want: RunConfig{}},
		{gpus: "all", want: RunConfig{Devices: devices, GroupAdd: groups}},
		{gpus: "-1", want: RunConfig{Devices: devices, GroupAdd: groups}},
		{gpus: "2", want: RunConfig{Devices: devices, GroupAdd: groups, Env: []string{"ROCR_VISIBLE_DEVICES=0,1"}}},
		{gpus: "device=1,3", want: RunConfig{Devices: devices, GroupAdd: groups, Env: []string{"ROCR_VISIBLE_DEVICES=1,3"}}},
		{gpus: "0", wantErr: `invalid GPU specification: "0"`},
		{gpus: "device=", wantErr: `invalid GPU specification: "device="`},
	}
	for _, tt := range tests {
		t.Run(tt.gpus, func(t *testing.T) {
			got, err := backend.RunConfig(tt.gpus)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
			assert.Empty(t, got.DeviceRequests, "AMD must not depend on an NVIDIA-style device driver")
		})
	}
}

func TestAMDRunConfigWithoutHostDevices(t *testing.T) {
	backend := amdBackend{host: fakeAMDHost{err: amdDevicesNotFound("/dev/kfd does not exist")}}
	_, err := backend.RunConfig("all")
	require.ErrorIs(t, err, ErrAMDDevicesNotFound)
	assert.ErrorContains(t, err, "/dev/kfd does not exist")
	assert.ErrorContains(t, err, "amdgpu kernel driver")

	// Running without GPUs never touches the host devices.
	got, err := backend.RunConfig("")
	require.NoError(t, err)
	assert.Equal(t, RunConfig{}, got)
}

func TestLookupROCm(t *testing.T) {
	r, err := LookupROCm("6.2")
	require.NoError(t, err)
	assert.Equal(t, "6.2", r.Version)
	assert.Equal(t, "rocm/dev-ubuntu-22.04:6.2", r.BaseImage)
	assert.Equal(t, "https://download.pytorch.org/whl/rocm6.2/", r.TorchIndexURL)

	r, err = LookupROCm("6.2.4")
	require.NoError(t, err)
	assert.Equal(t, "6.2.4", r.Version)

	r, err = LookupROCm("6.2.0")
	require.NoError(t, err)
	assert.Equal(t, "6.2", r.Version)

	_, err = LookupROCm("5.7")
	require.ErrorContains(t, err, "minimum supported ROCm version is 6.0")

	_, err = LookupROCm("6.9")
	require.ErrorContains(t, err, `unsupported ROCm version "6.9": supported versions are 6.0, 6.1, 6.2, 6.2.4`)

	_, err = LookupROCm("latest")
	require.ErrorContains(t, err, "unsupported ROCm version")
}

func TestROCmReleasesAreConsistent(t *testing.T) {
	releases := ROCmReleases()
	require.NotEmpty(t, releases)
	for i, r := range releases {
		assert.NotEmpty(t, r.Torch, r.Version)
		assert.NotEmpty(t, r.Torchvision, r.Version)
		assert.Equal(t, "rocm/dev-ubuntu-22.04:"+r.Version, r.BaseImage)
		assert.Equal(t, "https://download.pytorch.org/whl/rocm"+r.Version+"/", r.TorchIndexURL)
		if i > 0 {
			assert.NotEqual(t, releases[i-1].Version, r.Version)
		}
	}
	assert.Equal(t, releases[len(releases)-1], DefaultROCmRelease())
}

func TestROCmReleasesShipping(t *testing.T) {
	var versions []string
	for _, r := range ROCmReleasesShipping("torch", "2.5.1") {
		versions = append(versions, r.Version)
	}
	assert.Equal(t, []string{"6.1", "6.2"}, versions)

	assert.Empty(t, ROCmReleasesShipping("torch", "1.13.1"))
	assert.Empty(t, ROCmReleasesShipping("tensorflow", "2.16.1"))
}

func TestDetect(t *testing.T) {
	probe := func(files []string, executables []string) HostProbe {
		return HostProbe{
			Exists: func(path string) bool {
				return slices.Contains(files, path)
			},
			LookPath: func(file string) (string, error) {
				if slices.Contains(executables, file) {
					return "/usr/bin/" + file, nil
				}
				return "", errors.New("not found")
			},
		}
	}

	tests := []struct {
		name        string
		files       []string
		executables []string
		want        Kind
		wantErr     string
	}{
		{name: "no gpu", want: CPU},
		{name: "nvidia driver", files: []string{"/proc/driver/nvidia/version"}, want: NVIDIA},
		{name: "nvidia-smi", executables: []string{"nvidia-smi"}, want: NVIDIA},
		{name: "amd", files: []string{"/dev/kfd"}, want: AMD},
		{name: "both", files: []string{"/dev/kfd"}, executables: []string{"nvidia-smi"}, wantErr: "found both NVIDIA and AMD GPUs"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Detect(probe(tt.files, tt.executables))
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
