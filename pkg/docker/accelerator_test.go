package docker

import (
	"testing"

	"github.com/docker/docker/api/types/container"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/replicate/cog/pkg/accelerator"
	"github.com/replicate/cog/pkg/docker/command"
)

func TestApplyAcceleratorRunConfigNVIDIA(t *testing.T) {
	// Images built before accelerators existed have no accelerator; they and
	// CPU images keep the NVIDIA semantics of an explicit --gpus.
	for _, kind := range []accelerator.Kind{"", accelerator.CPU, accelerator.NVIDIA} {
		t.Run(string(kind), func(t *testing.T) {
			containerCfg := &container.Config{}
			hostCfg := &container.HostConfig{}
			err := applyAcceleratorRunConfig(command.RunOptions{GPUs: "all", Accelerator: kind}, containerCfg, hostCfg)
			require.NoError(t, err)

			assert.Equal(t, []container.DeviceRequest{{
				Driver:       "nvidia",
				Count:        -1,
				Capabilities: [][]string{{"gpu"}},
			}}, hostCfg.DeviceRequests)
			assert.Empty(t, hostCfg.Devices)
			assert.Empty(t, hostCfg.GroupAdd)
			assert.Empty(t, containerCfg.Env)
		})
	}
}

func TestApplyAcceleratorRunConfigInvalidGPUs(t *testing.T) {
	err := applyAcceleratorRunConfig(command.RunOptions{GPUs: "lots"}, &container.Config{}, &container.HostConfig{})
	require.ErrorContains(t, err, `invalid GPU specification: "lots"`)
}

func TestApplyAcceleratorRunConfigAMD(t *testing.T) {
	containerCfg := &container.Config{Env: []string{"FOO=bar"}}
	hostCfg := &container.HostConfig{}
	err := applyAcceleratorRunConfig(command.RunOptions{GPUs: "device=1", Accelerator: accelerator.AMD}, containerCfg, hostCfg)

	// This runs on the test machine's real devices, which usually have no AMD GPU.
	if err != nil {
		require.ErrorIs(t, err, accelerator.ErrAMDDevicesNotFound)
		assert.Empty(t, hostCfg.Devices)
		return
	}
	assert.Empty(t, hostCfg.DeviceRequests, "AMD must not request the nvidia device driver")
	assert.Equal(t, []container.DeviceMapping{
		{PathOnHost: "/dev/kfd", PathInContainer: "/dev/kfd", CgroupPermissions: "rwm"},
		{PathOnHost: "/dev/dri", PathInContainer: "/dev/dri", CgroupPermissions: "rwm"},
	}, hostCfg.Devices)
	assert.NotEmpty(t, hostCfg.GroupAdd)
	assert.Equal(t, []string{"FOO=bar", "ROCR_VISIBLE_DEVICES=1"}, containerCfg.Env)
}

func TestApplyAcceleratorRunConfigUnknownAccelerator(t *testing.T) {
	err := applyAcceleratorRunConfig(command.RunOptions{GPUs: "all", Accelerator: "intel"}, &container.Config{}, &container.HostConfig{})
	require.ErrorContains(t, err, `unsupported accelerator "intel"`)
}
