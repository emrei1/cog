package accelerator

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrAMDDevicesNotFound is returned when an AMD image is run on a host that
// does not expose the ROCm kernel interfaces.
var ErrAMDDevicesNotFound = errors.New("AMD GPU devices not found")

const (
	// amdKFDDevice is the ROCm compute interface provided by the amdgpu
	// kernel driver.
	amdKFDDevice = "/dev/kfd"
	// amdDRIDir holds the per-GPU render nodes (renderD*).
	amdDRIDir = "/dev/dri"
	// rocrVisibleDevices restricts which GPUs the ROCm runtime enumerates,
	// the ROCm counterpart of NVIDIA_VISIBLE_DEVICES.
	rocrVisibleDevices = "ROCR_VISIBLE_DEVICES"
)

// amdHost reports the AMD GPU device nodes on the machine running Docker.
type amdHost interface {
	// DeviceGroups returns the numeric group IDs that own /dev/kfd and the
	// render nodes. It returns an error wrapping ErrAMDDevicesNotFound if
	// the ROCm kernel interfaces are missing.
	DeviceGroups() ([]string, error)
}

type amdBackend struct {
	host amdHost
}

func newAMDBackend() amdBackend {
	return amdBackend{host: localAMDHost{}}
}

func (amdBackend) Kind() Kind          { return AMD }
func (amdBackend) DisplayName() string { return "AMD ROCm" }
func (amdBackend) ToolkitName() string { return "ROCm" }

// SupportsCogBaseImage is false because cog-base images are built on the
// NVIDIA CUDA images.
func (amdBackend) SupportsCogBaseImage() bool { return false }

// ImageEnv does not add /opt/rocm/lib to LD_LIBRARY_PATH: PyTorch ROCm wheels
// bundle the ROCm libraries they were built against, and LD_LIBRARY_PATH
// would take precedence over the wheel's RUNPATH.
func (amdBackend) ImageEnv() []string {
	return []string{
		"ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu",
		"ENV PATH=$PATH:/opt/rocm/bin",
	}
}

// RunConfig passes the ROCm device nodes through to the container, as AMD
// documents for Docker: --device /dev/kfd --device /dev/dri plus membership
// of the groups that own them. It needs no container toolkit on the host.
// GPU selection is applied with ROCR_VISIBLE_DEVICES because every GPU
// shares /dev/kfd.
func (b amdBackend) RunConfig(gpus string) (RunConfig, error) {
	if gpus == "" {
		return RunConfig{}, nil
	}
	sel, err := parseGPUSelection(gpus)
	if err != nil {
		return RunConfig{}, err
	}
	var visible string
	switch {
	case sel.all, sel.count == -1:
	case sel.deviceIDs != nil:
		visible = strings.Join(sel.deviceIDs, ",")
	case sel.count > 0:
		ids := make([]string, sel.count)
		for i := range ids {
			ids[i] = strconv.Itoa(i)
		}
		visible = strings.Join(ids, ",")
	default:
		return RunConfig{}, fmt.Errorf("invalid GPU specification: %q", gpus)
	}

	groups, err := b.host.DeviceGroups()
	if err != nil {
		return RunConfig{}, err
	}
	cfg := RunConfig{
		Devices:  []string{amdKFDDevice, amdDRIDir},
		GroupAdd: groups,
	}
	if visible != "" {
		cfg.Env = []string{rocrVisibleDevices + "=" + visible}
	}
	return cfg, nil
}

func amdDevicesNotFound(detail string) error {
	return fmt.Errorf(`%w: %s

Running an image built with --accelerator amd needs a Linux host with:
  - an AMD GPU supported by ROCm
  - the amdgpu kernel driver loaded, which provides %s and %s/renderD*
  - permission for the Docker daemon to access those devices

The image only contains the ROCm userspace libraries; the kernel driver
must come from the host. See https://rocm.docs.amd.com for driver installation`,
		ErrAMDDevicesNotFound, detail, amdKFDDevice, amdDRIDir)
}
