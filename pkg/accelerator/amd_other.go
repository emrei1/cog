//go:build !linux

package accelerator

import "runtime"

// localAMDHost reports that AMD GPUs are unavailable: ROCm containers need
// the amdgpu kernel driver, which only exists on Linux hosts.
type localAMDHost struct{}

func (localAMDHost) DeviceGroups() ([]string, error) {
	return nil, amdDevicesNotFound("AMD GPU containers are only supported on Linux hosts, not " + runtime.GOOS)
}
