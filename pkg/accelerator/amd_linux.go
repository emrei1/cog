//go:build linux

package accelerator

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"syscall"
)

// localAMDHost inspects the device nodes of the machine Cog runs on, which
// is also where the Docker daemon runs for local `cog predict` and friends.
type localAMDHost struct{}

func (localAMDHost) DeviceGroups() ([]string, error) {
	kfd, err := os.Stat(amdKFDDevice)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, amdDevicesNotFound(amdKFDDevice + " does not exist (is the amdgpu kernel driver loaded?)")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to inspect %s: %w", amdKFDDevice, err)
	}
	renderNodes, err := filepath.Glob(filepath.Join(amdDRIDir, "renderD*"))
	if err != nil {
		return nil, fmt.Errorf("failed to list %s: %w", amdDRIDir, err)
	}
	if len(renderNodes) == 0 {
		return nil, amdDevicesNotFound("no render nodes found in " + amdDRIDir)
	}

	groups := []string{}
	addGroup := func(info fs.FileInfo) {
		if st, ok := info.Sys().(*syscall.Stat_t); ok {
			gid := strconv.FormatUint(uint64(st.Gid), 10)
			if !slices.Contains(groups, gid) {
				groups = append(groups, gid)
			}
		}
	}
	addGroup(kfd)
	for _, node := range renderNodes {
		info, err := os.Stat(node)
		if err != nil {
			return nil, fmt.Errorf("failed to inspect %s: %w", node, err)
		}
		addGroup(info)
	}
	return groups, nil
}
