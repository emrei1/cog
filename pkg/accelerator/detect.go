package accelerator

import (
	"fmt"
	"os"
	"os/exec"
)

// HostProbe is the view of the build machine used by Detect. Tests replace
// it to simulate hardware.
type HostProbe struct {
	// Exists reports whether a file or device node exists.
	Exists func(path string) bool
	// LookPath reports whether an executable is on PATH.
	LookPath func(file string) (string, error)
}

// LocalHost probes the machine Cog is running on.
func LocalHost() HostProbe {
	return HostProbe{
		Exists: func(path string) bool {
			_, err := os.Stat(path)
			return err == nil
		},
		LookPath: exec.LookPath,
	}
}

// Detect picks an accelerator from the GPUs visible on the build machine.
// It returns CPU when no GPU is found, and an error when both NVIDIA and
// AMD GPUs are present, because the choice would then be arbitrary.
//
// Detection describes the build machine, which is not necessarily the
// machine that runs the image, so builds that rely on it are only as
// reproducible as the machine they run on.
func Detect(host HostProbe) (Kind, error) {
	nvidia := host.Exists("/proc/driver/nvidia/version") || hasExecutable(host, "nvidia-smi")
	amd := host.Exists(amdKFDDevice)
	switch {
	case nvidia && amd:
		return "", fmt.Errorf("found both NVIDIA and AMD GPUs on this machine; select one with --accelerator nvidia or --accelerator amd")
	case nvidia:
		return NVIDIA, nil
	case amd:
		return AMD, nil
	default:
		return CPU, nil
	}
}

func hasExecutable(host HostProbe, name string) bool {
	if host.LookPath == nil {
		return false
	}
	_, err := host.LookPath(name)
	return err == nil
}
