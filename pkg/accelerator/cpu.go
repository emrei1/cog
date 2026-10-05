package accelerator

type cpuBackend struct{}

func (cpuBackend) Kind() Kind                 { return CPU }
func (cpuBackend) DisplayName() string        { return "CPU" }
func (cpuBackend) ToolkitName() string        { return "" }
func (cpuBackend) SupportsCogBaseImage() bool { return true }

// ImageEnv keeps the environment CPU images have always had, so generated
// Dockerfiles for existing CPU projects stay byte-for-byte identical. The
// NVIDIA paths are inert without an NVIDIA runtime.
func (cpuBackend) ImageEnv() []string {
	return nvidiaImageEnv()
}

// RunConfig follows the historical behavior of an explicit --gpus flag on a
// CPU image, which is to request NVIDIA GPUs.
func (cpuBackend) RunConfig(gpus string) (RunConfig, error) {
	return nvidiaBackend{}.RunConfig(gpus)
}
