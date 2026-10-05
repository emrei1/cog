package accelerator

type nvidiaBackend struct{}

func (nvidiaBackend) Kind() Kind                 { return NVIDIA }
func (nvidiaBackend) DisplayName() string        { return "NVIDIA CUDA" }
func (nvidiaBackend) ToolkitName() string        { return "CUDA" }
func (nvidiaBackend) SupportsCogBaseImage() bool { return true }

func (nvidiaBackend) ImageEnv() []string {
	return nvidiaImageEnv()
}

func nvidiaImageEnv() []string {
	return []string{
		"ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin",
		"ENV NVIDIA_DRIVER_CAPABILITIES=all",
	}
}

// RunConfig requests GPUs from the NVIDIA Container Toolkit, equivalent to
// `docker run --gpus`.
func (nvidiaBackend) RunConfig(gpus string) (RunConfig, error) {
	if gpus == "" {
		return RunConfig{}, nil
	}
	sel, err := parseGPUSelection(gpus)
	if err != nil {
		return RunConfig{}, err
	}
	req := DeviceRequest{
		Driver:       "nvidia",
		Capabilities: [][]string{{"gpu"}},
	}
	switch {
	case sel.all:
		req.Count = -1
	case sel.deviceIDs != nil:
		req.DeviceIDs = sel.deviceIDs
	default:
		req.Count = sel.count
	}
	return RunConfig{DeviceRequests: []DeviceRequest{req}}, nil
}
