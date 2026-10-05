// Package accelerator describes the hardware accelerators a Cog image can
// target (CPU, NVIDIA CUDA, AMD ROCm) behind a single Backend interface.
//
// A Backend owns the vendor-specific facts Cog needs at two separate points:
//
//   - build time: the toolkit it uses (CUDA, ROCm), whether Cog's pre-built
//     base images cover it, and the environment baked into the image;
//   - run time: how the host's devices are exposed to the container.
//
// Build-time version resolution that depends on cog.yaml (for example
// matching a CUDA version to the pinned torch) lives in pkg/config, which
// dispatches on the Kind defined here. The rest of Cog asks the Backend
// instead of checking for a specific vendor.
package accelerator

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Kind identifies an accelerator target.
type Kind string

const (
	// Auto selects the accelerator declared in cog.yaml, falling back to
	// hardware detection on the build machine. Only valid as a CLI selection.
	Auto Kind = "auto"
	// CPU builds an image without an accelerator toolkit.
	CPU Kind = "cpu"
	// NVIDIA targets NVIDIA GPUs through CUDA.
	NVIDIA Kind = "nvidia"
	// AMD targets AMD GPUs through ROCm.
	AMD Kind = "amd"
)

// Targets lists the concrete accelerators an image can be built for.
var Targets = []Kind{CPU, NVIDIA, AMD}

// Parse parses an accelerator name. Auto is accepted only when allowAuto is
// true, because "auto" describes how to choose a target, not a target.
func Parse(s string, allowAuto bool) (Kind, error) {
	k := Kind(strings.ToLower(strings.TrimSpace(s)))
	if k == Auto && allowAuto {
		return k, nil
	}
	if slices.Contains(Targets, k) {
		return k, nil
	}
	valid := targetNames()
	if allowAuto {
		valid = append([]string{string(Auto)}, valid...)
	}
	return "", fmt.Errorf("unsupported accelerator %q: must be one of %s", s, strings.Join(valid, ", "))
}

// IsGPU reports whether the accelerator needs a GPU at run time.
func (k Kind) IsGPU() bool {
	return k == NVIDIA || k == AMD
}

func (k Kind) String() string {
	return string(k)
}

func targetNames() []string {
	names := make([]string, len(Targets))
	for i, t := range Targets {
		names[i] = string(t)
	}
	return names
}

// Backend is the vendor-specific behavior of one accelerator target.
type Backend interface {
	// Kind returns the accelerator this backend implements.
	Kind() Kind

	// DisplayName is a human readable name, e.g. "AMD ROCm".
	DisplayName() string

	// ToolkitName is the name of the userspace toolkit selected by
	// --accelerator-version (e.g. "CUDA", "ROCm"), or "" when the backend
	// has no versioned toolkit.
	ToolkitName() string

	// SupportsCogBaseImage reports whether Cog publishes pre-built
	// cog-base images for this accelerator.
	SupportsCogBaseImage() bool

	// ImageEnv returns Dockerfile ENV instructions written into images that
	// are built from a generated Dockerfile (not from a cog-base image).
	ImageEnv() []string

	// RunConfig returns how to expose the host's accelerators to a
	// container. gpus uses the `docker run --gpus` syntax: "all", a count,
	// or "device=ID[,ID...]".
	RunConfig(gpus string) (RunConfig, error)
}

// RunConfig is a Docker-agnostic description of the device configuration a
// container needs. pkg/docker translates it into a Docker host config.
type RunConfig struct {
	// DeviceRequests are driver-backed requests (docker run --gpus).
	DeviceRequests []DeviceRequest
	// Devices are host device paths passed through unchanged (docker run --device).
	Devices []string
	// GroupAdd are supplementary groups for the container process (docker run --group-add).
	GroupAdd []string
	// Env are extra environment variables in KEY=VALUE form.
	Env []string
}

// DeviceRequest mirrors Docker's container.DeviceRequest.
type DeviceRequest struct {
	Driver       string
	Count        int
	DeviceIDs    []string
	Capabilities [][]string
}

// For returns the backend for a concrete accelerator.
func For(k Kind) (Backend, error) {
	switch k {
	case CPU:
		return cpuBackend{}, nil
	case NVIDIA:
		return nvidiaBackend{}, nil
	case AMD:
		return newAMDBackend(), nil
	case Auto:
		return nil, fmt.Errorf("accelerator %q must be resolved to a concrete accelerator before use", k)
	default:
		return nil, fmt.Errorf("unsupported accelerator %q: this image may have been built by a newer version of Cog", k)
	}
}

// ForRuntime returns the backend that exposes GPUs to a container running an
// image built for k. Images without a GPU accelerator have always used
// NVIDIA semantics for an explicit --gpus flag, so CPU (and an unset
// accelerator) maps to NVIDIA here.
func ForRuntime(k Kind) (Backend, error) {
	if k == "" || k == CPU {
		return For(NVIDIA)
	}
	return For(k)
}

// MustFor is For for kinds known to be valid.
func MustFor(k Kind) Backend {
	b, err := For(k)
	if err != nil {
		panic(err)
	}
	return b
}

// gpuSelection is a parsed `--gpus` value.
type gpuSelection struct {
	all       bool
	count     int
	deviceIDs []string
}

func parseGPUSelection(gpus string) (gpuSelection, error) {
	if gpus == "all" {
		return gpuSelection{all: true}, nil
	}
	if count, err := strconv.Atoi(gpus); err == nil {
		return gpuSelection{count: count}, nil
	}
	if after, ok := strings.CutPrefix(gpus, "device="); ok && after != "" {
		return gpuSelection{deviceIDs: strings.Split(after, ",")}, nil
	}
	return gpuSelection{}, fmt.Errorf("invalid GPU specification: %q", gpus)
}
