package config

import (
	"fmt"
	"strings"

	"github.com/replicate/cog/pkg/accelerator"
	"github.com/replicate/cog/pkg/util/console"
)

// AcceleratorSelection is an accelerator choice made outside cog.yaml,
// typically by `cog build --accelerator ... --accelerator-version ...`.
// It overrides build.gpu, build.accelerator and the toolkit version fields.
type AcceleratorSelection struct {
	// Accelerator is "auto", "cpu", "nvidia", "amd", or "" to keep cog.yaml's choice.
	Accelerator string
	// Version is the toolkit version for the selected accelerator (CUDA for
	// nvidia, ROCm for amd), or "" to resolve it automatically.
	Version string
}

// IsZero reports whether the selection leaves cog.yaml unchanged.
func (s AcceleratorSelection) IsZero() bool {
	return s.Accelerator == "" && s.Version == ""
}

// toolkitField is a cog.yaml field that only applies to one accelerator.
type toolkitField struct {
	name  string
	owner accelerator.Kind
	get   func(*Build) *string
	// legacyCPU marks fields that CPU configs have always been allowed to
	// set (and that are ignored on CPU), so they stay valid there.
	legacyCPU bool
}

var toolkitFields = []toolkitField{
	{name: "cuda", owner: accelerator.NVIDIA, get: func(b *Build) *string { return &b.CUDA }, legacyCPU: true},
	{name: "cudnn", owner: accelerator.NVIDIA, get: func(b *Build) *string { return &b.CuDNN }, legacyCPU: true},
	{name: "rocm", owner: accelerator.AMD, get: func(b *Build) *string { return &b.ROCm }},
}

// AcceleratorKind returns the accelerator the image targets. build.accelerator
// wins when set; otherwise build.gpu: true means NVIDIA, as it always has.
func (b *Build) AcceleratorKind() accelerator.Kind {
	if b == nil {
		return accelerator.CPU
	}
	if b.Accelerator != "" {
		return accelerator.Kind(strings.ToLower(strings.TrimSpace(b.Accelerator)))
	}
	if b.GPU {
		return accelerator.NVIDIA
	}
	return accelerator.CPU
}

// setAccelerator stores k in its canonical form: CPU and NVIDIA are written
// with the legacy gpu field alone, so configs (and image labels) for those
// targets look exactly as they did before build.accelerator existed.
func (b *Build) setAccelerator(k accelerator.Kind) {
	b.GPU = k.IsGPU()
	if k == accelerator.CPU || k == accelerator.NVIDIA {
		b.Accelerator = ""
	} else {
		b.Accelerator = string(k)
	}
}

// Accelerator returns the accelerator the image targets.
func (c *Config) Accelerator() accelerator.Kind {
	if c == nil {
		return accelerator.CPU
	}
	return c.Build.AcceleratorKind()
}

// AcceleratorBackend returns the backend for the accelerator the image targets.
func (c *Config) AcceleratorBackend() (accelerator.Backend, error) {
	return accelerator.For(c.Accelerator())
}

// AcceleratorBaseImage returns the accelerator vendor's base image (for
// example nvidia/cuda or rocm/dev-ubuntu), or "" when the accelerator has
// none and the generic Python image should be used.
func (c *Config) AcceleratorBaseImage() (string, error) {
	tc, err := c.toolchain()
	if err != nil {
		return "", err
	}
	return tc.baseImage(c)
}

// AcceleratorVersion returns the resolved toolkit version (the CUDA version
// for NVIDIA, the ROCm version for AMD), or "" for CPU.
func (c *Config) AcceleratorVersion() string {
	tc, err := c.toolchain()
	if err != nil {
		return ""
	}
	if field := tc.versionField(c.Build); field != nil {
		return *field
	}
	return ""
}

// acceleratorFromFile resolves the accelerator declared by cog.yaml.
func acceleratorFromFile(b *buildFile) (accelerator.Kind, error) {
	if b == nil {
		return accelerator.CPU, nil
	}
	if b.Accelerator == nil || strings.TrimSpace(*b.Accelerator) == "" {
		if b.GetGPU() {
			return accelerator.NVIDIA, nil
		}
		return accelerator.CPU, nil
	}

	raw := *b.Accelerator
	if strings.EqualFold(strings.TrimSpace(raw), string(accelerator.Auto)) {
		return "", &ValidationError{
			Field:   "build.accelerator",
			Value:   raw,
			Message: "\"auto\" depends on the build machine's hardware, so it is only accepted on the command line (cog build --accelerator auto); set cpu, nvidia or amd in cog.yaml",
		}
	}
	k, err := accelerator.Parse(raw, false)
	if err != nil {
		return "", &ValidationError{Field: "build.accelerator", Value: raw, Message: err.Error()}
	}
	if b.GPU != nil && *b.GPU != k.IsGPU() {
		return "", &ValidationError{
			Field:   "build.accelerator",
			Value:   raw,
			Message: fmt.Sprintf("conflicts with gpu: %t; remove gpu or make it consistent (accelerator %s implies gpu: %t)", *b.GPU, k, k.IsGPU()),
		}
	}
	return k, nil
}

// acceleratorDeclared reports whether cog.yaml chooses an accelerator, as
// opposed to leaving it at the default.
func acceleratorDeclared(cfg *configFile) bool {
	return cfg != nil && cfg.Build != nil && (cfg.Build.GPU != nil || cfg.Build.Accelerator != nil)
}

// validateToolkitFields rejects toolkit settings that belong to a different
// accelerator than the one being built, e.g. build.cuda with accelerator amd.
func validateToolkitFields(k accelerator.Kind, b *Build) error {
	for _, f := range toolkitFields {
		value := *f.get(b)
		if value == "" || f.owner == k || (k == accelerator.CPU && f.legacyCPU) {
			continue
		}
		return &ValidationError{
			Field:   "build." + f.name,
			Value:   value,
			Message: fmt.Sprintf("only applies to accelerator %s, but this build targets %s", f.owner, k),
		}
	}
	return nil
}

// applyAcceleratorSelection applies a command-line accelerator selection on
// top of the accelerator declared in cog.yaml. The precedence is:
//
//  1. an explicit --accelerator (cpu, nvidia, amd)
//  2. the accelerator declared in cog.yaml (build.accelerator or build.gpu)
//  3. hardware detection on the build machine, only for --accelerator auto
//  4. CPU
func applyAcceleratorSelection(c *Config, declared bool, sel AcceleratorSelection, probe accelerator.HostProbe) error {
	if sel.IsZero() {
		return nil
	}
	current := c.Accelerator()
	target := current

	if sel.Accelerator != "" {
		requested, err := accelerator.Parse(sel.Accelerator, true)
		if err != nil {
			return fmt.Errorf("invalid --accelerator: %w", err)
		}
		switch {
		case requested != accelerator.Auto:
			target = requested
		case declared:
			console.Infof("--accelerator auto: using accelerator %s declared in cog.yaml", target)
		default:
			detected, err := accelerator.Detect(probe)
			if err != nil {
				return err
			}
			target = detected
			console.Infof("--accelerator auto: detected %s on this machine. The image targets this machine's hardware; pass --accelerator explicitly for reproducible builds.", target)
		}
	}

	if target != current {
		// Retargeting replaces cog.yaml's choice, so its toolkit pins for
		// other accelerators no longer apply.
		for _, f := range toolkitFields {
			field := f.get(c.Build)
			if f.owner != target && *field != "" {
				console.Warnf("Ignoring build.%s: %s from cog.yaml because the build targets accelerator %s", f.name, *field, target)
				*field = ""
			}
		}
		c.Build.setAccelerator(target)
	}

	if sel.Version != "" {
		tc, err := c.toolchain()
		if err != nil {
			return err
		}
		field := tc.versionField(c.Build)
		if field == nil {
			return fmt.Errorf("--accelerator-version requires a GPU accelerator (nvidia or amd), but this build targets %s", target)
		}
		*field = strings.TrimSpace(sel.Version)
	}
	return nil
}
