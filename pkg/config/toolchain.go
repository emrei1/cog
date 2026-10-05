package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/replicate/cog/pkg/accelerator"
	"github.com/replicate/cog/pkg/requirements"
	"github.com/replicate/cog/pkg/util/console"
)

// toolchain is the build-time part of an accelerator backend that depends on
// cog.yaml: resolving the toolkit version against the Python requirements
// and mapping framework packages to builds for that toolkit. Adding an
// accelerator means adding a toolchain here and a Backend in pkg/accelerator.
type toolchain interface {
	// complete validates and resolves the toolkit version in c.Build.
	complete(c *Config) error
	// baseImage returns the accelerator base image, or "" for the generic
	// Python image. Only valid after complete.
	baseImage(c *Config) (string, error)
	// versionField returns the Build field holding the toolkit version, or
	// nil when the accelerator has no versioned toolkit.
	versionField(b *Build) *string
	// pinnedPackage maps a pinned name==version requirement to the build
	// for this accelerator.
	pinnedPackage(c *Config, name, version, goos, goarch string) (pkgName, pkgVersion, findLinks, extraIndexURL string, err error)
	// unpinnedPackageIndexes returns extra package indexes needed by an
	// unpinned requirement line.
	unpinnedPackageIndexes(c *Config, requirement string) []string
}

var toolchains = map[accelerator.Kind]toolchain{
	accelerator.CPU:    cpuToolchain{},
	accelerator.NVIDIA: cudaToolchain{},
	accelerator.AMD:    rocmToolchain{},
}

func (c *Config) toolchain() (toolchain, error) {
	k := c.Accelerator()
	tc, ok := toolchains[k]
	if !ok {
		_, err := accelerator.For(k)
		return nil, err
	}
	return tc, nil
}

// completeAccelerator validates the accelerator settings and resolves the
// toolkit version.
func (c *Config) completeAccelerator() error {
	tc, err := c.toolchain()
	if err != nil {
		return err
	}
	if err := validateToolkitFields(c.Accelerator(), c.Build); err != nil {
		return err
	}
	return tc.complete(c)
}

// torchFamily are the PyTorch packages published on the per-toolkit
// download.pytorch.org indexes.
var torchFamily = []string{"torch", "torchvision", "torchaudio"}

func isTorchFamily(name string) bool {
	return slices.Contains(torchFamily, requirements.NormalizePackageName(name))
}

type cpuToolchain struct{}

func (cpuToolchain) complete(*Config) error                          { return nil }
func (cpuToolchain) baseImage(*Config) (string, error)               { return "", nil }
func (cpuToolchain) versionField(*Build) *string                     { return nil }
func (cpuToolchain) unpinnedPackageIndexes(*Config, string) []string { return nil }

func (cpuToolchain) pinnedPackage(_ *Config, name, version, goos, goarch string) (string, string, string, string, error) {
	switch name {
	case "torch":
		return torchCPUPackage(version, goos, goarch)
	case "torchvision":
		return torchvisionCPUPackage(version, goos, goarch)
	}
	// There is no CPU case for tensorflow because the default package is just
	// the CPU package, so no transformation of version is needed.
	return name, version, "", "", nil
}

type cudaToolchain struct{}

func (cudaToolchain) complete(c *Config) error { return c.validateAndCompleteCUDA() }

func (cudaToolchain) baseImage(c *Config) (string, error) {
	return c.CUDABaseImageTag()
}

func (cudaToolchain) versionField(b *Build) *string                   { return &b.CUDA }
func (cudaToolchain) unpinnedPackageIndexes(*Config, string) []string { return nil }

func (cudaToolchain) pinnedPackage(c *Config, name, version, _, _ string) (string, string, string, string, error) {
	switch name {
	case "tensorflow":
		name, version, err := tfGPUPackage(version, c.Build.CUDA)
		return name, version, "", "", err
	case "torch":
		return torchGPUPackage(version, c.Build.CUDA)
	case "torchvision":
		return torchvisionGPUPackage(version, c.Build.CUDA)
	}
	return name, version, "", "", nil
}

type rocmToolchain struct{}

func (rocmToolchain) versionField(b *Build) *string { return &b.ROCm }

func (rocmToolchain) baseImage(c *Config) (string, error) {
	release, err := accelerator.LookupROCm(c.Build.ROCm)
	if err != nil {
		return "", err
	}
	return release.BaseImage, nil
}

// complete resolves build.rocm. An explicit version must be supported and
// must ship the pinned torch and torchvision; otherwise Cog picks the newest
// ROCm release that ships them, or the default release when nothing is pinned.
func (rocmToolchain) complete(c *Config) error {
	pins := map[string]string{}
	for _, pkg := range []string{"torch", "torchvision"} {
		if v, ok := c.pinnedVersionOnDefaultIndex(pkg); ok {
			if strings.Contains(v, "+cu") || strings.Contains(v, "+cpu") {
				return fmt.Errorf("%s==%s is a CUDA/CPU build of PyTorch and cannot run on accelerator amd; pin %s==%s without the local version suffix and Cog will install the ROCm build", pkg, v, pkg, strings.SplitN(v, "+", 2)[0])
			}
			pins[pkg] = v
		}
	}
	for _, pkg := range []string{"torch", "torchvision"} {
		if c.hasUnpinnedPythonPackage(pkg) {
			console.Warnf("%s is not pinned to an exact version. pip picks the newest %s on any index, which may be a CUDA build from PyPI rather than the ROCm build; pin %s==<version> so Cog installs the ROCm build.", pkg, pkg, pkg)
		}
	}

	if c.Build.ROCm != "" {
		release, err := accelerator.LookupROCm(c.Build.ROCm)
		if err != nil {
			return err
		}
		c.Build.ROCm = release.Version
		for _, pkg := range []string{"torch", "torchvision"} {
			v, ok := pins[pkg]
			if !ok || release.Ships(pkg, v) {
				continue
			}
			return fmt.Errorf("%s==%s has no ROCm %s build.%s", pkg, v, release.Version, suggestROCmFor(pkg, v))
		}
		return nil
	}

	candidates := accelerator.ROCmReleases()
	matched := []string{}
	for _, pkg := range []string{"torch", "torchvision"} {
		v, ok := pins[pkg]
		if !ok {
			continue
		}
		var shipping []accelerator.ROCmRelease
		for _, r := range candidates {
			if r.Ships(pkg, v) {
				shipping = append(shipping, r)
			}
		}
		switch {
		case len(shipping) > 0:
		case len(matched) > 0 && len(accelerator.ROCmReleasesShipping(pkg, v)) > 0:
			return fmt.Errorf("no ROCm version has builds of both %s and %s==%s; pin versions from the same PyTorch release", strings.Join(matched, " and "), pkg, v)
		default:
			return fmt.Errorf("cog doesn't know of a ROCm build of %s==%s.%s", pkg, v, suggestROCmFor(pkg, v))
		}
		candidates = shipping
		matched = append(matched, pkg+"=="+v)
	}
	release := candidates[len(candidates)-1]
	if len(pins) == 0 {
		release = accelerator.DefaultROCmRelease()
	}
	c.Build.ROCm = release.Version
	console.Debugf("Setting ROCm to version %s", c.Build.ROCm)
	return nil
}

func suggestROCmFor(pkg, version string) string {
	releases := accelerator.ROCmReleasesShipping(pkg, version)
	if len(releases) == 0 {
		return fmt.Sprintf(" Supported ROCm versions are %s; pin a %s version that one of them ships, or set the package index yourself with --extra-index-url.", strings.Join(accelerator.ROCmVersions(), ", "), pkg)
	}
	versions := make([]string, len(releases))
	for i, r := range releases {
		versions[i] = r.Version
	}
	return fmt.Sprintf(" It is available for ROCm %s.", strings.Join(versions, ", "))
}

func (rocmToolchain) pinnedPackage(c *Config, name, version, _, _ string) (string, string, string, string, error) {
	switch {
	case name == "tensorflow":
		console.Warnf("Cog installs tensorflow unchanged for accelerator amd, which is the CPU build. Use the tensorflow-rocm package for GPU support.")
		return name, version, "", "", nil
	case !isTorchFamily(name):
		return name, version, "", "", nil
	}
	release, err := accelerator.LookupROCm(c.Build.ROCm)
	if err != nil {
		return "", "", "", "", err
	}
	return name, version, "", release.TorchIndexURL, nil
}

func (rocmToolchain) unpinnedPackageIndexes(c *Config, requirement string) []string {
	if !isTorchFamily(requirements.PackageName(requirement)) {
		return nil
	}
	release, err := accelerator.LookupROCm(c.Build.ROCm)
	if err != nil {
		return nil
	}
	return []string{release.TorchIndexURL}
}

// hasUnpinnedPythonPackage reports whether the requirements include name
// without pinning it to an exact version.
func (c *Config) hasUnpinnedPythonPackage(name string) bool {
	for _, pkg := range c.Build.pythonRequirementsContent {
		if isLocalPackageArtifactRequirement(pkg) || requirements.NormalizePackageName(requirements.PackageName(pkg)) != name {
			continue
		}
		if _, _, _, _, err := requirements.SplitPinnedPythonRequirement(pkg); err != nil {
			return true
		}
	}
	return false
}

// pinnedVersionOnDefaultIndex returns the version of a name==version
// requirement that does not set its own package index.
func (c *Config) pinnedVersionOnDefaultIndex(name string) (string, bool) {
	for _, pkg := range c.Build.pythonRequirementsContent {
		if isLocalPackageArtifactRequirement(pkg) {
			continue
		}
		pkgName, version, findLinks, extraIndexURLs, err := requirements.SplitPinnedPythonRequirement(pkg)
		if err != nil || requirements.NormalizePackageName(pkgName) != name {
			continue
		}
		if len(findLinks) > 0 || len(extraIndexURLs) > 0 {
			return "", false
		}
		return version, true
	}
	return "", false
}
