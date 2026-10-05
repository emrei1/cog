package accelerator

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/replicate/cog/pkg/util/version"
)

// MinimumROCmVersion is the oldest ROCm release Cog builds images for.
const MinimumROCmVersion = "6.0"

// ROCmRelease is a ROCm userspace release Cog can build images for: the AMD
// base image that provides the ROCm runtime, and the PyTorch wheel index
// built against it.
type ROCmRelease struct {
	Version       string   `json:"version"`
	BaseImage     string   `json:"base_image"`
	TorchIndexURL string   `json:"torch_index_url"`
	Torch         []string `json:"torch"`
	Torchvision   []string `json:"torchvision"`
}

// Ships reports whether the release's PyTorch index has pkg at ver.
// ver may be a prefix such as "2.5", matching any 2.5.x build.
func (r ROCmRelease) Ships(pkg, ver string) bool {
	var versions []string
	switch pkg {
	case "torch":
		versions = r.Torch
	case "torchvision":
		versions = r.Torchvision
	default:
		return false
	}
	return slices.ContainsFunc(versions, func(v string) bool { return version.Matches(v, ver) })
}

// rocmReleasesData is maintained by hand from the pytorch.org ROCm wheel
// indexes and the rocm/dev-ubuntu-22.04 tags on Docker Hub.
//
//go:embed rocm_releases.json
var rocmReleasesData []byte

var rocmReleases = mustLoadROCmReleases(rocmReleasesData)

func mustLoadROCmReleases(data []byte) []ROCmRelease {
	var releases []ROCmRelease
	if err := json.Unmarshal(data, &releases); err != nil {
		panic(fmt.Sprintf("failed to load embedded ROCm releases: %s", err))
	}
	slices.SortFunc(releases, func(a, b ROCmRelease) int {
		switch {
		case version.Equal(a.Version, b.Version):
			return 0
		case version.Greater(a.Version, b.Version):
			return 1
		default:
			return -1
		}
	})
	return releases
}

// ROCmReleases returns the supported ROCm releases, oldest first.
func ROCmReleases() []ROCmRelease {
	return slices.Clone(rocmReleases)
}

// ROCmVersions returns the supported ROCm version strings, oldest first.
func ROCmVersions() []string {
	versions := make([]string, len(rocmReleases))
	for i, r := range rocmReleases {
		versions[i] = r.Version
	}
	return versions
}

// DefaultROCmRelease is the release used when nothing constrains the choice.
func DefaultROCmRelease() ROCmRelease {
	return rocmReleases[len(rocmReleases)-1]
}

// LookupROCm returns the release for a ROCm version such as "6.2" or
// "6.2.4". A trailing ".0" patch is ignored, so "6.2.0" selects "6.2".
func LookupROCm(v string) (ROCmRelease, error) {
	v = strings.TrimSpace(v)
	candidates := []string{v}
	if trimmed, ok := strings.CutSuffix(v, ".0"); ok && strings.Count(v, ".") == 2 {
		candidates = append(candidates, trimmed)
	}
	for _, c := range candidates {
		for _, r := range rocmReleases {
			if r.Version == c {
				return r, nil
			}
		}
	}
	if parsed, err := version.NewVersion(v); err == nil && !parsed.GreaterOrEqual(version.MustVersion(MinimumROCmVersion)) {
		return ROCmRelease{}, fmt.Errorf("minimum supported ROCm version is %s, requested %q", MinimumROCmVersion, v)
	}
	return ROCmRelease{}, fmt.Errorf("unsupported ROCm version %q: supported versions are %s", v, strings.Join(ROCmVersions(), ", "))
}

// ROCmReleasesShipping returns the releases whose PyTorch index has pkg at
// ver, oldest first.
func ROCmReleasesShipping(pkg, ver string) []ROCmRelease {
	var out []ROCmRelease
	for _, r := range rocmReleases {
		if r.Ships(pkg, ver) {
			out = append(out, r)
		}
	}
	return out
}
