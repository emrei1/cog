package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/replicate/cog/pkg/accelerator"
	"github.com/replicate/cog/pkg/config"
)

func resetAcceleratorFlags(t *testing.T) {
	t.Cleanup(func() {
		buildAccelerator = ""
		buildAcceleratorVersion = ""
	})
}

func TestAcceleratorFlagsOnBuildingCommands(t *testing.T) {
	commands := map[string]*cobra.Command{
		"build":   newBuildCommand(),
		"push":    newPushCommand(),
		"predict": newPredictCommand(),
		"serve":   newServeCommand(),
		"train":   newTrainCommand(),
		"exec":    newExecCommand(),
	}
	for name, cmd := range commands {
		t.Run(name, func(t *testing.T) {
			for _, flag := range []string{acceleratorFlagKey, acceleratorVersionFlagKey} {
				f := cmd.Flags().Lookup(flag)
				require.NotNil(t, f, "--%s missing", flag)
				assert.Equal(t, "", f.DefValue, "--%s must default to cog.yaml's setting", flag)
			}
		})
	}
}

func TestAcceleratorSelectionFromFlags(t *testing.T) {
	resetAcceleratorFlags(t)
	cmd := newBuildCommand()
	require.NoError(t, cmd.ParseFlags([]string{"--accelerator", "amd", "--accelerator-version", "6.2"}))
	assert.Equal(t, config.AcceleratorSelection{Accelerator: "amd", Version: "6.2"}, acceleratorSelectionFromFlags(cmd))
}

func TestAcceleratorSelectionDefaultsToConfig(t *testing.T) {
	resetAcceleratorFlags(t)
	cmd := newBuildCommand()
	require.NoError(t, cmd.ParseFlags([]string{}))
	assert.True(t, acceleratorSelectionFromFlags(cmd).IsZero())

	// Commands without the flags never override cog.yaml.
	assert.True(t, acceleratorSelectionFromFlags(&cobra.Command{}).IsZero())
}

func TestLoadSourceAppliesAcceleratorFlag(t *testing.T) {
	resetAcceleratorFlags(t)
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "cog.yaml"), []byte("build:\n  gpu: true\n  python_version: \"3.12\"\n"), 0o644))
	t.Chdir(dir)

	cmd := newBuildCommand()
	require.NoError(t, cmd.ParseFlags([]string{}))
	src, err := loadSource(cmd)
	require.NoError(t, err)
	defer src.Close()
	assert.Equal(t, accelerator.NVIDIA, src.Config.Accelerator(), "gpu: true without flags stays NVIDIA")

	cmd = newBuildCommand()
	require.NoError(t, cmd.ParseFlags([]string{"--accelerator", "amd", "--accelerator-version", "6.3"}))
	src, err = loadSource(cmd)
	require.NoError(t, err)
	defer src.Close()
	assert.Equal(t, accelerator.AMD, src.Config.Accelerator())
	assert.Equal(t, "6.3", src.Config.Build.ROCm)

	cmd = newBuildCommand()
	require.NoError(t, cmd.ParseFlags([]string{"--accelerator", "quantum"}))
	_, err = loadSource(cmd)
	require.ErrorContains(t, err, `invalid --accelerator: unsupported accelerator "quantum"`)
}
