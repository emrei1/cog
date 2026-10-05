package cli

import (
	"github.com/spf13/cobra"

	"github.com/replicate/cog/pkg/config"
	"github.com/replicate/cog/pkg/model"
)

const (
	acceleratorFlagKey        = "accelerator"
	acceleratorVersionFlagKey = "accelerator-version"
)

var (
	buildAccelerator        string
	buildAcceleratorVersion string
)

func addAcceleratorFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&buildAccelerator, acceleratorFlagKey, "", "Accelerator to build the image for: 'cpu', 'nvidia', 'amd', or 'auto' (use cog.yaml's accelerator, or detect this machine's GPU if cog.yaml doesn't set one). Overrides build.gpu and build.accelerator in cog.yaml")
	cmd.Flags().StringVar(&buildAcceleratorVersion, acceleratorVersionFlagKey, "", "Accelerator toolkit version: the CUDA version for nvidia, the ROCm version for amd. Overrides build.cuda or build.rocm in cog.yaml")
}

// acceleratorSelectionFromFlags returns the accelerator chosen on the
// command line, or a zero selection for commands without the flags.
func acceleratorSelectionFromFlags(cmd *cobra.Command) config.AcceleratorSelection {
	if cmd.Flags().Lookup(acceleratorFlagKey) == nil {
		return config.AcceleratorSelection{}
	}
	return config.AcceleratorSelection{
		Accelerator: buildAccelerator,
		Version:     buildAcceleratorVersion,
	}
}

// loadSource loads cog.yaml with the command's accelerator flags applied.
func loadSource(cmd *cobra.Command) (*model.Source, error) {
	return model.NewSource(configFilename, config.WithAcceleratorSelection(acceleratorSelectionFromFlags(cmd)))
}
