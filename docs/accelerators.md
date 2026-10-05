# GPU accelerators

Cog can build images for three targets:

| Accelerator | Toolkit | Base image                                         | Runtime device access                 |
| ----------- | ------- | -------------------------------------------------- | ------------------------------------- |
| `cpu`       | none    | `python:<version>-slim`                            | none                                  |
| `nvidia`    | CUDA    | `nvidia/cuda` or [Cog base images](base-images.md) | `--gpus` (NVIDIA Container Toolkit)   |
| `amd`       | ROCm    | `rocm/dev-ubuntu-22.04`                            | `/dev/kfd` and `/dev/dri` passthrough |

Cog packages and configures the accelerator's userspace environment. Your model code doesn't change: PyTorch runs on CUDA in an NVIDIA image and on ROCm/HIP in an AMD image, through the same `torch.cuda` APIs.

## Choosing an accelerator

Set the accelerator in [`cog.yaml`](yaml.md#accelerator):

```yaml
build:
  accelerator: amd
  python_version: "3.12"
  python_packages:
    - torch==2.5.1
```

Or choose it when you build, which overrides `cog.yaml`:

```console
cog build --accelerator amd
cog build --accelerator nvidia
cog build --accelerator cpu
```

`--accelerator-version` sets the toolkit version for the chosen accelerator: the CUDA version for `nvidia` and the ROCm version for `amd`. It overrides [`build.cuda`](yaml.md#cuda) or [`build.rocm`](yaml.md#rocm):

```console
cog build --accelerator amd --accelerator-version 6.2
cog build --accelerator nvidia --accelerator-version 12.4
```

Both flags work with every command that builds an image: `cog build`, `cog push`, `cog predict`, `cog serve`, `cog train` and `cog exec`.

Cog resolves the accelerator in this order:

1. `--accelerator cpu`, `--accelerator nvidia` or `--accelerator amd`
2. `build.accelerator` in `cog.yaml`, otherwise `build.gpu` (`gpu: true` means `nvidia`)
3. With `--accelerator auto` only: the GPU detected on the build machine
4. `cpu`

When `--accelerator` changes the accelerator from the one in `cog.yaml`, Cog ignores toolkit settings that belong to another accelerator (for example `build.cuda` when building for `amd`) and prints a warning.

### The build machine doesn't need the target GPU

An explicit accelerator never looks at the build machine's hardware. You can build an AMD image on a laptop without a GPU, on an NVIDIA workstation, or in CI. The build only needs Docker. The GPU is only needed when the image runs.

### `--accelerator auto`

`--accelerator auto` uses the accelerator declared in `cog.yaml`. If `cog.yaml` declares neither `accelerator` nor `gpu`, Cog looks at the build machine:

- `nvidia` if it finds the NVIDIA driver (`/proc/driver/nvidia/version` or `nvidia-smi`)
- `amd` if it finds `/dev/kfd`
- `cpu` if it finds neither
- an error if it finds both, so you can choose explicitly

The result depends on the machine you build on, so the same project can produce different images on different machines. Cog prints the detected accelerator. Use an explicit accelerator for reproducible builds. For the same reason, `auto` isn't accepted in `cog.yaml`, and it isn't the default: without `--accelerator`, Cog builds exactly what `cog.yaml` says.

## Existing projects and `gpu: true`

Existing projects behave exactly as before:

- `gpu: true` means an NVIDIA CUDA image. Cog never treats it as AMD or auto-detects anything for it.
- No `gpu` (or `gpu: false`) means a CPU image.
- Images built for CPU or NVIDIA have the same `run.cog.config` label as before. Only AMD images add `"accelerator": "amd"` and `"rocm"` to it.

`accelerator: nvidia` is equivalent to `gpu: true`, and `accelerator: cpu` is equivalent to `gpu: false`. You can set both `gpu` and `accelerator` as long as they agree: `gpu: false` with `accelerator: amd` is an error.

## NVIDIA (CUDA)

NVIDIA support is unchanged. Cog picks CUDA and cuDNN versions that match your PyTorch or TensorFlow version, installs the matching CUDA builds of `torch` and `torchvision`, and uses a [Cog base image](base-images.md) or an `nvidia/cuda` base image.

When you run the image with `cog predict`, `cog serve`, `cog train` or `cog exec`, Cog requests all GPUs, like `docker run --gpus all`. This needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. If the host has no NVIDIA driver, Cog retries without a GPU.

## AMD (ROCm)

### What's in the image

An AMD image contains the ROCm **userspace**. It doesn't contain the kernel driver, which must come from the host:

| Where | What                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------- |
| Host  | Linux kernel, AMD GPU, `amdgpu` kernel driver, `/dev/kfd`, `/dev/dri`                          |
| Image | ROCm userspace from `rocm/dev-ubuntu-22.04:<rocm>`, Python, ROCm builds of PyTorch, your model |

Cog builds AMD images like this:

- The base image is AMD's `rocm/dev-ubuntu-22.04:<rocm>`, which provides the HIP runtime and tools such as `rocminfo` and `rocm-smi`. Cog installs the requested Python version with uv.
- `torch`, `torchvision` and `torchaudio` come from the PyTorch ROCm package index, `https://download.pytorch.org/whl/rocm<rocm>/`. These wheels bundle the ROCm libraries they were built against.
- `/opt/rocm/bin` is on `PATH`. No NVIDIA environment variables are set.

### ROCm versions

| ROCm  | PyTorch         |
| ----- | --------------- |
| 6.0   | 2.3.0 – 2.4.1   |
| 6.1   | 2.4.0 – 2.6.0   |
| 6.2   | 2.5.0, 2.5.1    |
| 6.2.4 | 2.6.0 – 2.7.1   |
| 6.3   | 2.7.0 – 2.9.1   |
| 6.4   | 2.8.0 – 2.9.1   |
| 7.0   | 2.10.0          |
| 7.1   | 2.10.0 – 2.13.0 |
| 7.2   | 2.11.0 – 2.14.1 |

Cog chooses the ROCm version like this:

- If you set `build.rocm` or `--accelerator-version`, Cog uses that version. The build fails if it isn't in the table, or if the pinned `torch` or `torchvision` has no build for it. The error lists the ROCm versions that do have it.
- Otherwise, Cog picks the newest ROCm version that has the pinned `torch` and `torchvision`.
- If neither is pinned, Cog uses the newest ROCm version.

Pin `torch` to an exact version (`torch==2.5.1`). If it isn't pinned, pip may install the newest CUDA build from PyPI instead of the ROCm build, and Cog prints a warning. Requirements that set their own `--extra-index-url` are left as they are.

### Building

```console
cog build -t my-model --accelerator amd
```

This works on any machine with Docker; it doesn't need an AMD GPU.

`--use-cuda-base-image=false` also applies to AMD: Cog uses `python:<version>-slim` instead of the ROCm base image and relies on the libraries bundled in the PyTorch ROCm wheels. The image is smaller, but tools such as `rocminfo` aren't available.

If you compile GPU extensions during the build (for example in [`run`](yaml.md#run) commands), the build machine has no GPU for them to detect. Set the target architecture with `environment` in `cog.yaml`, which also applies during the build:

```yaml
environment:
  - PYTORCH_ROCM_ARCH=gfx1100
```

### Host requirements

To run an AMD image, the host needs:

- Linux on x86_64
- an AMD GPU [supported by ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- the `amdgpu` kernel driver loaded, which provides `/dev/kfd` and `/dev/dri/renderD*`. Install it with [AMD's driver instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/). You don't need ROCm itself on the host.
- Docker

You don't need the AMD Container Toolkit.

### Running

`cog predict`, `cog serve`, `cog train` and `cog exec` pass the GPU devices to the container automatically. They add `/dev/kfd` and `/dev/dri`, and add the container to the groups that own those devices on the host.

`--gpus` selects GPUs the same way it does for NVIDIA. Because every AMD GPU shares `/dev/kfd`, Cog applies the selection with `ROCR_VISIBLE_DEVICES`:

```console
cog predict -i prompt="hello"                  # all GPUs
cog predict --gpus device=0,1 -i prompt="hello" # GPUs 0 and 1
cog predict --gpus 1 -i prompt="hello"          # the first GPU
```

If the host doesn't have `/dev/kfd` or any render nodes, Cog stops with an error that explains what's missing. Unlike NVIDIA images, it doesn't fall back to running on the CPU.

To run an AMD image with Docker directly:

```console
docker run -d -p 5001:5000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" \
  my-model
```

Some consumer GPUs that ROCm doesn't officially support work if you set `HSA_OVERRIDE_GFX_VERSION`, for example `-e HSA_OVERRIDE_GFX_VERSION=10.3.0` with `cog predict` or `docker run`.

### Limitations

- There are no [Cog base images](base-images.md) for AMD, so builds take longer and models start more slowly than NVIDIA models that use them. `--use-cog-base-image=true` with `--accelerator amd` is an error.
- Cog doesn't map TensorFlow to a ROCm build. `tensorflow` installs the CPU build; use `tensorflow-rocm` yourself.
- AMD images are `linux/amd64` only.
- `cog predict` and the other run commands look for the AMD devices on the machine running Cog. A remote Docker daemon (`DOCKER_HOST`) needs the same devices.
- The ROCm and PyTorch version table is built into Cog. Newer ROCm releases need a newer Cog, or you can set the package index yourself with `--extra-index-url`.
- `cog doctor`'s GPU compatibility check only covers NVIDIA.
- Tools and platforms that only read `"gpu": true` from the image's `run.cog.config` label can't tell an AMD image from an NVIDIA image. Check the `accelerator` field.
