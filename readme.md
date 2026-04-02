# CUDA Multiplication Toolkit

CUDA Multiplication Toolkit is a compact CUDA/C/Python codebase for benchmarking dense matrix multiplication across multiple execution paths and exposing selected GPU kernels to Python through a lightweight native interface.

The repository combines low-level CUDA implementations, cuBLAS-based acceleration, benchmark automation, and Python interoperability in a single workspace. In addition to matrix multiplication, it also includes CPU and GPU image convolution routines exported from the same shared library.

## Highlights

- Multiple matrix multiplication backends:
  - CPU baseline in C
  - Naive CUDA kernel
  - Shared-memory tiled CUDA kernel
  - cuBLAS SGEMM comparison path
- Benchmark scripts for repeatable runs across common matrix sizes
- Native CUDA library exported for Python `ctypes` integration
- Python examples for GPU matrix multiplication and CPU/GPU image convolution
- Sample benchmark outputs and generated convolution images included in-repo

## Repository Layout

```text
CUDA-Multiplication-Toolkit/
├── lib/                    # Export metadata / runtime library artifacts
├── python/
│   ├── conv_outputs/       # Generated convolution output images
│   ├── images/             # Input images for Python convolution tests
│   ├── test_conv_ctypes.py
│   └── test_matmul_ctypes.py
├── results/                # Benchmark logs
├── scripts/                # PowerShell benchmark runners
├── src/                    # CPU and CUDA source files
└── README.md
```

## Core Components

### Standalone Benchmarks

- `src/matmul_cpu.c`
  CPU reference implementation using a straightforward triple-loop matrix multiply.
- `src/matrix_gpu.cu`
  Naive CUDA implementation for baseline GPU timing.
- `src/matrix_gpu_with_optimize.cu`
  Naive and tiled CUDA kernels in one executable, including speedup reporting.
- `src/matrix_gpu_opt_cublas.cu`
  End-to-end comparison of naive CUDA, tiled CUDA, and cuBLAS SGEMM.

### Shared Library for Python

`src/matrix_lib.cu` exports:

- `gpu_matrix_multiply(float* A, float* B, float* C, int N)`
- `cpu_convolve_u32(const uint32_t* img, int M, const int32_t* filt, int N, uint32_t* out)`
- `gpu_convolve_u32(const uint32_t* img, int M, const int32_t* filt, int N, uint32_t* out)`

The Python bindings are intentionally minimal and use `ctypes` with contiguous NumPy buffers instead of a heavier wrapper layer.

## Platform Assumptions

The current repository is oriented toward a Windows CUDA environment:

- PowerShell scripts under `scripts/`
- DLL exports via `__declspec(dllexport)`
- Python examples loading `lib/matrix.dll`

Expected dependencies:

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- Python 3
- `numpy`
- `Pillow`

cuBLAS is required for `src/matrix_gpu_opt_cublas.cu`.

## Build and Run

### Benchmark the CPU baseline

`scripts/run_cpu.ps1` expects `bin/matmul_cpu.exe` to already exist.

Example manual build:

```powershell
gcc src/matmul_cpu.c -O3 -o bin/matmul_cpu.exe
```

```powershell
cd scripts
./run_cpu.ps1
```

Results are written to `results/cpu_results.txt`.

### Benchmark the optimized CUDA implementation

`scripts/run_gpu_opt.ps1` compiles and runs the tiled benchmark executable.

```powershell
cd scripts
./run_gpu_opt.ps1
```

### Benchmark the cuBLAS comparison build

```powershell
cd scripts
./run_gpu_opt_cublas.ps1
```

This script links against:

```text
-lcublas -lcublasLt
```

### Build the naive CUDA executable

If you want to run `scripts/run_gpu.ps1`, build the executable first:

```powershell
nvcc -O3 src/matrix_gpu.cu -o src/matrix_gpu.exe
```

### Build the Python-facing DLL

The Python examples load `lib/matrix.dll`, so the shared library must be built to that location or copied there after compilation.

Example:

```powershell
nvcc -O3 -shared -o lib/matrix.dll src/matrix_lib.cu
```

### Run the Python examples

```powershell
python python/test_matmul_ctypes.py
python python/test_conv_ctypes.py
```

Both scripts perform basic correctness checks before reporting timings.
`python/test_conv_ctypes.py` reads images from `python/images/` and writes processed outputs to `python/conv_outputs/`.

## Implementation Notes

- The optimized matrix multiplication kernels use a tile size of `16x16`.
- Boundary checks are implemented for non-multiple matrix sizes.
- The convolution kernels use zero padding at the image borders.
- Convolution output is clamped to the `[0, 255]` range for grayscale image export.

## Included Artifacts

- `results/` contains sample benchmark logs from previous runs
- `python/conv_outputs/` contains saved CPU and GPU convolution outputs

These files make the repository useful both as runnable code and as a record of prior benchmark experiments.

## Roadmap

- Add a unified build flow instead of split manual/script-driven steps
- Provide Linux shared-library support in addition to the current Windows DLL path
- Add correctness validation and numerical comparison utilities
- Extend benchmark reporting with throughput metrics such as GFLOPS

## Technology Stack

- C
- CUDA C++
- NVIDIA cuBLAS
- Python
- `ctypes`
- NumPy
- Pillow
