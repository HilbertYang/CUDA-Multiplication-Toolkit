import ctypes
import numpy as np
import time
from pathlib import Path

dll_path = Path(__file__).resolve().parent.parent / "lib" / "matrix.dll"
lib = ctypes.CDLL(str(dll_path))

lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]
lib.gpu_matrix_multiply.restype = None

N = 256
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

t0 = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
t1 = time.time()

ref = A @ B
max_abs_err = float(np.max(np.abs(C - ref)))
ok = np.allclose(C, ref, rtol=1e-4, atol=1e-3)

print(f"Python call to GPU tiled matmul completed in {t1 - t0:.4f} seconds")
print(f"Validation: {'PASS' if ok else 'FAIL'}")
print(f"Max abs error: {max_abs_err:.6f}")
print("C[0,0] =", C[0, 0])

if not ok:
    raise SystemExit("GPU matmul validation failed")
