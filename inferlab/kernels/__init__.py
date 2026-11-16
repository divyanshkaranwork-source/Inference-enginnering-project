"""Layer 3 - Kernels: C++ CPU kernels (ctypes) + NumPy reference algorithms.

GPU versions of the same kernels (CUDA C++ and Triton) live in the top-level
``gpu_kernels/`` directory for readers with an NVIDIA GPU.
"""

from inferlab.kernels import cpp, reference


def get_kernels(name: str = "numpy"):
    """Return a kernel backend for ``LlamaModel``: None (NumPy) or CppKernels."""
    if name == "numpy":
        return None
    if name == "cpp":
        if not cpp.available():
            raise RuntimeError(cpp.load_error() or "C++ kernels unavailable")
        return cpp.CppKernels()
    raise ValueError("kernels must be 'numpy' or 'cpp'")


__all__ = ["cpp", "get_kernels", "reference"]
