// Shared helpers for the kernels: argument validation and portable SIMD hints.
#pragma once

#include <cstdint>
#include <limits>

#include "inferlab_kernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace il {

// Upper bound for any single buffer we allocate (elements). Prevents a
// caller from asking for terabytes through an integer overflow.
constexpr int64_t kMaxElems = int64_t(1) << 31;

inline bool positive(int64_t v) { return v > 0; }

// a * b with overflow detection; returns false on overflow or non-positive.
inline bool mul_ok(int64_t a, int64_t b, int64_t* out) {
    if (a <= 0 || b <= 0) return false;
    if (a > std::numeric_limits<int64_t>::max() / b) return false;
    *out = a * b;
    return true;
}

inline bool mul3_ok(int64_t a, int64_t b, int64_t c, int64_t* out) {
    int64_t t = 0;
    return mul_ok(a, b, &t) && mul_ok(t, c, out);
}

inline bool size_ok(int64_t elems) { return elems > 0 && elems <= kMaxElems; }

}  // namespace il

#define IL_CHECK_PTR(p) \
    do {                \
        if (!(p)) return IL_ERR_NULL; \
    } while (0)

#define IL_CHECK_POS(v) \
    do {                \
        if (!il::positive(v)) return IL_ERR_SIZE; \
    } while (0)

// Ask the compiler to vectorise a loop (OpenMP 4 simd or GCC/Clang ivdep).
#define IL_PRAGMA(x) _Pragma(#x)
#if defined(_OPENMP)
#define IL_SIMD _Pragma("omp simd")
#define IL_SIMD_REDUCE(var) IL_PRAGMA(omp simd reduction(+ : var))
#else
#define IL_SIMD
#define IL_SIMD_REDUCE(var)
#endif
