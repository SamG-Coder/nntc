// cpu/cpu_sample.h: sample.cuh, COPIED - the sampling rule, the feature order and the site set.
//
// sample.cuh opens with #include <cuda_runtime.h> and is one of the nine files that are not edited
// (docs/CPU_BACKEND_PLAN.md decision 1), so the CPU backend carries its own copy, function for function and in the
// same order. Read sample.cuh for the WHY of every rule below; the comments here say only what differs, and two things
// do:
//
//   * NNTC_HOST_DEVICE / __host__ __device__ are gone, and <cuda_runtime.h> is <cmath>;
//   * floorf is spelled std::floor on the float argument (plan section 1.8). sample.cuh calls floorf unqualified.
//     <cmath> guarantees std::floor (and its float overload) but does NOT guarantee ::floorf at global scope, which is
//     exactly the kind of thing that compiles on x86 glibc and MSVC and then does not on another Arm toolchain - the
//     platform this backend exists for. std::floor(float) IS floorf, so no result moves.
//
// The copy is guarded by the per-kernel harness (src/backend_check.cpp): every CPU kernel that samples through this
// file is compared against the CUDA kernel that samples through the original, on identical inputs.

#pragma once

#include <cmath>
#include <cstddef>

namespace nntc_cpu
{

struct BilinearTap
{
    int x0, x1, y0, y1;
    float fx, fy;
};

inline int clamp_index(int i, int n)
{
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// The hardware's bilinear addressing with clamp-to-edge, for a site given as a PIXEL of the plane being fitted.
inline BilinearTap bilinear_tap_pixel(int px, int py, float dx, float dy, int w0, int h0, int w, int h)
{
    const float x = w == w0 ? (float)px + dx : (((float)px + 0.5f + dx) * (float)w) / (float)w0 - 0.5f;
    const float y = h == h0 ? (float)py + dy : (((float)py + 0.5f + dy) * (float)h) / (float)h0 - 0.5f;
    const int x0 = (int)std::floor(x);
    const int y0 = (int)std::floor(y);
    BilinearTap t;
    t.fx = x - (float)x0;
    t.fy = y - (float)y0;
    t.x0 = clamp_index(x0, w);
    t.x1 = clamp_index(x0 + 1, w);
    t.y0 = clamp_index(y0, h);
    t.y1 = clamp_index(y0 + 1, h);
    return t;
}

// The blend of the four clamped taps, channel by channel.
inline void sample_bilinear(const float* plane, int w, int c, const BilinearTap& t, float* out)
{
    const float* a = plane + ((size_t)t.y0 * w + t.x0) * c;
    const float* b = plane + ((size_t)t.y0 * w + t.x1) * c;
    const float* e = plane + ((size_t)t.y1 * w + t.x0) * c;
    const float* f = plane + ((size_t)t.y1 * w + t.x1) * c;
    const float w00 = (1.0f - t.fx) * (1.0f - t.fy), w10 = t.fx * (1.0f - t.fy);
    const float w01 = (1.0f - t.fx) * t.fy, w11 = t.fx * t.fy;
    for (int k = 0; k < c; k++)
        out[k] = w00 * a[k] + w10 * b[k] + w01 * e[k] + w11 * f[k];
}

// phi(s, c) in the decoder's order. Returns the feature count written.
inline int build_phi(const float* s, int c0, const float* c, int c1, float* phi)
{
    int k = 0;
    for (int j = 0; j < c1; j++)
        phi[k++] = c[j];
    for (int i = 0; i < c0; i++)
        phi[k++] = s[i];
    for (int i = 0; i < c0; i++)
        for (int j = 0; j < c1; j++)
            phi[k++] = s[i] * c[j];
    return k;
}

// The K fixed subtexel offsets of pixel (tx, ty), turned a quarter turn per texel of a 2x2 cell (sample.cuh says why).
inline void subtexel_offset(int k_count, int j, int tx, int ty, float& dx, float& dy)
{
    if (k_count == 2)
    {
        dx = j == 0 ? -0.25f : 0.25f;
        dy = j == 0 ? 0.25f : -0.25f;
        return;
    }
    float ox = 0.25f, oy = 0.25f;
    if (k_count == 4)
    {
        const float tx4[4] = { -0.375f, -0.125f, 0.125f, 0.375f };
        const float ty4[4] = { -0.125f, 0.375f, -0.375f, 0.125f };
        ox = tx4[j];
        oy = ty4[j];
    }
    const int n = (ty & 1) * 2 + (tx & 1);
    for (int r = 0; r < n; r++)
    {
        const float t = ox;
        ox = -oy;
        oy = t;
    }
    dx = ox;
    dy = oy;
}

// Site j of pixel (px, py): j = 0 is the centre, j = 1..K the fractional sites.
inline void site_delta(int k_count, int j, int px, int py, float& dx, float& dy)
{
    dx = 0.0f;
    dy = 0.0f;
    if (j > 0)
        subtexel_offset(k_count, j - 1, px, py, dx, dy);
}

// The decoder's input phi and the ground truth at one site of one plane, in fp32 - the solver's arithmetic.
inline void site_row(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0, int c1,
                     int nout, int k_count, int j, int px, int py, float* phi, float* target)
{
    float dx, dy;
    site_delta(k_count, j, px, py, dx, dy);
    float s[4], c[4];   // 4 is the per-level channel cap the shader and the .dds formats share
    const BilinearTap t1 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w1, h1);
    sample_bilinear(v1, w1, c1, t1, c);
    if (j == 0)
    {
        const size_t idx = (size_t)py * w0 + px;
        for (int i = 0; i < c0; i++)
            s[i] = v0[idx * c0 + i];
        for (int i = 0; i < nout; i++)
            target[i] = src[idx * nout + i];
    }
    else
    {
        const BilinearTap t0 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w0, h0);
        sample_bilinear(v0, w0, c0, t0, s);
        sample_bilinear(src, w0, nout, t0, target);
    }
    build_phi(s, c0, c, c1, phi);
}

}   // namespace nntc_cpu
