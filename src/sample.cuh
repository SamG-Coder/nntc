// sample.cuh: the sampling rule, written once for the host and the device.
//
// BOTH latent textures are read through the hardware filter, at the same position: x = u * w - 0.5, x0 = floor(x),
// fx = x - x0, the two texel indices x0 and x0 + 1 CLAMPED into [0, w-1] (so a position past the last texel centre has
// both taps on the same texel and their weights add), likewise in y, and the sample is the blend of the four taps.
//
// The position is carried as the PIXEL of the plane being fitted, (px + 0.5 + dx), and scaled into the plane being read
// as (px + 0.5 + dx) * w / w0 - 0.5, rather than through a normalised u: see bilinear_tap_pixel for what the round trip
// through u costs.
//
// A pixel p = (px, py) of a plane of width w and height h has its centre at u = (px + 0.5) / w, v = (py + 0.5) / h. At
// that position level 0's blend collapses onto the single texel p, because level 0 is at the decoded extent and the
// fractional parts are zero; the CENTRE sites below therefore read it as that texel directly, which is the same number
// the filter returns and is cheaper to differentiate. Level 1, at a quarter of the extent, blends four texels there.
// Between centres both are genuinely filtered, and that is what the fractional sites are for.
//
// phi is built in the order the shader builds it: [a] c_j, [b] s_i, [sc] s_i c_j with i outer and j inner.

#pragma once

#include <cuda_runtime.h>

struct BilinearTap
{
    int x0, x1, y0, y1;
    float fx, fy;
};

__host__ __device__ inline int clamp_index(int i, int n)
{
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// The hardware's bilinear addressing with clamp-to-edge, for a site given as a PIXEL of the plane being fitted rather
// than as a normalised coordinate.
//
// A site is always (px + 0.5 + dx) of a w0 x h0 plane, and the position of that site in a w x h plane is that pixel
// coordinate scaled by w / w0. Going through u = (px + 0.5 + dx) / w0 and then x = u * w - 0.5 rounds twice, and the
// first rounding is relative to u, so the error in x grows with the plane: about 1e-4 of a texel at w0 = 2888, which is
// a thousandth of a subtexel offset and rises with the image. Scaling the exact pixel coordinate instead leaves the
// rounding proportional to x itself - and at w == w0, which is how level 0 and the source are read, the ratio is
// exactly 1 and the addressing is exact.
__host__ __device__ inline BilinearTap bilinear_tap_pixel(int px, int py, float dx, float dy, int w0, int h0, int w,
                                                          int h)
{
    const float x = w == w0 ? (float)px + dx : (((float)px + 0.5f + dx) * (float)w) / (float)w0 - 0.5f;
    const float y = h == h0 ? (float)py + dy : (((float)py + 0.5f + dy) * (float)h) / (float)h0 - 0.5f;
    const int x0 = (int)floorf(x);
    const int y0 = (int)floorf(y);
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
__host__ __device__ inline void sample_bilinear(const float* plane, int w, int c, const BilinearTap& t, float* out)
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
__host__ __device__ inline int build_phi(const float* s, int c0, const float* c, int c1, float* phi)
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

// ---------------------------------------------------------------------------------------------------------------
// The site set
// ---------------------------------------------------------------------------------------------------------------
//
// Every pixel of every plane carries one CENTRE site and K FRACTIONAL sites. The centre site is the pixel itself, read
// the way the shader reads a pixel: at a pixel centre level 0's filtered sample IS that texel, and level 1 blends
// four. A fractional
// site is a position between texel centres, where both levels are read bilinearly and the ground truth is the source of
// that plane read bilinearly at the same position with the same clamp. Fitting those positions is what makes the
// representation valid under the hardware's sampling operator rather than only at the texel grid.
//
// The K offsets are fixed - there is no seed and no hash anywhere in the encoder. The base pattern is the standard
// rotated grid of four positions, in units of one pixel of the plane being fitted:
//
//     P = { (-3/8, -1/8), (-1/8, +3/8), (+1/8, -3/8), (+3/8, +1/8) }
//
// - eight numbers, reproducible by hand. So that neighbouring texels do not all carry the same sub-pixel bias (a single
// systematic shift could otherwise be fitted away instead of being resolved), the pattern is turned a quarter turn per
// texel: (dx, dy) -> (-dy, dx) applied n = (ty & 1) * 2 + (tx & 1) times, so the four texels of a 2x2 cell carry the
// four rotations. P is itself invariant under that quarter turn, so at K = 4 the rotation only permutes a texel's four
// sites - the four rotations are already present inside one texel, which is the property the phase exists to provide.
// It bites at K = 1, where the single offset (+1/4, +1/4) visits a different quadrant in each texel of a 2x2 cell.
//
// K = 2 is the diagonal pair (-1/4, +1/4), (+1/4, -1/4), unrotated: a quarter turn maps that pair off itself, and the
// two positions are already opposite. K = 0 is centres only.
__host__ __device__ inline void subtexel_offset(int k_count, int j, int tx, int ty, float& dx, float& dy)
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

// Site j of pixel (px, py): j = 0 is the centre, j = 1..K the fractional sites, and the answer is the offset from the
// pixel centre in units of one pixel of the plane being fitted. Both levels and the source are addressed from this one
// position, so an offset of a plane pixel is a quarter of a level-1 texel - which is exactly the relation the hardware
// has between the two samplers.
__host__ __device__ inline void site_delta(int k_count, int j, int px, int py, float& dx, float& dy)
{
    dx = 0.0f;
    dy = 0.0f;
    if (j > 0)
        subtexel_offset(k_count, j - 1, px, py, dx, dy);
}

// The decoder's input phi and the ground truth at one site of one plane, in fp32 - the solver's arithmetic.
// At the centre site level 0's filtered sample is the texel itself and the target is the source pixel itself; at a
// fractional site level 0, level 1 and the source are all read bilinearly with the clamp.
__host__ __device__ inline void site_row(const float* v0, const float* v1, const float* src, int w0, int h0, int w1,
                                         int h1, int c0, int c1, int nout, int k_count, int j, int px, int py,
                                         float* phi, float* target)
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
