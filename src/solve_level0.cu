// solve_level0.cu: block (c), level 0 as one exact search per texel.
//
// Hold level 1 and the decoder fixed and ask, for one level-0 texel, which of its palette states minimises the
// objective. Every other texel stays where it is, so only the sites whose level-0 footprint touches this texel can
// move at all, and over those sites the objective is an exact quadratic in the texel's own C0 dequantised values.
//
// THE QUADRATIC. The decoder is affine in its features and the features are affine in the level-0 sample for a fixed
// level-1 sample, so at a site whose level-1 sample is c and whose level-0 sample is s,
//
//     out = b + A c + (B + sum_j c_j M_j) s ,     A = the a-block of W, B the b-block, M_j the sc-columns of c_j
//
// Split the site's level-0 sample into this texel's part and the rest: s = beta + alpha x, where x is the texel's own
// value vector, alpha is the weight this texel carries in that site's blend and beta is what the other taps contribute.
// Then out(x) = p + alpha Q x, with
//
//     p      = the decode with this texel's channels at zero and everything else as it stands  (nout)
//     Q[:,i] = B[:,i] + sum_j c_j M_i,j                                                        (nout x C0)
//
// and, writing r = p - target and stacking the site's contributions,
//
//     E(x) = A0 + 2 x.A1 + x.A2.x ,   A0 += r^T diag(cw) r ,   A1 += alpha Q^T diag(cw) r ,   A2 += alpha^2 Q^T diag(cw) Q
//
// which is the same quadratic C0 + 1 decodes per site would produce - one at x = 0 and one per channel moved by alpha -
// with the decodes replaced by the affine structure they were sampling.
//
// THE SITES of one texel, because level 0 is at full resolution and a texel is a pixel:
//
//   - its own CENTRE site, where level 0 is read nearest, so the site's whole level-0 sample is this texel: alpha = 1,
//     beta = 0, and the site depends on no other texel;
//   - the K fractional sites of each pixel of the 3x3 neighbourhood, where level 0 is read bilinearly: a position
//     inside a pixel addresses the two texels either side of it in each axis, so the pixels whose four-tap footprint
//     can reach this texel are exactly this pixel and its eight neighbours. alpha is the texel's weight in that blend,
//     and a texel that appears twice in the four taps because the clamp pulled both indices onto it carries the SUM of
//     the two weights - the hardware's own rule, and the rule the objective is measured under.
//
// A site whose alpha is zero contributes the same constant to every state and is skipped; A0 is therefore not the
// texel's share of E, only the part of it that this texel can move plus a constant, which is all an argmin needs.
//
// THE ARGMIN is exact. With sum of bits0 at most 8 the joint enumeration walks all prod_c 2^bits_c states - at most 256
// - and takes the best; above that it is --sweeps coordinate sweeps, each channel in turn over its whole alphabet with
// the others held, which is exact in one channel but not jointly. Ties keep the current state: only a strict decrease
// moves a texel, which is what stops a pair of equal-valued states from cycling for ever.
//
// THE FOUR COLOUR PASSES. A pass moves the texels of one (tx & 1, ty & 1) class, one kernel launch each. Two texels of
// one class are two apart in an axis, and a site's four taps are adjacent, so no site can touch two of them: within a
// pass the texels' quadratics are independent of each other, which makes the pass an exact joint minimisation over all
// of them at once and not merely a sequence of single-texel steps. That is also why the kernel may write its texels in
// place while reading its neighbours: every neighbour a site of this pass reads belongs to another class.
//
// The accumulators are double. A texel sees at most 9K + 1 sites - its own centre site, where level 0 is read nearest,
// plus the K fractional sites of each of the nine pixels around it - so the cost of the wider accumulation is
// nothing next to the sampling around it, and the argmin compares states that can differ in the last few bits of a
// float - a comparison the tie rule then turns into a decision about whether a texel moves.

#include <vector>

#include "device.cuh"
#include "sample.cuh"

static const int SEARCH_THREADS = 128;

// E(x) = A0 + 2 x.A1 + x.A2.x, with A2 held in full (it is symmetric, and C0 is at most 4).
__device__ inline double search_energy(double a0, const double* a1, const double* a2, const float* x, int c0)
{
    double e = a0;
    for (int i = 0; i < c0; i++)
    {
        e += 2.0 * (double)x[i] * a1[i];
        for (int j = 0; j < c0; j++)
            e += (double)x[i] * (double)x[j] * a2[i * c0 + j];
    }
    return e;
}

__global__ void k_level0_search(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1,
                                int c0, int c1, int nin, int nout, const float* weights, const float* bias,
                                const float* cw, const float* palette, ChannelBits bits, int k_count, int colour_x,
                                int colour_y, int joint, int sweeps, float* v0_out, uint8_t* k0,
                                unsigned int* moved)
{
    // One thread per texel OF THIS COLOUR: the pass's texels are the sub-grid (colour_x + 2 i, colour_y + 2 j).
    const int cw0 = (w0 - colour_x + 1) / 2, ch0 = (h0 - colour_y + 1) / 2;
    const size_t ci = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= (size_t)cw0 * ch0)
        return;
    const int tx = colour_x + 2 * (int)(ci % (size_t)cw0);
    const int ty = colour_y + 2 * (int)(ci / (size_t)cw0);
    const size_t t = (size_t)ty * w0 + tx;

    double a0 = 0.0, a1[MAX_CHANNELS], a2[MAX_CHANNELS * MAX_CHANNELS];
    for (int i = 0; i < c0; i++)
        a1[i] = 0.0;
    for (int i = 0; i < c0 * c0; i++)
        a2[i] = 0.0;

    // One site: its level-1 sample cbar, the level-0 contribution of the other taps beta, this texel's weight alpha and
    // the target. Everything the quadratic needs is p = decode(beta) and the columns Q.
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            const int px = tx + dx, py = ty + dy;
            if (px < 0 || py < 0 || px >= w0 || py >= h0)
                continue;
            const int first = (dx == 0 && dy == 0) ? 0 : 1;   // only this texel's own pixel carries a centre site
            for (int j = first; j <= k_count; j++)
            {
                float dx, dy;
                site_delta(k_count, j, px, py, dx, dy);

                float alpha = 0.0f, beta[MAX_CHANNELS], cbar[MAX_CHANNELS], target[MAX_NOUT];
                const BilinearTap t1 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w1, h1);
                sample_bilinear(v1, w1, c1, t1, cbar);
                for (int i = 0; i < c0; i++)
                    beta[i] = 0.0f;

                if (j == 0)
                {
                    // The centre site reads level 0 nearest, so the texel IS the sample.
                    alpha = 1.0f;
                    for (int r = 0; r < nout; r++)
                        target[r] = src[t * nout + r];
                }
                else
                {
                    const BilinearTap t0 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w0, h0);
                    sample_bilinear(src, w0, nout, t0, target);
                    const int cx[4] = { t0.x0, t0.x1, t0.x0, t0.x1 };
                    const int cy[4] = { t0.y0, t0.y0, t0.y1, t0.y1 };
                    const float cwt[4] = { (1.0f - t0.fx) * (1.0f - t0.fy), t0.fx * (1.0f - t0.fy),
                                           (1.0f - t0.fx) * t0.fy, t0.fx * t0.fy };
                    for (int q = 0; q < 4; q++)
                    {
                        // A tap the clamp pulled onto this texel adds its weight to alpha; every other tap's value
                        // goes into beta. A texel appearing twice therefore carries the sum of both weights.
                        if (cx[q] == tx && cy[q] == ty)
                            alpha += cwt[q];
                        else
                        {
                            const float* nv = v0 + ((size_t)cy[q] * w0 + cx[q]) * c0;
                            for (int i = 0; i < c0; i++)
                                beta[i] += cwt[q] * nv[i];
                        }
                    }
                    if (!(alpha > 0.0f))
                        continue;   // this site cannot see the texel, so it is the same constant in every state
                }

                // p = the decode with this texel's channels at zero, and Q's columns, which are the decode's slope in
                // each of them at this site's level-1 sample.
                float qcol[MAX_NOUT * MAX_CHANNELS], resid[MAX_NOUT];
                for (int r = 0; r < nout; r++)
                {
                    const float* row = weights + (size_t)r * nin;
                    float out = bias[r];
                    for (int q = 0; q < c1; q++)
                        out += row[q] * cbar[q];
                    for (int i = 0; i < c0; i++)
                    {
                        float slope = row[c1 + i];
                        for (int q = 0; q < c1; q++)
                            slope += row[c1 + c0 + i * c1 + q] * cbar[q];
                        qcol[r * c0 + i] = slope;
                        out += slope * beta[i];
                    }
                    resid[r] = out - target[r];
                }

                for (int i = 0; i < nout; i++)
                    a0 += (double)cw[i] * (double)resid[i] * (double)resid[i];
                for (int i = 0; i < c0; i++)
                {
                    double g = 0.0;
                    for (int r = 0; r < nout; r++)
                        g += (double)cw[r] * (double)qcol[r * c0 + i] * (double)resid[r];
                    a1[i] += (double)alpha * g;
                    for (int j2 = 0; j2 < c0; j2++)
                    {
                        double s = 0.0;
                        for (int r = 0; r < nout; r++)
                            s += (double)cw[r] * (double)qcol[r * c0 + i] * (double)qcol[r * c0 + j2];
                        a2[i * c0 + j2] += (double)alpha * (double)alpha * s;
                    }
                }
            }
        }

    // The argmin over the palette, starting from the current state so that a tie keeps it.
    int cur[MAX_CHANNELS], best[MAX_CHANNELS], levels[MAX_CHANNELS];
    float x[MAX_CHANNELS];
    for (int i = 0; i < c0; i++)
    {
        levels[i] = (1 << bits.b[i]) - 1;
        cur[i] = (int)k0[t * c0 + i];
        best[i] = cur[i];
        x[i] = palette[(size_t)i * 16 + cur[i]];
    }
    double best_e = search_energy(a0, a1, a2, x, c0);

    if (joint)
    {
        long long total = 1;
        for (int i = 0; i < c0; i++)
            total *= (long long)(levels[i] + 1);
        for (long long n = 0; n < total; n++)
        {
            long long rest = n;
            bool same = true;
            int state[MAX_CHANNELS];
            for (int i = 0; i < c0; i++)
            {
                state[i] = (int)(rest % (long long)(levels[i] + 1));
                rest /= (long long)(levels[i] + 1);
                if (state[i] != cur[i])
                    same = false;
                x[i] = palette[(size_t)i * 16 + state[i]];
            }
            if (same)
                continue;
            const double e = search_energy(a0, a1, a2, x, c0);
            if (e < best_e)
            {
                best_e = e;
                for (int i = 0; i < c0; i++)
                    best[i] = state[i];
            }
        }
    }
    else
    {
        for (int sw = 0; sw < sweeps; sw++)
        {
            bool changed = false;
            for (int i = 0; i < c0; i++)
            {
                for (int q = 0; q < c0; q++)
                    x[q] = palette[(size_t)q * 16 + best[q]];
                const int hold = best[i];
                for (int k = 0; k <= levels[i]; k++)
                {
                    if (k == best[i])
                        continue;
                    x[i] = palette[(size_t)i * 16 + k];
                    const double e = search_energy(a0, a1, a2, x, c0);
                    if (e < best_e)
                    {
                        best_e = e;
                        best[i] = k;
                    }
                }
                if (best[i] != hold)
                    changed = true;
            }
            if (!changed)
                break;   // a sweep that moves nothing cannot be followed by one that does
        }
    }

    bool moved_here = false;
    for (int i = 0; i < c0; i++)
        moved_here = moved_here || best[i] != cur[i];
    if (moved_here)
    {
        for (int i = 0; i < c0; i++)
        {
            v0_out[t * c0 + i] = palette[(size_t)i * 16 + best[i]];
            k0[t * c0 + i] = (uint8_t)best[i];
        }
        atomicAdd(moved, 1u);
    }
}

// Block (c) on every plane: four colour passes each, every plane on its own stream, all of them waited for once. With
// level 1 and the decoder held the planes are independent problems, so the only ordering the streams need is at the end
// of the block, and the deep planes - whose passes are a few thousand texels - ride along behind the base's.
double solve_level0_all(DeviceModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep)
{
    const size_t np = d->planes.size();
    rep.assign(np, Level0Report());
    ChannelBits bits = {};
    long long states = 1;
    int total_bits = 0;
    for (int c = 0; c < m.c0; c++)
    {
        bits.b[c] = m.bits0[(size_t)c];
        states *= (long long)1 << m.bits0[(size_t)c];
        total_bits += m.bits0[(size_t)c];
    }
    const bool joint = total_bits <= 8;

    device_block_begin(d);
    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        // One counter per colour pass, copied back once at the end: nothing in the block waits on the host.
        CUDA_CHECK(cudaMemsetAsync(p.counters + 1, 0, 4 * sizeof(unsigned int), p.stream));
        CUDA_CHECK(cudaEventRecord(p.ev[EV_C_START], p.stream));
        for (int colour = 0; colour < 4; colour++)
        {
            const int cx = colour & 1, cy = colour >> 1;
            const size_t texels = (size_t)((p.w0 - cx + 1) / 2) * (size_t)((p.h0 - cy + 1) / 2);
            if (texels == 0)
                continue;
            const int blocks = (int)((texels + SEARCH_THREADS - 1) / SEARCH_THREADS);
            k_level0_search<<<blocks, SEARCH_THREADS, 0, p.stream>>>(
                p.v0, p.v1, p.src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout, d->weights, d->bias, d->cw,
                d->palette, bits, k, cx, cy, joint ? 1 : 0, sweeps, p.v0, p.k0, p.counters + 1 + colour);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(p.ev[EV_C_END], p.stream));
        CUDA_CHECK(cudaMemcpyAsync(d->host_counters + i * 5 + 1, p.counters + 1, 4 * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, p.stream));
    }

    const double ms = device_block_end(d);
    for (size_t i = 0; i < np; i++)
    {
        rep[i].joint = joint;
        rep[i].states = states;
        rep[i].ms = device_elapsed(d->planes[i], EV_C_START, EV_C_END);
        for (int colour = 0; colour < 4; colour++)
        {
            rep[i].moved[colour] = (long long)d->host_counters[i * 5 + 1 + colour];
            rep[i].moved_total += rep[i].moved[colour];
        }
    }
    return ms;
}

// The plane's level-0 values replaced by the caller's: the same texels, on the values the shipped format decodes to
// rather than on the ones the search chose. The upload goes through the master stream and fans out, so a plane's own
// stream cannot read the old plane afterwards.
void level0_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v0)
{
    DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w0 * p.h0 * m.c0;
    if (v0.size() != n)
        return;
    CUDA_CHECK(cudaMemcpyAsync(p.v0, v0.data(), n * sizeof(float), cudaMemcpyHostToDevice, d->master));
    CUDA_CHECK(cudaStreamSynchronize(d->master));
    device_fan_out(d);
}

// The plane's dequantised values as the device holds them, which under --l0 bc8 is what the grid is fitted over.
// The plane's indices written back, which the outer repack loop's restore needs beside its values.
void level0_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k0)
{
    DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w0 * p.h0 * m.c0;
    if (k0.size() != n)
        return;
    device_upload_fan_out(d, p.k0, k0.data(), n);
}

void level0_download_values(DeviceModel* d, const Model& m, int plane, std::vector<float>& v0)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w0 * p.h0 * m.c0;
    v0.resize(n);
    CUDA_CHECK(cudaMemcpy(v0.data(), p.v0, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// The plane's palette indices as the .dds will store them.
void level0_download(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k0)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w0 * p.h0 * m.c0;
    k0.resize(n);
    CUDA_CHECK(cudaMemcpy(k0.data(), p.k0, n, cudaMemcpyDeviceToHost));
}
