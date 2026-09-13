// objective.cu: the objective, and the encoder's own decode of a stored level.
//
// The objective is the weighted squared error of the decoded material over every site of every stored plane:
//
//     E = sum over planes m of w_m * sum over the sites of plane m of sum over outputs c of cw[c] (out_c - t_c)^2
//     out = W phi(z) + b, phi in the shader's order
//
// The sites are the centre of every pixel plus the K fixed subtexel positions of sample.cuh, and cw[c] is the output's
// own weight, weights[c / 3] * rgb_weights[c % 3] * 3 / sum(rgb_weights). The mip weight w_m is the per-site weight of
// mips.cpp, so a plane's sites carry exactly the share of the objective that plane was given.
//
// Reported numbers are normalised by sum(cw) times the total weighted site count, so E reads like a mean squared error
// in [0,1] units: at K = 0 with one stored plane and the default weights it IS the plain mean squared error of the
// decode at the texel centres, which is the scalar reference main.cpp asserts against.
//
// Everything inside this file is evaluated in double, on both paths. The planes, the source and the decoder are fp32
// and are the same bytes for both, so the device pass and the host twin behind --check differ only in the order their
// partial sums are added, and a disagreement above that noise means a real disagreement about the site set, the
// feature order, the weights or the normalisation. The solver's own kernels stay in fp32; E is a measurement.

#include <cmath>
#include <vector>

#include "device.cuh"
#include "sample.cuh"

static const int OBJ_THREADS = 256;

// The objective is not part of any block: it is the measurement the round loop is held to, taken after each of the
// three blocks so that the log carries the evidence that none of them raised E. It is therefore its own line in the
// report's time split, and this is where it is counted.
static double g_objective_ms = 0.0;
static long long g_objective_calls = 0;

double objective_total_ms()
{
    return g_objective_ms;
}

long long objective_passes()
{
    return g_objective_calls;
}

// ---------------------------------------------------------------------------------------------------------------
// The encoder's own decode of a stored level (the recon PNGs)
// ---------------------------------------------------------------------------------------------------------------
//
// For pixel p of stored level m, out = W phi(z) + b with s_i level 0's plane at the texel p and c_j level 1's plane
// sampled bilinearly at the pixel centre. The output rule is the identity followed by the caller's saturate, so the
// bytes below are the arithmetic the GPU performs on the shipped asset.

__global__ void k_decode(const float* v0, const float* v1, int w0, int h0, int w1, int h1, int c0, int c1, int nin,
                         int nout, const float* weights, const float* bias, uint8_t* out)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)w0 * h0)
        return;
    const int px = (int)(idx % (size_t)w0), py = (int)(idx / (size_t)w0);
    float s[MAX_CHANNELS], c[MAX_CHANNELS], phi[MAX_NIN];
    for (int i = 0; i < c0; i++)
        s[i] = v0[idx * c0 + i];
    const BilinearTap t = bilinear_tap_pixel(px, py, 0.0f, 0.0f, w0, h0, w1, h1);
    sample_bilinear(v1, w1, c1, t, c);
    build_phi(s, c0, c, c1, phi);
    for (int r = 0; r < nout; r++)
    {
        float acc = bias[r];
        const float* row = weights + (size_t)r * nin;
        for (int i = 0; i < nin; i++)
            acc += row[i] * phi[i];
        acc = acc < 0.0f ? 0.0f : (acc > 1.0f ? 1.0f : acc);
        out[idx * nout + r] = (uint8_t)lroundf(acc * 255.0f);
    }
}

void decode_plane(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8)
{
    const DevPlane& p = d->planes[plane];
    const size_t n = (size_t)p.w0 * p.h0;
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    // On the master stream, which the planes fan into at the end of every block: the decode reads planes their own
    // streams write, and the legacy stream is ordered against neither.
    k_decode<<<blocks, threads, 0, d->master>>>(p.v0, p.v1, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout,
                                                d->weights, d->bias, d->rgb8);
    CUDA_CHECK(cudaGetLastError());
    rgb8.resize(n * m.nout);
    CUDA_CHECK(cudaMemcpyAsync(rgb8.data(), d->rgb8, n * m.nout, cudaMemcpyDeviceToHost, d->master));
    CUDA_CHECK(cudaStreamSynchronize(d->master));
}

// ---------------------------------------------------------------------------------------------------------------
// The device pass
// ---------------------------------------------------------------------------------------------------------------

__device__ inline int obj_clamp(int i, int n)
{
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// The hardware's bilinear rule in double: x = u w - 0.5, the two texel indices clamped into the plane so an edge
// position's duplicated tap simply adds its weight again.
__device__ inline void obj_sample(const float* plane, int w, int h, int c, double u, double v, double* out)
{
    const double x = u * (double)w - 0.5, y = v * (double)h - 0.5;
    const double xf = floor(x), yf = floor(y);
    const double fx = x - xf, fy = y - yf;
    const int x0 = obj_clamp((int)xf, w), x1 = obj_clamp((int)xf + 1, w);
    const int y0 = obj_clamp((int)yf, h), y1 = obj_clamp((int)yf + 1, h);
    const float* a = plane + ((size_t)y0 * w + x0) * c;
    const float* b = plane + ((size_t)y0 * w + x1) * c;
    const float* e = plane + ((size_t)y1 * w + x0) * c;
    const float* f = plane + ((size_t)y1 * w + x1) * c;
    const double w00 = (1.0 - fx) * (1.0 - fy), w10 = fx * (1.0 - fy), w01 = (1.0 - fx) * fy, w11 = fx * fy;
    for (int k = 0; k < c; k++)
        out[k] = w00 * (double)a[k] + w10 * (double)b[k] + w01 * (double)e[k] + w11 * (double)f[k];
}

// Three sums for one plane: the cw-weighted squared error over every site, and - unweighted, for the two PSNR lines -
// the squared error over the centre sites alone and over the fractional sites alone. The mip weight is applied by the
// caller, because every site of a plane carries the same one.
__global__ void k_objective(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1,
                            int c0, int c1, int nin, int nout, const float* weights, const float* bias,
                            const float* cw, int k_count, double* partial)
{
    __shared__ double sh[3 * OBJ_THREADS];
    const int tid = (int)threadIdx.x;
    const int per_pixel = 1 + k_count;
    const size_t sites = (size_t)w0 * h0 * per_pixel;

    double weighted = 0.0, centre = 0.0, fractional = 0.0;

    // A fixed grid, each block walking its own stride of the sites in increasing order: the same sites added in the
    // same order whatever the scheduler does. E decides when the loop stops, so a reduction that depended on the
    // scheduling would make two identical runs produce two different assets.
    for (size_t si = (size_t)blockIdx.x * blockDim.x + tid; si < sites; si += (size_t)gridDim.x * blockDim.x)
    {
        const size_t pixel = si / (size_t)per_pixel;
        const int j = (int)(si % (size_t)per_pixel);
        const int px = (int)(pixel % (size_t)w0), py = (int)(pixel / (size_t)w0);

        float dx = 0.0f, dy = 0.0f;
        if (j > 0)
            subtexel_offset(k_count, j - 1, px, py, dx, dy);
        const double u = ((double)px + 0.5 + (double)dx) / (double)w0;
        const double v = ((double)py + 0.5 + (double)dy) / (double)h0;

        double s[MAX_CHANNELS], c[MAX_CHANNELS], target[MAX_NOUT];
        obj_sample(v1, w1, h1, c1, u, v, c);
        if (j == 0)
        {
            for (int i = 0; i < c0; i++)
                s[i] = (double)v0[pixel * c0 + i];
            for (int i = 0; i < nout; i++)
                target[i] = (double)src[pixel * nout + i];
        }
        else
        {
            obj_sample(v0, w0, h0, c0, u, v, s);
            obj_sample(src, w0, h0, nout, u, v, target);
        }

        double phi[MAX_NIN];
        int n = 0;
        for (int q = 0; q < c1; q++)
            phi[n++] = c[q];
        for (int i = 0; i < c0; i++)
            phi[n++] = s[i];
        for (int i = 0; i < c0; i++)
            for (int q = 0; q < c1; q++)
                phi[n++] = s[i] * c[q];

        for (int r = 0; r < nout; r++)
        {
            double acc = (double)bias[r];
            const float* row = weights + (size_t)r * nin;
            for (int i = 0; i < nin; i++)
                acc += (double)row[i] * phi[i];
            const double e = acc - target[r];
            weighted += (double)cw[r] * e * e;
            if (j == 0)
                centre += e * e;
            else
                fractional += e * e;
        }
    }

    // Stage one: the block's sites reduced in shared memory, a fixed tree. Stage two: the block's three partials into
    // its own three slots, which k_obj_reduce then adds by index.
    sh[tid] = weighted;
    sh[OBJ_THREADS + tid] = centre;
    sh[2 * OBJ_THREADS + tid] = fractional;
    __syncthreads();
    for (int half = OBJ_THREADS / 2; half > 0; half >>= 1)
    {
        if (tid < half)
        {
            sh[tid] += sh[tid + half];
            sh[OBJ_THREADS + tid] += sh[OBJ_THREADS + tid + half];
            sh[2 * OBJ_THREADS + tid] += sh[2 * OBJ_THREADS + tid + half];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        partial[(size_t)0 * SITE_BLOCKS + blockIdx.x] = sh[0];
        partial[(size_t)1 * SITE_BLOCKS + blockIdx.x] = sh[OBJ_THREADS];
        partial[(size_t)2 * SITE_BLOCKS + blockIdx.x] = sh[2 * OBJ_THREADS];
    }
}

// The second stage of one plane's three sums: the blocks added by index, one thread per sum.
__global__ void k_obj_reduce(const double* partial, double* out)
{
    const int row = (int)threadIdx.x;
    double sum = 0.0;
    for (int b = 0; b < SITE_BLOCKS; b++)
        sum += partial[(size_t)row * SITE_BLOCKS + b];
    out[row] = sum;
}

// The three raw sums of every plane turned into the reported numbers.
static void normalise(const Model& m, int k, const std::vector<double>& per_site, const std::vector<double>& acc,
                      Objective& r)
{
    double sum_cw = 0.0;
    for (int c = 0; c < m.nout; c++)
        sum_cw += (double)m.cw[c];

    double num = 0.0, den = 0.0;
    r.e_plane.assign(m.planes.size(), 0.0);
    r.centre_plane.assign(m.planes.size(), 0.0);
    for (size_t p = 0; p < m.planes.size(); p++)
    {
        const double pixels = (double)m.planes[p].w0 * (double)m.planes[p].h0;
        const double sites = pixels * (double)(1 + k);
        r.e_plane[p] = acc[3 * p] / (sum_cw * sites);
        // The plane's own centre error as a mean over its pixels and its outputs: the mean squared error its level's
        // PSNR is taken from, in the same fp units as the base's.
        r.centre_plane[p] = acc[3 * p + 1] / (pixels * (double)m.nout);
        num += per_site[p] * acc[3 * p];
        den += per_site[p] * sites;
    }
    r.e = den > 0.0 ? num / (sum_cw * den) : 0.0;

    const double base = (double)m.planes[0].w0 * (double)m.planes[0].h0 * (double)m.nout;
    r.centre_mse = acc[1] / base;
    r.sampled_mse = k > 0 ? acc[2] / (base * (double)k) : 0.0;
}

void objective_eval(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    const size_t np = d->planes.size();

    // Every plane on its own stream, as the blocks are: a pass over a deep plane is a handful of blocks of work and
    // would otherwise cost more in launches than it does in arithmetic. Each plane reduces its own partials into its
    // own triple, so the planes still share nothing.
    device_block_begin(d);
    for (size_t p = 0; p < np; p++)
    {
        const DevPlane& dp = d->planes[p];
        double* partial = d->obj_partial + 3 * p * (size_t)SITE_BLOCKS;
        k_objective<<<SITE_BLOCKS, OBJ_THREADS, 0, dp.stream>>>(dp.v0, dp.v1, dp.src, dp.w0, dp.h0, dp.w1, dp.h1, m.c0,
                                                                m.c1, m.dec.nin, m.nout, d->weights, d->bias, d->cw, k,
                                                                partial);
        CUDA_CHECK(cudaGetLastError());
        k_obj_reduce<<<1, 3, 0, dp.stream>>>(partial, d->obj + 3 * p);
        CUDA_CHECK(cudaGetLastError());
    }
    const double ms = device_block_end(d);

    std::vector<double> acc(3 * np, 0.0);
    CUDA_CHECK(cudaMemcpy(acc.data(), d->obj, 3 * np * sizeof(double), cudaMemcpyDeviceToHost));
    normalise(m, k, per_site, acc, r);
    r.ms = ms;
    g_objective_ms += ms;
    g_objective_calls++;
}

// ---------------------------------------------------------------------------------------------------------------
// The brute-force twin behind --check
// ---------------------------------------------------------------------------------------------------------------
//
// Deliberately the slowest, plainest thing that can compute E: download every plane, walk every pixel and every site
// of every plane in order, sample each plane by hand and accumulate. It shares nothing with the pass above but the
// table of subtexel offsets, so when the two agree, the site set, the feature order, the weights and the
// normalisation agree - which is the only thing this check is for.

static void download(const float* device_ptr, size_t count, std::vector<float>& host)
{
    host.resize(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
}

// The same bilinear rule, written out again: x = u w - 0.5, floor, both taps clamped into the plane, blend.
static void twin_sample(const std::vector<float>& plane, int w, int h, int c, double u, double v, double* out)
{
    const double x = u * (double)w - 0.5, y = v * (double)h - 0.5;
    const double xf = std::floor(x), yf = std::floor(y);
    const double fx = x - xf, fy = y - yf;
    int x0 = (int)xf, x1 = (int)xf + 1, y0 = (int)yf, y1 = (int)yf + 1;
    x0 = x0 < 0 ? 0 : (x0 >= w ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= w ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 >= h ? h - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 >= h ? h - 1 : y1);
    for (int k = 0; k < c; k++)
    {
        const double a = (double)plane[((size_t)y0 * w + x0) * c + k];
        const double b = (double)plane[((size_t)y0 * w + x1) * c + k];
        const double e = (double)plane[((size_t)y1 * w + x0) * c + k];
        const double f = (double)plane[((size_t)y1 * w + x1) * c + k];
        out[k] = (1.0 - fx) * (1.0 - fy) * a + fx * (1.0 - fy) * b + (1.0 - fx) * fy * e + fx * fy * f;
    }
}

void objective_check_host(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    const size_t np = d->planes.size();
    std::vector<double> acc(3 * np, 0.0);
    const int c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;

    for (size_t p = 0; p < np; p++)
    {
        const DevPlane& dp = d->planes[p];
        std::vector<float> v0, v1, src;
        download(dp.v0, (size_t)dp.w0 * dp.h0 * c0, v0);
        download(dp.v1, (size_t)dp.w1 * dp.h1 * c1, v1);
        download(dp.src, (size_t)dp.w0 * dp.h0 * nout, src);

        for (int py = 0; py < dp.h0; py++)
            for (int px = 0; px < dp.w0; px++)
                for (int j = 0; j <= k; j++)
                {
                    float dx = 0.0f, dy = 0.0f;
                    if (j > 0)
                        subtexel_offset(k, j - 1, px, py, dx, dy);
                    const double u = ((double)px + 0.5 + (double)dx) / (double)dp.w0;
                    const double v = ((double)py + 0.5 + (double)dy) / (double)dp.h0;

                    double s[MAX_CHANNELS], c[MAX_CHANNELS], target[MAX_NOUT];
                    twin_sample(v1, dp.w1, dp.h1, c1, u, v, c);
                    if (j == 0)
                    {
                        const size_t idx = (size_t)py * dp.w0 + px;
                        for (int i = 0; i < c0; i++)
                            s[i] = (double)v0[idx * c0 + i];
                        for (int i = 0; i < nout; i++)
                            target[i] = (double)src[idx * nout + i];
                    }
                    else
                    {
                        twin_sample(v0, dp.w0, dp.h0, c0, u, v, s);
                        twin_sample(src, dp.w0, dp.h0, nout, u, v, target);
                    }

                    double phi[MAX_NIN];
                    int n = 0;
                    for (int q = 0; q < c1; q++)
                        phi[n++] = c[q];
                    for (int i = 0; i < c0; i++)
                        phi[n++] = s[i];
                    for (int i = 0; i < c0; i++)
                        for (int q = 0; q < c1; q++)
                            phi[n++] = s[i] * c[q];

                    for (int row = 0; row < nout; row++)
                    {
                        double y = (double)m.dec.b[row];
                        for (int i = 0; i < nin; i++)
                            y += (double)m.dec.w[(size_t)row * nin + i] * phi[i];
                        const double e = y - target[row];
                        acc[3 * p] += (double)m.cw[row] * e * e;
                        acc[3 * p + (j == 0 ? 1 : 2)] += e * e;
                    }
                }
    }
    normalise(m, k, per_site, acc, r);
    r.ms = 0.0;
}
