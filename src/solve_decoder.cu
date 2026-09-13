// solve_decoder.cu: block (a), the decoder as one global least squares.
//
// With v = [phi ; 1] of length mm = nin + 1, the exact minimiser of
//
//     E(W, b) = sum over sites of omega_site * sum over outputs c of cw[c] ( W_c . phi + b_c - t_c )^2
//
// is the solution of the mm x mm normal equations
//
//     A       = sum over sites of omega_site * v v^T          (one matrix, shared by every output)
//     rhs_c   = sum over sites of omega_site * v t_c          (one right-hand side per output)
//     [W_c b_c] = A^-1 rhs_c .
//
// The per-output weight cw[c] scales output c's own equation uniformly and therefore cannot move its argmin, which is
// why it does not appear above. The per-site weight omega does not cancel: it multiplies v v^T as well as the
// right-hand side, and it is what makes a deep mip plane count for less (or more) than the base. omega is the per-site
// mip weight of mips.cpp, the same for every site of a plane, so each plane's block reduction is scaled once.
//
// The site set is every site of every stored plane: the centre of each pixel and the K fixed subtexel positions.
// There is no step size and no schedule: the block takes the argmin in (W, b) of the whole objective at once.
//
// The outer products are accumulated on the device (a block gathers its sites' v and t into shared memory, then walks
// the mm*mm + nout*mm entries, one dot product each, in double) and the small dense system is solved on the host by
// Gauss-Jordan with partial pivoting in double. A ridge of 1e-9 times the mean diagonal keeps a constant feature -
// level 0's second channel starts at zero everywhere - from making A singular. What is solved is therefore the RIDGED
// problem, whose minimiser is the true one only while the ridge is negligible against it; where it is not, the block
// falls back on smaller ridges and finally on the decoder it came in with, and the objective's own quadratic - which A
// and rhs already are - says which. See the ladder at the foot of this file; E cannot rise across the block.
//
// The accumulation is a two-stage reduction on a FIXED grid, not a set of atomic adds. Floating-point addition is not
// associative, so an atomic accumulator returns whatever the blocks' scheduling made of it, and two identical runs can
// then produce two different decoders and two different assets. Instead each of the SITE_BLOCKS blocks of a plane walks
// its own strided share of the sites, holds its own partial of every entry in registers and writes it to its own slot;
// a second pass adds, for each entry, the planes in index order and the blocks in index order, applying each plane's
// site weight to that plane's own sum. Every addition happens in the same order every time, so an encode of one image
// is reproducible bit for bit.

#include <algorithm>
#include <cmath>
#include <vector>

#include "device.cuh"
#include "sample.cuh"

static const int LS_THREADS = 256;

// The stride between two entries of the staged block in shared memory. A block stages its sites' v and t entry-major
// and then walks the entries, one dot product each: at a given site every thread of a warp reads a DIFFERENT entry at
// the SAME site, so with a stride that is a multiple of 32 floats those reads all fall in one bank and the warp is
// serialised thirty-two ways. One float of padding puts consecutive entries in consecutive banks. It is not what makes
// this kernel fast - the walk is bound by its double multiply-adds, at a sixty-fourth of the device's float rate, and
// removing the conflict alone changed nothing measurable - but a thirty-two-way conflict has no business being left in.
static const int LS_STRIDE = LS_THREADS + 1;

// A is symmetric, so only its upper triangle is accumulated: entry (i, j) with i <= j lives at
//
//     tri_index(i, j) = i mm - i (i - 1) / 2 + (j - i)
//
// and the host mirrors it into the full matrix before the solve. That is not only a third fewer multiply-adds; it is
// what brings the entry count under one per thread, so the walk is one pass in which every thread carries one entry
// instead of two passes in which most of them carry nothing.
__host__ __device__ inline int tri_index(int i, int j, int mm)
{
    return i * mm - i * (i - 1) / 2 + (j - i);
}

// The most entries one thread can end up carrying: the walk hands thread `tid` the entries tid, tid + LS_THREADS, ...,
// so at the channel caps (nin 24, nout 18, hence 325 triangle entries plus 450 right-hand-side entries = 775) it is
// four. The cap moved from twelve outputs to eighteen with the six-texture material; the comment said three for a long
// time after it stopped being true, which is the kind of number a reader trusts and should not have to recompute.
static const int LS_SLOTS = (((MAX_NIN + 1) * (MAX_NIN + 2) / 2 + MAX_NOUT * (MAX_NIN + 1)) + LS_THREADS - 1) /
                            LS_THREADS;

// The staged block is the one thing here that grows with the caps, and it lives in shared memory, whose per-block
// limit is 48 KB on every architecture this builds for. The launch below asks for
//
//     (mm + nout) * LS_STRIDE * sizeof(float),   mm = nin + 1
//
// which at the caps (nin 24, nout 18, LS_STRIDE 257) is (25 + 18) * 257 * 4 = 44,204 bytes. Raising MAX_TEXTURES or
// MAX_NIN far enough would make the launch fail at run time with nothing in the source to say why, so it fails here
// instead, at compile time, where the constant that did it is in view.
static_assert((size_t)(MAX_NIN + 1 + MAX_NOUT) * LS_STRIDE * sizeof(float) <= 48u * 1024u,
              "the least-squares staging block must fit the 48 KB of shared memory a block may have");

__global__ void k_ls_accumulate(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1,
                                int c0, int c1, int nin, int nout, int k_count, double* partial)
{
    extern __shared__ float sh[];
    const int mm = nin + 1;
    float* sv = sh;                                 // entry-major: sv[i * LS_STRIDE + tid]
    float* st = sh + (size_t)mm * LS_STRIDE;        // st[c * LS_STRIDE + tid]
    const int tid = (int)threadIdx.x;
    const int per_pixel = 1 + k_count;
    const size_t sites = (size_t)w0 * h0 * per_pixel;
    const size_t chunks = (sites + LS_THREADS - 1) / LS_THREADS;
    const int tri = mm * (mm + 1) / 2;
    const int entries = tri + nout * mm;

    double acc[LS_SLOTS];
    for (int s = 0; s < LS_SLOTS; s++)
        acc[s] = 0.0;

    // A block takes the chunks of LS_THREADS sites that fall on its own stride, in increasing order. The grid is fixed,
    // so which sites a block adds and in what order is fixed too, whatever the plane's size.
    for (size_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x)
    {
        const size_t si = chunk * LS_THREADS + tid;
        float phi[MAX_NIN];
        for (int i = 0; i < nin; i++)
            phi[i] = 0.0f;
        float target[MAX_NOUT];
        for (int c = 0; c < nout; c++)
            target[c] = 0.0f;
        float one = 0.0f;
        if (si < sites)
        {
            const size_t pixel = si / (size_t)per_pixel;
            const int j = (int)(si % (size_t)per_pixel);
            const int px = (int)(pixel % (size_t)w0), py = (int)(pixel / (size_t)w0);
            site_row(v0, v1, src, w0, h0, w1, h1, c0, c1, nout, k_count, j, px, py, phi, target);
            one = 1.0f;
        }
        __syncthreads();   // the previous chunk's walk has finished reading the staged rows
        for (int i = 0; i < nin; i++)
            sv[(size_t)i * LS_STRIDE + tid] = phi[i];
        sv[(size_t)nin * LS_STRIDE + tid] = one;    // a site outside the plane contributes a row of zeros
        for (int c = 0; c < nout; c++)
            st[(size_t)c * LS_STRIDE + tid] = target[c];
        __syncthreads();

        for (int e = tid, slot = 0; e < entries; e += LS_THREADS, slot++)
        {
            const float* a;
            const float* b;
            if (e < tri)
            {
                // The (i, j) of the upper triangle that entry e stands for, walked row by row.
                int i = 0, rest = e;
                while (rest >= mm - i)
                    rest -= mm - i++;
                a = sv + (size_t)i * LS_STRIDE;
                b = sv + (size_t)(i + rest) * LS_STRIDE;
            }
            else
            {
                const int e2 = e - tri;
                a = st + (size_t)(e2 / mm) * LS_STRIDE;
                b = sv + (size_t)(e2 % mm) * LS_STRIDE;
            }
            double sum = 0.0;
            for (int p = 0; p < LS_THREADS; p++)
                sum += (double)a[p] * (double)b[p];
            acc[slot] += sum;
        }
    }

    for (int e = tid, slot = 0; e < entries; e += LS_THREADS, slot++)
        partial[(size_t)e * SITE_BLOCKS + blockIdx.x] = acc[slot];
}

// The second stage: entry by entry, the planes in index order and inside each plane the blocks in index order, with the
// plane's own site weight applied to its own sum. One thread per entry, and every addition in a fixed place.
__global__ void k_ls_reduce(const double* partial, int entries, int planes, const double* omega, double* out)
{
    const int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= entries)
        return;
    double total = 0.0;
    for (int p = 0; p < planes; p++)
    {
        const double* row = partial + ((size_t)p * entries + e) * SITE_BLOCKS;
        double sum = 0.0;
        for (int b = 0; b < SITE_BLOCKS; b++)
            sum += row[b];
        total += omega[p] * sum;
    }
    out[e] = total;
}

// Gauss-Jordan with partial pivoting on [A | rhs^T], the nout right-hand sides carried along. On return row c of g
// holds output c's solution in its last nout columns.
static bool gauss_jordan(std::vector<double>& g, int m, int nrhs)
{
    const int cols = m + nrhs;
    for (int k = 0; k < m; k++)
    {
        int pivot = k;
        double best = std::fabs(g[(size_t)k * cols + k]);
        for (int i = k + 1; i < m; i++)
        {
            const double a = std::fabs(g[(size_t)i * cols + k]);
            if (a > best)
            {
                best = a;
                pivot = i;
            }
        }
        if (best == 0.0)
            return false;
        if (pivot != k)
            for (int j = 0; j < cols; j++)
                std::swap(g[(size_t)k * cols + j], g[(size_t)pivot * cols + j]);
        const double inv = 1.0 / g[(size_t)k * cols + k];
        for (int j = 0; j < cols; j++)
            g[(size_t)k * cols + j] *= inv;
        for (int i = 0; i < m; i++)
        {
            if (i == k)
                continue;
            const double f = g[(size_t)i * cols + k];
            if (f == 0.0)
                continue;
            for (int j = 0; j < cols; j++)
                g[(size_t)i * cols + j] -= f * g[(size_t)k * cols + j];
        }
    }
    return true;
}

// Returns the kernel time in milliseconds, measured with CUDA events around the accumulation of every plane.
double solve_decoder(DeviceModel* d, Model& m, int k, const std::vector<double>& per_site)
{
    const int nin = m.dec.nin, nout = m.nout, mm = nin + 1;
    const int tri = mm * (mm + 1) / 2;
    const int entries = tri + nout * mm;

    // Each plane writes its own partials, so the planes need no ordering between them at all; the master stream waits
    // for them all and then runs the one in-order pass that turns the partials into the matrix. The timing is read from
    // the base plane's pair of events, which brackets the whole block because its stream is the first to start and the
    // last to finish.
    const size_t planes = d->planes.size();
    CUDA_CHECK(cudaMemcpyAsync(d->ls_omega, per_site.data(), planes * sizeof(double), cudaMemcpyHostToDevice,
                               d->master));
    device_block_begin(d);
    const size_t shared = (size_t)(mm + nout) * LS_STRIDE * sizeof(float);
    for (size_t i = 0; i < planes; i++)
    {
        DevPlane& p = d->planes[i];
        CUDA_CHECK(cudaEventRecord(p.ev[EV_A_START], p.stream));
        k_ls_accumulate<<<SITE_BLOCKS, LS_THREADS, shared, p.stream>>>(
            p.v0, p.v1, p.src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, nin, nout, k,
            d->ls_partial + i * (size_t)entries * SITE_BLOCKS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p.ev[EV_A_END], p.stream));
    }
    const double ms = device_block_end(d);

    const int reduce_threads = 64;
    k_ls_reduce<<<(entries + reduce_threads - 1) / reduce_threads, reduce_threads, 0, d->master>>>(
        d->ls_partial, entries, (int)planes, d->ls_omega, d->reduction);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> acc((size_t)entries, 0.0);
    CUDA_CHECK(cudaMemcpyAsync(acc.data(), d->reduction, (size_t)entries * sizeof(double), cudaMemcpyDeviceToHost,
                               d->master));
    CUDA_CHECK(cudaStreamSynchronize(d->master));

    double trace = 0.0;
    for (int i = 0; i < mm; i++)
        trace += acc[(size_t)tri_index(i, i, mm)];
    const double mean_diag = trace > 0.0 ? trace / mm : 1.0;

    // The decoder the caller arrives with, as the flat x = [W_c ; b_c] the quadratic below is written in. It is the
    // candidate every solve is measured against, and the one that is kept when no solve beats it.
    const bool had_previous = !m.dec.w.empty() && !m.dec.b.empty();
    std::vector<double> x_prev((size_t)nout * mm, 0.0);
    if (had_previous)
        for (int c = 0; c < nout; c++)
        {
            for (int i = 0; i < nin; i++)
                x_prev[(size_t)c * mm + i] = (double)m.dec.w[(size_t)c * nin + i];
            x_prev[(size_t)c * mm + nin] = (double)m.dec.b[c];
        }

    // E's own numerator as a function of the decoder, up to the constant sum omega cw t^2 that does not depend on it:
    //
    //     Q(x) = sum_c cw[c] ( x_c^T A x_c - 2 x_c . rhs_c )
    //
    // A and rhs are exactly the sums E is a quadratic in, so Q(x_new) - Q(x_old) IS the change in E's weighted sum over
    // every site of every plane, and dividing it by sum(cw) times the weighted site count gives the change in the E the
    // round loop prints. It therefore costs no objective pass to know whether a candidate decoder lowers E: a few
    // thousand double multiply-adds on the host, in a fixed order, on numbers that are already here.
    auto qform = [&](const std::vector<double>& x) {
        double q = 0.0;
        for (int c = 0; c < nout; c++)
        {
            double qc = 0.0;
            for (int i = 0; i < mm; i++)
            {
                for (int j = 0; j < mm; j++)
                {
                    const double a = i <= j ? acc[(size_t)tri_index(i, j, mm)] : acc[(size_t)tri_index(j, i, mm)];
                    qc += x[(size_t)c * mm + i] * a * x[(size_t)c * mm + j];
                }
                qc -= 2.0 * x[(size_t)c * mm + i] * acc[(size_t)tri + (size_t)c * mm + i];
            }
            q += (double)m.cw[c] * qc;
        }
        return q;
    };

    // THE RIDGE IS NOT FREE, AND ON SOME IMAGES IT IS WHAT DECIDES THE STEP.
    //
    // The solve returns the minimiser of Q(x) + ridge ||x||^2, not of Q. When A is well conditioned the ridge term is
    // a billionth of the diagonal against a solution of order one and the two minimisers agree to far below any
    // tolerance. When A is nearly singular they do not: a 100x36 black image with one white pixel drives level 1's
    // channels to within 0.16 of constant, the exact minimiser's norm grows past 100, and the ridge's own penalty then
    // moves the solution far enough that Q - and therefore E - RISES across the block. Measured on that image at the
    // default layout: E 1.875599e-06 after round 8's block (c'), 2.112583e-06 after round 9's block (a), a 13 % rise,
    // repeated every round from there; raising the ridge to 1e-6 makes the whole run six times worse (E 8.80e-06
    // against 1.27e-06 at round 20) and still rises, and dropping it to zero makes A singular at the init, where
    // level 0 is a constant plane, and the run collapses at 8.95e-05.
    //
    // So the ridge stays, and the block reports what it actually guarantees instead: the SHIPPED ridge is tried first,
    // so every run whose A is well conditioned gets exactly the decoder it got before, byte for byte; if that
    // candidate raises Q by more than Q's own rounding it is the ridge and not the data that is speaking, and smaller
    // ridges are tried in turn; and if none of them clears that bar the previous decoder is kept and the block takes
    // no step at all. E cannot rise across block (a) by construction rather than by argument.
    static const double RIDGE_LADDER[] = { 1e-9, 1e-12, 0.0 };

    // Q is an unnormalised sum; the round loop's E is that sum over sum(cw) times the weighted site count. Dividing by
    // the same number here puts the test in the loop's own units, so what this block refuses is exactly what the loop
    // would print as a rise.
    double sum_cw = 0.0;
    for (int c = 0; c < nout; c++)
        sum_cw += (double)m.cw[c];
    double den = 0.0;
    for (size_t i = 0; i < planes; i++)
        den += per_site[i] * (double)d->planes[i].w0 * (double)d->planes[i].h0 * (double)(1 + k);
    const double inv_norm = sum_cw * den > 0.0 ? 1.0 / (sum_cw * den) : 0.0;

    // A candidate is refused only when it raises E by more than this, in E's own [0,1] units. It is NOT a tolerance on
    // a real rise: the ridge's own effect on the dot image above is 2.4e-07, five orders of magnitude above it, and the
    // smallest E a real image reaches is 1e-05. It is there because Q is a sum of some hundreds of double products of
    // order one, so its own rounding is a few times 1e-16 in these units, and because a call that re-solves an already
    // converged system returns the same decoder and the same Q to the last bit - a step of exactly zero, which is not a
    // rise and must go on being taken, or the run lands in a different basin for no reason.
    static const double LS_ACCEPT_RISE = 1e-12;

    const double q_prev = had_previous ? qform(x_prev) : 0.0;
    const int cols = mm + nout;
    std::vector<double> g((size_t)mm * cols, 0.0);
    std::vector<double> x_new((size_t)nout * mm, 0.0);
    bool accepted = false, solved_any = false;

    for (size_t step = 0; step < sizeof(RIDGE_LADDER) / sizeof(RIDGE_LADDER[0]) && !accepted; step++)
    {
        const double ridge = RIDGE_LADDER[step] * mean_diag;

        // The triangle mirrored into the augmented system [A + ridge I | rhs^T].
        std::fill(g.begin(), g.end(), 0.0);
        for (int i = 0; i < mm; i++)
        {
            for (int j = i; j < mm; j++)
            {
                const double a = acc[(size_t)tri_index(i, j, mm)];
                g[(size_t)i * cols + j] = a;
                g[(size_t)j * cols + i] = a;
            }
            g[(size_t)i * cols + i] += ridge;
            for (int c = 0; c < nout; c++)
                g[(size_t)i * cols + mm + c] = acc[(size_t)tri + (size_t)c * mm + i];
        }
        if (!gauss_jordan(g, mm, nout))
            continue;
        solved_any = true;

        // Rounded to the float the asset carries BEFORE it is judged: the decoder that ships is the one whose Q the
        // objective will measure, and on an ill-conditioned system the two need not agree in the last places.
        for (int c = 0; c < nout; c++)
            for (int i = 0; i < mm; i++)
                x_new[(size_t)c * mm + i] = (double)(float)g[(size_t)i * cols + mm + c];
        if (!had_previous || (qform(x_new) - q_prev) * inv_norm <= LS_ACCEPT_RISE)
            accepted = true;
    }

    if (accepted)
    {
        m.dec.w.assign((size_t)nout * nin, 0.0f);
        m.dec.b.assign((size_t)nout, 0.0f);
        for (int c = 0; c < nout; c++)
        {
            for (int i = 0; i < nin; i++)
                m.dec.w[(size_t)c * nin + i] = (float)x_new[(size_t)c * mm + i];
            m.dec.b[c] = (float)x_new[(size_t)c * mm + nin];
        }
    }
    else if (!solved_any && !had_previous)
    {
        // A singular system with nothing to fall back on means the features carry no information at all; the best
        // constant fit is the weighted mean target, which is the last row of the right-hand side divided by the
        // weighted site count in A's last entry.
        m.dec.w.assign((size_t)nout * nin, 0.0f);
        m.dec.b.assign((size_t)nout, 0.0f);
        const double count = acc[(size_t)tri_index(nin, nin, mm)];
        for (int c = 0; c < nout; c++)
            m.dec.b[c] = count > 0.0 ? (float)(acc[(size_t)tri + (size_t)c * mm + nin] / count) : 0.0f;
    }
    // Otherwise m.dec is left exactly as the caller had it: no candidate lowered E, so the block takes no step.

    device_upload_decoder(d, m);
    return ms;
}
