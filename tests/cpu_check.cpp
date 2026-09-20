// tests/cpu_check.cpp: the unit exercise of the CPU backend's pool, its model's transfers and its kernels
// (nntc_cpu_check).
//
//     nntc_cpu_check            the contract, the transfers and the ported kernels, at several thread counts; exits 1
//                               naming a failure
//     nntc_cpu_check --bench    the two measurements stage C1 records (docs/CPU_BACKEND_PLAN.md section 6)
//
// WHY A PROGRAM OF ITS OWN. The pool's contract (cpu/pool.h, plan section 3.1) is what makes every CPU result independent
// of the thread count, and the transfers are the instrument the per-kernel harness reads both backends through; both
// are worth proving before a single kernel rests on them, and neither needs CUDA, an image or an encode to prove. So
// this builds from the CPU backend's own sources alone, on every configure - which also makes it the program a
// ThreadSanitizer build runs, threaded, to show that the pool and the transfers have no data race (section 3.4).
//
// What it checks, rule by rule:
//   rule 1 and 2  the partition is a pure function that covers [0, chunks) exactly once, contiguously, for every
//                 chunk count and participant count; every chunk body runs exactly once at every thread count
//   rule 3        a static partition: the chunk a thread is handed is computable, and the same run over the same data
//                 gives the same bits at -j1, -j2, -j3, -j7, -j16, -j32, -j64 and -j0
//   rule 4        a reduction whose summation order visibly matters (a wide dynamic range, where a plain left fold
//                 and a pairwise sum differ) folds to the same bits at every thread count, and to the serial fold in
//                 chunk order computed with no pool at all
//   rule 5        -j1 goes through the same walk: there is no other path to compare it with, so it is held to the
//                 serial reference like every other count
//   more threads than chunks, and zero chunks: every count from 0 to 3 chunks at -j64, repeated, with no deadlock
//   the transfers every seam transfer round-trips bit for bit (negative zero, a subnormal and the float extremes
//                 included), an upload of the wrong size is ignored as the CUDA twin ignores it, level1_range equals
//                 the serial loop, and bytes_for equals device_memory; the same at -j1, -j4 and -j13
//   the kernels   every ported kernel's every output is the same bits at -j1, -j4, -j16 and -j0 on synthetic layouts
//                 of both level-0 modes (block (a)'s normal equations, decoder and rung included, and blocks (b) and
//                 (c')'s stencils, gradients, preconditioners, corrections, sweeps and reports, and block (c)'s
//                 palette search on both branches of its argmin, and the BC pack and its refinement, seeded and
//                 continued), the objective agrees
//                 with its brute-force twin to 1e-9, and block (b) passes the gate's three arms on the smallest plane:
//                 the CG residual under its bar, the dense direct solve agreeing with the CG, and a zero gradient

// export.cpp, which the BC refinement packs through, writes its PNGs with stb_image_write; main.cpp compiles that
// library's implementation for the encoder, and this program has no main.cpp, so it compiles it here in the same way -
// plainly, with nothing suppressed around it, because the header is clean at this program's warning level on every
// compiler and a warning a future version brings with it should be seen rather than hidden.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace nntc_cpu;

namespace
{

int g_failures = 0;

void fail(const std::string& what)
{
    std::printf("FAIL: %s\n", what.c_str());
    g_failures++;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------------------------------------------------

void check_partition()
{
    for (int chunks = 0; chunks <= 70; chunks++)
        for (int np = 1; np <= 40; np++)
        {
            int expect = 0;
            for (int p = 0; p < np; p++)
            {
                int b, e;
                Pool::partition(chunks, np, p, b, e);
                int b2, e2;
                Pool::partition(chunks, np, p, b2, e2);
                if (b != expect || e < b || b2 != b || e2 != e)
                {
                    fail("partition(" + std::to_string(chunks) + ", " + std::to_string(np) + ", " + std::to_string(p) +
                         ") is not contiguous or not pure");
                    return;
                }
                expect = e;
            }
            if (expect != chunks)
            {
                fail("partition over " + std::to_string(np) + " participants does not cover " +
                     std::to_string(chunks) + " chunks");
                return;
            }
        }
    std::printf("  rule 1-3: the static partition covers [0, chunks) once and contiguously for 0..70 chunks over 1..40 "
                "participants\n");
}

void check_visits(int threads)
{
    Pool pool(threads);
    const int counts[] = { 0, 1, 2, 3, 5, 31, 32, 33, 64, 65, 1000 };
    for (int chunks : counts)
    {
        std::vector<int> hits((size_t)chunks, 0);
        pool.run(chunks, [&](int c) { hits[(size_t)c]++; });
        for (int c = 0; c < chunks; c++)
            if (hits[(size_t)c] != 1)
            {
                fail("-j" + std::to_string(threads) + ": chunk " + std::to_string(c) + " of " +
                     std::to_string(chunks) + " ran " + std::to_string(hits[(size_t)c]) + " times");
                return;
            }
    }
}

// Values across twenty decimal orders of magnitude with both signs, where the order of a sum shows in its last bits.
std::vector<double> wide_values(size_t n)
{
    std::vector<double> v(n);
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++)
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double mant = (double)(s >> 11) / 9007199254740992.0;
        const int expo = (int)((s >> 3) % 40) - 20;
        v[i] = ((s & 1) ? -1.0 : 1.0) * mant * std::pow(10.0, (double)expo);
    }
    return v;
}

double chunked_sum(Pool& pool, const std::vector<double>& v)
{
    const size_t n = v.size();
    return fold_chunks(
        pool, chunks_for(n, CPU_CHUNK), 0.0,
        [&](int c) {
            double s = 0.0;
            const size_t b = (size_t)c * CPU_CHUNK, e = std::min(n, b + CPU_CHUNK);
            for (size_t i = b; i < e; i++)
                s += v[i];
            return s;
        },
        [](double a, double b) { return a + b; });
}

void check_reduction()
{
    const std::vector<double> v = wide_values(1000003);
    // The reference: the same chunk sums folded in chunk order, with no pool at all.
    double reference = 0.0;
    for (size_t b = 0; b < v.size(); b += CPU_CHUNK)
    {
        double s = 0.0;
        for (size_t i = b; i < std::min(v.size(), b + CPU_CHUNK); i++)
            s += v[i];
        reference += s;
    }
    double plain = 0.0;
    for (double x : v)
        plain += x;
    const int counts[] = { 1, 2, 3, 7, 16, 32, 64, 0 };
    for (int t : counts)
    {
        Pool pool(t);
        for (int rep = 0; rep < 3; rep++)
        {
            const double s = chunked_sum(pool, v);
            if (std::memcmp(&s, &reference, sizeof(double)) != 0)
            {
                fail("-j" + std::to_string(t) + ": the chunked sum is not the serial chunk-order fold");
                return;
            }
        }
    }
    std::printf("  rule 4-5: a wide-range sum of %zu values folds to the same bits at -j1, 2, 3, 7, 16, 32, 64 and 0, "
                "equal to the serial chunk-order fold (a plain left fold %s)\n", v.size(),
                std::memcmp(&plain, &reference, sizeof(double)) != 0 ? "differs, so order is visible here"
                                                                     : "happens to agree on this data");
}

void check_more_threads_than_chunks()
{
    Pool pool(64);
    for (int rep = 0; rep < 2000; rep++)
        for (int chunks = 0; chunks <= 3; chunks++)
        {
            std::vector<int> hits((size_t)chunks, 0);
            pool.run(chunks, [&](int c) { hits[(size_t)c] = c + 1; });
            for (int c = 0; c < chunks; c++)
                if (hits[(size_t)c] != c + 1)
                {
                    fail("-j64 with " + std::to_string(chunks) + " chunks missed chunk " + std::to_string(c));
                    return;
                }
        }
    std::printf("  more threads than chunks: 0 to 3 chunks at -j64, 2000 times each, every chunk once and no hang\n");
}

void check_resolution()
{
    unsigned hw = std::thread::hardware_concurrency();
    const int expect = hw == 0 ? 1 : (hw > (unsigned)POOL_MAX_THREADS ? POOL_MAX_THREADS : (int)hw);
    if (pool_resolve_threads(0) != expect || Pool(0).threads() != expect)
        fail("-j0 does not resolve to hardware_concurrency() clamped into 1..1024");
    if (pool_resolve_threads(1) != 1 || pool_resolve_threads(5000) != POOL_MAX_THREADS ||
        pool_resolve_threads(-3) != expect)
        fail("-j resolution does not clamp");
    std::printf("  -j0 resolves to %d (this machine reports %u)\n", expect, hw);
}

// Many short runs back to back, which is where a lost wake-up or a stale job slot would show - and what gives
// ThreadSanitizer the most hand-offs to look at.
void check_churn()
{
    Pool pool(8);
    std::vector<long long> slot(8, 0);
    for (int rep = 0; rep < 20000; rep++)
    {
        const int chunks = 1 + rep % 8;
        pool.run(chunks, [&](int c) { slot[(size_t)c] += c + 1; });
    }
    long long expect[8] = {};
    for (int rep = 0; rep < 20000; rep++)
        for (int c = 0; c < 1 + rep % 8; c++)
            expect[c] += c + 1;
    for (int c = 0; c < 8; c++)
        if (slot[(size_t)c] != expect[c])
            fail("20000 back-to-back runs at -j8 lost a chunk");
    std::printf("  20000 back-to-back runs at -j8 with 1..8 chunks: every chunk accounted for\n");

    // The other side of the hand-off: runs spaced wider than the spin budget, so every worker has parked and must be
    // woken, and runs whose last chunk outlasts the budget, so the caller parks and must be woken by the last worker.
    std::vector<long long> parked(8, 0);
    for (int rep = 0; rep < 60; rep++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        pool.run(8, [&](int c) {
            if (rep % 2 == 1 && c == 7)
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            parked[(size_t)c] += 1;
        });
    }
    for (int c = 0; c < 8; c++)
        if (parked[(size_t)c] != 60)
            fail("a run after the pool had parked lost a chunk");
    std::printf("  60 runs after the workers parked, half of them with the caller parked too: every chunk accounted "
                "for\n");
}

// ---------------------------------------------------------------------------------------------------------------------
// The transfers
// ---------------------------------------------------------------------------------------------------------------------

float pattern_value(size_t i)
{
    switch (i % 11)
    {
    case 0:
        return -0.0f;
    case 1:
        return std::numeric_limits<float>::denorm_min();
    case 2:
        return std::numeric_limits<float>::max();
    case 3:
        return -std::numeric_limits<float>::max();
    default:
        return (float)((double)((i * 2654435761ull) % 100003ull) / 50001.5 - 1.0);
    }
}

// A three-plane chain under --l0 bc8 with a residual buffer, wide enough that every copy spans several chunks, and a
// deep plane of one level-1 texel.
Model synthetic_model()
{
    Model m;
    m.textures = 2;
    m.nout = 6;
    m.c0 = 3;
    m.bits0 = { 8, 8, 8 };
    m.c1 = 4;
    m.bits1 = 5;
    m.l0_bc8 = true;
    m.init0_residual = true;
    const int dims[3][4] = { { 300, 212, 75, 53 }, { 150, 106, 38, 27 }, { 4, 3, 1, 1 } };
    for (const auto& d : dims)
    {
        PlaneSize p;
        p.w0 = d[0];
        p.h0 = d[1];
        p.w1 = d[2];
        p.h1 = d[3];
        m.planes.push_back(p);
    }
    m.width = 300;
    m.height = 212;
    m.dec.nin = feature_count(m.c0, m.c1);
    m.dec.nout = m.nout;
    m.dec.w.assign((size_t)m.dec.nin * m.nout, 0.0f);
    m.dec.b.assign((size_t)m.nout, 0.0f);
    for (size_t i = 0; i < m.dec.w.size(); i++)
        m.dec.w[i] = pattern_value(i + 4);
    for (size_t i = 0; i < m.dec.b.size(); i++)
        m.dec.b[i] = pattern_value(i + 40);
    m.cw.assign((size_t)m.nout, 1.0f);
    m.lo0 = { -1.0f, -0.5f, 0.0f };
    m.hi0 = { 1.0f, 0.5f, 2.0f };
    m.lo1 = { -1.0f, -2.0f, -3.0f, -4.0f };
    m.hi1 = { 1.0f, 2.0f, 3.0f, 4.0f };
    return m;
}

std::vector<Image> synthetic_chain(const Model& m)
{
    std::vector<Image> chain(m.planes.size());
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        chain[i].w = m.planes[i].w0;
        chain[i].h = m.planes[i].h0;
        chain[i].nc = m.nout;
        chain[i].v.resize((size_t)chain[i].w * chain[i].h * m.nout);
        for (size_t j = 0; j < chain[i].v.size(); j++)
            chain[i].v[j] = (float)((j * 7 + i) % 256) / 255.0f;
    }
    return chain;
}

// Everything a transfer can read back, for comparing one thread count's run with another's.
struct Readback
{
    std::vector<std::vector<float>> v0, v1, lo, hi;
    std::vector<std::vector<uint8_t>> k0, k1, ep, sel;
    std::vector<std::vector<double>> delta;
};

Readback exercise_transfers(int threads)
{
    const Model m = synthetic_model();
    const std::vector<Image> chain = synthetic_chain(m);
    CpuModel* d = device_create(m, chain, threads);
    const std::string j = "-j" + std::to_string(threads) + ": ";

    if (device_memory(d) != bytes_for(m))
        fail(j + "device_memory " + std::to_string(device_memory(d)) + " is not bytes_for " +
             std::to_string(bytes_for(m)));
    for (size_t i = 0; i < m.planes.size(); i++)
        if (!same_bits(d->planes[i].src, chain[i].v))
            fail(j + "the source chain did not land bit for bit");

    Readback r;
    const size_t np = m.planes.size();
    r.v0.resize(np);
    r.v1.resize(np);
    r.k0.resize(np);
    r.k1.resize(np);
    r.lo.resize(np);
    r.hi.resize(np);
    r.delta.resize(np);
    for (size_t i = 0; i < np; i++)
    {
        const int p = (int)i;
        const size_t n0 = (size_t)m.planes[i].w0 * m.planes[i].h0 * m.c0;
        const size_t n1 = (size_t)m.planes[i].w1 * m.planes[i].h1 * m.c1;
        std::vector<float> a(n0), b(n1), out;
        for (size_t k = 0; k < n0; k++)
            a[k] = pattern_value(k + i);
        for (size_t k = 0; k < n1; k++)
            b[k] = pattern_value(k + 7 * i + 3);

        level0_upload(d, m, p, a);
        level0_download_values(d, m, p, out);
        if (!same_bits(out, a))
            fail(j + "level-0 values do not round-trip on plane " + std::to_string(i));
        std::vector<float> wrong(n0 + 1, 9.0f);
        level0_upload(d, m, p, wrong);
        level0_download_values(d, m, p, out);
        if (!same_bits(out, a))
            fail(j + "a level-0 upload of the wrong size was not ignored");
        r.v0[i] = out;

        level1_upload(d, m, p, b);
        level1_download(d, m, p, out);
        if (!same_bits(out, b))
            fail(j + "level-1 values do not round-trip on plane " + std::to_string(i));
        level1_poke(d, p, n1 / 2, 0.75f);
        b[n1 / 2] = 0.75f;
        level1_download(d, m, p, out);
        if (!same_bits(out, b))
            fail(j + "level1_poke did not write exactly one value");

        // The range against the serial loop solve_level1.cu runs on the host.
        std::vector<float> lo, hi, slo((size_t)m.c1, 1e30f), shi((size_t)m.c1, -1e30f);
        level1_range(d, m, p, lo, hi);
        for (size_t k = 0; k < b.size(); k++)
        {
            const size_t c = k % (size_t)m.c1;
            slo[c] = b[k] < slo[c] ? b[k] : slo[c];
            shi[c] = b[k] > shi[c] ? b[k] : shi[c];
        }
        if (!same_bits(lo, slo) || !same_bits(hi, shi))
            fail(j + "level1_range is not the serial loop's answer on plane " + std::to_string(i));
        r.v1[i] = out;
        r.lo[i] = lo;
        r.hi[i] = hi;

        std::vector<uint8_t> ka(n0), kb(n1), kout;
        for (size_t k = 0; k < n0; k++)
            ka[k] = (uint8_t)(k * 37 + i);
        for (size_t k = 0; k < n1; k++)
            kb[k] = (uint8_t)(k * 101 + 3 * i);
        level0_upload_indices(d, m, p, ka);
        level0_download(d, m, p, kout);
        if (!same_bits(kout, ka))
            fail(j + "level-0 indices do not round-trip");
        r.k0[i] = kout;
        level1_upload_indices(d, m, p, kb);
        level1_download_indices(d, m, p, kout);
        if (!same_bits(kout, kb))
            fail(j + "level-1 indices do not round-trip");
        r.k1[i] = kout;

        for (size_t k = 0; k < d->planes[i].cg_x.size(); k++)
            d->planes[i].cg_x[k] = (double)pattern_value(k + 11) * 1e-3;
        level1_delta(d, m, p, r.delta[i]);
        if (!same_bits(r.delta[i], d->planes[i].cg_x))
            fail(j + "level1_delta is not the correction");
    }

    std::vector<std::vector<uint8_t>> ep, sel, ep2, sel2;
    bc_state_save(d, m, ep, sel);
    for (size_t i = 0; i < ep.size(); i++)
    {
        if (ep[i].size() != plane_blocks(m, (int)i) * (size_t)m.c0 * 2 || sel[i].size() != ep[i].size() * 8)
            fail(j + "the BC state is not sized by the plane's blocks");
        for (size_t k = 0; k < ep[i].size(); k++)
            ep[i][k] = (uint8_t)(k * 13 + i);
        for (size_t k = 0; k < sel[i].size(); k++)
            sel[i][k] = (uint8_t)(k * 29 + i);
    }
    bc_state_restore(d, m, ep, sel);
    bc_state_save(d, m, ep2, sel2);
    if (ep2 != ep || sel2 != sel)
        fail(j + "the BC state does not round-trip");
    r.ep = ep2;
    r.sel = sel2;

    upload_decoder_to_device(d, m);
    if (!same_bits(d->weights, m.dec.w) || !same_bits(d->bias, m.dec.b))
        fail(j + "the decoder did not land bit for bit");
    Model g = m;
    build_level0_palette({ 3, 2, 4 }, 3, true, g.palette0);
    upload_grids_to_device(d, g);
    for (int c = 0; c < 3; c++)
        for (size_t k = 0; k < 16; k++)
        {
            const float want = k < g.palette0[(size_t)c].size() ? g.palette0[(size_t)c][k] : 0.0f;
            if (std::memcmp(&d->palette[(size_t)c * 16 + k], &want, sizeof(float)) != 0)
                fail(j + "the palette is not device.cuh's padded layout");
        }
    for (int c = 0; c < 4; c++)
        if (d->lo1[(size_t)c] != g.lo1[(size_t)c] || d->hi1[(size_t)c] != g.hi1[(size_t)c] ||
            (c < 3 && (d->lo0[(size_t)c] != g.lo0[(size_t)c] || d->hi0[(size_t)c] != g.hi0[(size_t)c])))
            fail(j + "a grid did not land");

    device_destroy(d);
    return r;
}

void check_transfers()
{
    const Readback one = exercise_transfers(1);
    const int counts[] = { 4, 13 };
    for (int t : counts)
    {
        const Readback r = exercise_transfers(t);
        if (r.v0 != one.v0 || r.v1 != one.v1 || r.k0 != one.k0 || r.k1 != one.k1 || r.ep != one.ep ||
            r.sel != one.sel || r.delta != one.delta || r.lo != one.lo || r.hi != one.hi)
            fail("-j" + std::to_string(t) + ": the transfers read back differently from -j1");
    }
    const Model m = synthetic_model();
    std::printf("  transfers: every one round-trips bit for bit on a %zu-plane --l0 bc8 chain (%.2f MB), the same at "
                "-j1, -j4 and -j13\n", m.planes.size(), (double)bytes_for(m) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------------------------------------------------
// The kernels: the same bits at every thread count
// ---------------------------------------------------------------------------------------------------------------------
//
// Every ported kernel, run on the same synthetic model at -j1, -j4, -j16 and -j0, and every output it produces written
// into a log of raw bytes; the logs must be identical, which is decision 9's second half (docs/CPU_BACKEND_PLAN.md
// section 7.3) proved kernel by kernel rather than inferred from a whole encode. The synthetic data is smooth where an
// image is smooth and noisy where one is not, spans several chunks on the base plane and one chunk on the deep plane,
// and every value is in the range an encode actually holds, so the arithmetic is the arithmetic an encode does.
//
// The objective is also held to its own brute-force twin here, to the 1e-9 the gate's objective_checks holds the CUDA
// pair to (stage C2's acceptance), at every thread count.

// A deterministic value in [0, 1) from a counter: a 64-bit LCG step and its top 24 bits.
float unit_hash(unsigned long long i)
{
    const unsigned long long s = i * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((s >> 40) & 0xFFFFFFull) / 16777216.0f;
}

struct KernelCase
{
    const char* name;
    int textures, c0, c1, bits1, k;
    bool bc8;
    bool init0_chain, init0_texture;   // the residual seed's scope and policy
};

// A three-plane chain whose base spans several CPU_CHUNKs of texels and sites (and two of the histogram's larger
// chunks), and whose deepest plane is one chunk.
Model kernel_model(const KernelCase& kc)
{
    Model m;
    m.textures = kc.textures;
    m.nout = 3 * kc.textures;
    m.c0 = kc.c0;
    m.c1 = kc.c1;
    m.bits0.assign((size_t)kc.c0, kc.bc8 ? 8 : 4);
    m.bits1 = kc.bits1;
    m.l0_bc8 = kc.bc8;
    m.init0_residual = true;
    m.init0_chain = kc.init0_chain;
    m.init0_texture = kc.init0_texture;
    const int dims[3][4] = { { 300, 230, 75, 58 }, { 150, 115, 38, 29 }, { 9, 7, 3, 2 } };
    for (const auto& dm : dims)
    {
        PlaneSize p;
        p.w0 = dm[0];
        p.h0 = dm[1];
        p.w1 = dm[2];
        p.h1 = dm[3];
        m.planes.push_back(p);
    }
    m.width = 300;
    m.height = 230;
    m.dec.nin = feature_count(m.c0, m.c1);
    m.dec.nout = m.nout;
    m.dec.w.resize((size_t)m.dec.nin * m.nout);
    m.dec.b.resize((size_t)m.nout);
    for (size_t i = 0; i < m.dec.w.size(); i++)
        m.dec.w[i] = 0.6f * unit_hash(1000 + i) - 0.3f;
    for (size_t i = 0; i < m.dec.b.size(); i++)
        m.dec.b[i] = 0.5f * unit_hash(2000 + i) + 0.25f;
    m.cw.resize((size_t)m.nout);
    for (int c = 0; c < m.nout; c++)
        m.cw[(size_t)c] = c % 3 == 1 ? 1.5f : 0.75f;
    if (!kc.bc8)
        build_level0_palette(m.bits0, m.c0, false, m.palette0);
    return m;
}

std::vector<Image> kernel_chain(const Model& m)
{
    std::vector<Image> chain(m.planes.size());
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        Image& im = chain[i];
        im.w = m.planes[i].w0;
        im.h = m.planes[i].h0;
        im.nc = m.nout;
        im.v.resize((size_t)im.w * im.h * m.nout);
        for (int y = 0; y < im.h; y++)
            for (int x = 0; x < im.w; x++)
                for (int c = 0; c < m.nout; c++)
                {
                    const double smooth = 0.5 + 0.3 * std::sin(0.11 * (x + 3 * c) * (1.0 + (double)i)) *
                                                    std::cos(0.07 * (y - 2 * c) * (1.0 + (double)i));
                    const double noise = 0.15 * ((double)unit_hash(((i * 977 + (size_t)y) * 1009 + (size_t)x) * 31 +
                                                                   (size_t)c) - 0.5);
                    const double v = smooth + noise;
                    im.v[((size_t)y * im.w + x) * m.nout + c] = (float)(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
                }
    }
    return chain;
}

// The latents a solve might have left: smooth in [-1, 1] with a little noise.
void kernel_latents(CpuModel* d, const Model& m)
{
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        std::vector<float> v0((size_t)m.planes[i].w0 * m.planes[i].h0 * m.c0);
        std::vector<float> v1((size_t)m.planes[i].w1 * m.planes[i].h1 * m.c1);
        for (size_t t = 0; t < v0.size(); t++)
            v0[t] = (float)(0.8 * std::sin(0.013 * (double)t + (double)i)) + 0.2f * unit_hash(3000 + t) - 0.1f;
        for (size_t t = 0; t < v1.size(); t++)
            v1[t] = (float)(0.9 * std::cos(0.029 * (double)t - (double)i)) + 0.1f * unit_hash(7000 + t) - 0.05f;
        level0_upload(d, m, (int)i, v0);
        level1_upload(d, m, (int)i, v1);
    }
    upload_decoder_to_device(d, m);
    upload_grids_to_device(d, m);
}

// A log of raw output bytes, one named section per output, so that a difference between two thread counts is named.
struct ByteLog
{
    std::vector<std::string> names;
    std::vector<std::vector<unsigned char>> data;

    template <class T>
    void add(const std::string& name, const T* p, size_t n)
    {
        names.push_back(name);
        const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
        data.emplace_back(b, b + n * sizeof(T));
    }
    template <class T>
    void add(const std::string& name, const std::vector<T>& v)
    {
        add(name, v.data(), v.size());
    }
    void add(const std::string& name, double x) { add(name, &x, 1); }
};

void log_objective(ByteLog& log, const std::string& what, const Objective& r)
{
    log.add(what + " E", r.e);
    log.add(what + " centre mse", r.centre_mse);
    log.add(what + " sampled mse", r.sampled_mse);
    log.add(what + " per-plane E", r.e_plane);
    log.add(what + " per-plane centre mse", r.centre_plane);
}

double objective_gap(const Objective& a, const Objective& b)
{
    auto rel = [](double x, double y) {
        const double s = std::max(std::fabs(x), std::fabs(y));
        return s > 0.0 ? std::fabs(x - y) / s : 0.0;
    };
    double worst = std::max(rel(a.e, b.e), std::max(rel(a.centre_mse, b.centre_mse), rel(a.sampled_mse, b.sampled_mse)));
    for (size_t i = 0; i < a.e_plane.size() && i < b.e_plane.size(); i++)
        worst = std::max(worst, std::max(rel(a.e_plane[i], b.e_plane[i]), rel(a.centre_plane[i], b.centre_plane[i])));
    return worst;
}

double g_objective_twin_worst = 0.0;

// Stage C2: the objective, its twin and the decode.
void run_objective_kernels(CpuModel* d, const Model& m, const KernelCase& kc, int threads, ByteLog& log)
{
    std::vector<double> per_site(m.planes.size());
    for (size_t i = 0; i < per_site.size(); i++)
        per_site[i] = 1.0 / (double)(1u << (2 * i));
    Objective r, twin;
    objective_eval(d, m, kc.k, per_site, r);
    objective_check_host(d, m, kc.k, per_site, twin);
    log_objective(log, "objective", r);
    log_objective(log, "twin", twin);
    const double gap = objective_gap(r, twin);
    g_objective_twin_worst = std::max(g_objective_twin_worst, gap);
    if (!(gap <= 1e-9))
        fail(std::string(kc.name) + " -j" + std::to_string(threads) + ": the objective and its brute-force twin differ by " +
             std::to_string(gap) + " relative, past 1e-9");
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        std::vector<uint8_t> rgb8;
        decode_plane(d, m, (int)i, rgb8);
        log.add("decode M" + std::to_string(i), rgb8);
    }
}

// Both latents' planes and indices, and the host Model's copies of the indices and the grids.
void log_state(ByteLog& log, CpuModel* d, const Model& m, const std::string& what)
{
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const std::string p = what + " M" + std::to_string(i);
        log.add(p + " v0", d->planes[i].v0);
        log.add(p + " k0", d->planes[i].k0);
        log.add(p + " v1", d->planes[i].v1);
        log.add(p + " k1", d->planes[i].k1);
    }
    for (size_t i = 0; i < m.k0.size(); i++)
        log.add(what + " host k0 M" + std::to_string(i), m.k0[i]);
    for (size_t i = 0; i < m.k1.size(); i++)
        log.add(what + " host k1 M" + std::to_string(i), m.k1[i]);
    log.add(what + " lo1", m.lo1);
    log.add(what + " hi1", m.hi1);
    log.add(what + " lo0", m.lo0);
    log.add(what + " hi0", m.hi0);
}

// Stage C3: both level-0 inits, both level-1 inits, the residual seed of every channel, and the five grid entry points,
// in the order an encode reaches them.
void run_init_kernels(CpuModel* d, Model& m, const KernelCase& kc, ByteLog& log)
{
    const float luma[3] = { 0.2126f, 0.7152f, 0.0722f };
    init_level0(d, m, luma, true);
    log_state(log, d, m, "init_level0 luma");
    init_level0(d, m, luma, false);
    log_state(log, d, m, "init_level0 zero");

    Level1InitReport box, pca;
    init_level1(d, m, false, box);
    log_state(log, d, m, "init_level1 box");
    init_level1(d, m, true, pca);
    log_state(log, d, m, "init_level1 pca");
    log.add("pca components", (double)pca.components);
    log.add("pca eigenvalues", pca.eigenvalue, 4);
    log.add("pca peaks", pca.peak, 4);
    log.add("pca total variance", pca.total_variance);

    std::vector<double> plane_weight(m.planes.size());
    for (size_t i = 0; i < plane_weight.size(); i++)
        plane_weight[i] = 1.0 / (double)(1u << (2 * i));
    for (int c = 0; c < m.c0; c++)
    {
        Init0ChannelReport r;
        init0_residual_channel(d, m, c, plane_weight, r);
        const std::string ch = "seed channel " + std::to_string(c);
        log_state(log, d, m, ch);
        log.add(ch + " eigenvalue", r.eigenvalue);
        log.add(ch + " total variance", r.total_variance);
        log.add(ch + " share", r.share);
        log.add(ch + " peak", r.peak);
        log.add(ch + " divisor", r.divisor);
        log.add(ch + " direction", r.dir, (size_t)r.nout);
        log.add(ch + " texture", (double)r.texture);
    }

    quantise_level1(d, m, "pct");
    log_state(log, d, m, "quantise_level1 pct");
    freeze_level1_grid(d, m, "minmax");
    log_state(log, d, m, "freeze_level1_grid minmax");
    if (kc.bc8)
    {
        freeze_level0_grid(d, m, "pct");
        log_state(log, d, m, "freeze_level0_grid pct");
        // Move the plane off its grid, then snap it back with the grid held.
        for (size_t i = 0; i < m.planes.size(); i++)
        {
            std::vector<float> v0 = d->planes[i].v0;
            for (size_t t = 0; t < v0.size(); t++)
                v0[t] += 0.01f * (unit_hash(9000 + t) - 0.5f);
            level0_upload(d, m, (int)i, v0);
        }
        snap_level0_on_grid(d, m);
        log_state(log, d, m, "snap_level0_on_grid");
        quantise_level0(d, m, "minmax");
        log_state(log, d, m, "quantise_level0 minmax");
    }
}

// Stage C4: block (a) twice - once with no decoder to fall back on (the first call of an encode, which accepts its
// first finite rung) and once from the decoder that call left (every later call, which the ladder measures against) -
// logging the normal equations, the decoder and the rung each call took. The rung is read as the tally's change over
// the call, because the tally itself counts every call this process has made, at every thread count.
void run_decoder_kernels(CpuModel* d, Model& m, const KernelCase& kc, ByteLog& log)
{
    std::vector<double> per_site(m.planes.size());
    for (size_t i = 0; i < per_site.size(); i++)
        per_site[i] = 1.0 / (double)(1u << (2 * i));
    m.dec.w.clear();
    m.dec.b.clear();
    for (int call = 0; call < 2; call++)
    {
        long long before[4] = {}, after[4] = {};
        nntc_cpu::decoder_ridge_tally(before);   // qualified: model.h declares the CUDA arm's global of that name
        solve_decoder(d, m, kc.k, per_site);
        nntc_cpu::decoder_ridge_tally(after);
        const std::string what = "block (a) call " + std::to_string(call);
        log.add(what + " normal equations", d->reduction);
        log.add(what + " weights", m.dec.w);
        log.add(what + " bias", m.dec.b);
        log.add(what + " model weights", d->weights);
        for (int r = 0; r < 4; r++)
            log.add(what + " rung " + std::to_string(r), (double)(after[r] - before[r]));
    }
}

// One block's per-plane reports, every field but the clocks.
void log_level1_reports(ByteLog& log, const std::string& what, const std::vector<Level1Report>& rep)
{
    for (size_t i = 0; i < rep.size(); i++)
    {
        const Level1Report& r = rep[i];
        const std::string p = what + " M" + std::to_string(i);
        const double fields[] = { (double)r.iterations, r.residual, r.lambda, r.min_diag, r.max_diag,
                                  (double)r.moved_values, r.delta2, r.plane2, (double)r.quantised, (double)r.sweeps,
                                  (double)r.moved };
        log.add(p + " report", fields, sizeof(fields) / sizeof(fields[0]));
        log.add(p + " report range", r.lo, 4);
        log.add(p + " report range hi", r.hi, 4);
    }
}

// One latent's sparse least-squares workspace on every plane, after a block over it.
void log_workspace(ByteLog& log, CpuModel* d, const Model& m, bool level0, const std::string& what)
{
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const PlaneWork w = plane_work(d, m, d->planes[i], level0);
        const std::string p = what + " M" + std::to_string(i);
        const size_t n = w.values(), cc = (size_t)w.c * (size_t)w.c;
        log.add(p + " stencil", w.stencil, w.texels() * STENCIL_BLOCKS * cc);
        log.add(p + " gradient", w.grad, n);
        log.add(p + " preconditioner", w.precond, w.texels() * cc);
        log.add(p + " correction", w.cg_x, n);
        log.add(p + " sweeps' origin", w.prev, n);
        log.add(p + " scalars", d->planes[i].scalars);
        log.add(p + " counters", d->planes[i].counters);
    }
    log.add(what + " monomial matrices", level0 ? d->mono0 : d->mono);
}

double g_level1_resid_worst = 0.0, g_level1_dense_worst = 0.0, g_level1_fd_worst = 0.0;
int g_level1_iterations = 0;

// The gate's three block (b) arms (tests/run_checks.py's objective_checks), on the deepest plane of the synthetic chain,
// which is small enough for the dense twin: the CG residual under its 1e-10 stop, the dense direct solve of the same
// system near the CG's answer, and the finite-difference gradient of the plane's E at the answer under 1e-5 - main.cpp's
// probe, the same fixed sequence of values and the realised float step.
//
// The dense arm's bar here is DENSE_BAR, not the gate's 1e-6, and the difference is measured, not assumed. The dense twin
// forms the gradient in double; the assembly (solve_level1.cu's, which this backend copies) forms each site's residual in
// fp32, and on one of these layouts - a random decoder, a plane of six texels whose correction is small - that rounding
// alone puts the two answers 3.2e-6 of the range apart. Driving the CG to a residual of 4e-16 leaves the gap where it is,
// so it is not the iteration's stop; forming the residual in double in a scratch copy of the assembly brings it to
// 3.2e-7, so it is the fp32 residual, and it is the CUDA kernel's as much as this backend's (the per-kernel harness holds
// the two assemblies bit-identical). On tests/tiny.png the same arm measures 1.5e-7, under the gate's bar.
const double DENSE_BAR = 1e-5;
void check_block_b_arms(CpuModel* d, const Model& m, const KernelCase& kc, int threads, int plane,
                        const std::vector<float>& c_prev, const Level1Report& rep)
{
    const std::string j = std::string(kc.name) + " -j" + std::to_string(threads) + ": plane " + std::to_string(plane);
    g_level1_resid_worst = std::max(g_level1_resid_worst, rep.residual);
    g_level1_iterations = std::max(g_level1_iterations, rep.iterations);
    if (!(rep.residual < 1e-10) && rep.iterations < 200)
        fail(j + ": the CG stopped at a relative residual of " + std::to_string(rep.residual));

    std::vector<double> dense, cg;
    level1_delta(d, m, plane, cg);
    if (!solve_level1_dense_host(d, m, kc.k, plane, rep.lambda, c_prev, dense, 1024))
        fail(j + ": the dense twin refused a plane it should solve");
    double worst = 0.0, lo = 1e300, hi = -1e300;
    for (size_t i = 0; i < dense.size() && i < cg.size(); i++)
    {
        lo = std::min(lo, dense[i]);
        hi = std::max(hi, dense[i]);
        worst = std::max(worst, std::fabs(dense[i] - cg[i]));
    }
    const double rel = hi > lo ? worst / (hi - lo) : 0.0;
    g_level1_dense_worst = std::max(g_level1_dense_worst, rel);
    if (!(rel < DENSE_BAR))
        fail(j + ": the dense direct solve differs from the CG by " + std::to_string(rel) + " of the range");

    const std::vector<double> per_site(m.planes.size(), 1.0);
    Objective base, up, down;
    objective_eval(d, m, kc.k, per_site, base);
    const double scale = base.e_plane[(size_t)plane];
    std::vector<float> cur;
    level1_download(d, m, plane, cur);
    unsigned long long state = 0x9E3779B97F4A7C15ull;
    const double h = 1e-3;
    double fd = 0.0;
    for (int i = 0; i < 20 && scale > 0.0; i++)
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const size_t index = (size_t)((state >> 16) % (unsigned long long)cur.size());
        const float b = cur[index];
        const float plus = (float)((double)b + h), minus = (float)((double)b - h);
        const double step = (double)plus - (double)minus;
        level1_poke(d, plane, index, plus);
        objective_eval(d, m, kc.k, per_site, up);
        level1_poke(d, plane, index, minus);
        objective_eval(d, m, kc.k, per_site, down);
        level1_poke(d, plane, index, b);
        if (step > 0.0)
            fd = std::max(fd, std::fabs((up.e_plane[(size_t)plane] - down.e_plane[(size_t)plane]) / step) * h / scale);
    }
    g_level1_fd_worst = std::max(g_level1_fd_worst, fd);
    if (!(fd < 1e-5))
        fail(j + ": the finite-difference gradient at the answer is " + std::to_string(fd));
}

// Stage C5: block (b) continuous and then on its grid, and under --l0 bc8 block (c') both ways and the BC
// refinement's assembly, each logged in full; the gate's block (b) arms on the deepest plane after the continuous solve.
void run_level1_kernels(CpuModel* d, Model& m, const KernelCase& kc, int threads, ByteLog& log)
{
    const double ridge = 1e-4;
    const int deep = (int)m.planes.size() - 1;
    const std::vector<float> c_prev = d->planes[(size_t)deep].v1;
    std::vector<Level1Report> rep;
    solve_level1_all(d, m, kc.k, ridge, rep);
    log_level1_reports(log, "solve_level1_all", rep);
    log_workspace(log, d, m, false, "solve_level1_all");
    log_state(log, d, m, "solve_level1_all");
    check_block_b_arms(d, m, kc, threads, deep, c_prev, rep[(size_t)deep]);

    sweep_level1_all(d, m, kc.k, ridge, 4, rep);
    log_level1_reports(log, "sweep_level1_all", rep);
    log_workspace(log, d, m, false, "sweep_level1_all");
    log_state(log, d, m, "sweep_level1_all");

    if (kc.bc8)
    {
        solve_level0_cont_all(d, m, kc.k, ridge, rep);
        log_level1_reports(log, "solve_level0_cont_all", rep);
        log_workspace(log, d, m, true, "solve_level0_cont_all");
        log_state(log, d, m, "solve_level0_cont_all");
        sweep_level0_cont_all(d, m, kc.k, ridge, 4, rep);
        log_level1_reports(log, "sweep_level0_cont_all", rep);
        log_workspace(log, d, m, true, "sweep_level0_cont_all");
        log_state(log, d, m, "sweep_level0_cont_all");
        assemble_level0_for_refine(d, m, kc.k);
        log_workspace(log, d, m, true, "assemble_level0_for_refine");
    }
}

// Stage C6: under --l0 palette, block (c) twice - the second from the planes the first left - each call's reports and
// both latents logged in full. The layouts' bits put one palette case on each branch of the argmin: C0 2 at 4 bits is
// 8 bits in all and enumerates jointly, C0 3 at 4 bits is 12 and sweeps.
void run_level0_kernels(CpuModel* d, Model& m, const KernelCase& kc, ByteLog& log)
{
    if (kc.bc8)
        return;
    for (int call = 0; call < 2; call++)
    {
        std::vector<Level0Report> rep;
        solve_level0_all(d, m, kc.k, 2, rep);
        const std::string what = "solve_level0_all call " + std::to_string(call);
        for (size_t i = 0; i < rep.size(); i++)
        {
            const Level0Report& r = rep[i];
            const double fields[] = { (double)r.moved[0], (double)r.moved[1], (double)r.moved[2], (double)r.moved[3],
                                      (double)r.moved_total, (double)r.joint, (double)r.states };
            log.add(what + " M" + std::to_string(i) + " report", fields, sizeof(fields) / sizeof(fields[0]));
        }
        log_state(log, d, m, what);
    }
}

// The BC refinement's state, the plane it decodes to and the Model's blocks, after one of its calls.
void log_bc(ByteLog& log, CpuModel* d, const Model& m, const std::string& what, double pack_psnr, size_t pack_texels)
{
    log.add(what + " packing psnr", pack_psnr);
    log.add(what + " packing texels", (double)pack_texels);
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const std::string p = what + " M" + std::to_string(i);
        log.add(p + " endpoints", d->planes[i].bc_ep);
        log.add(p + " selectors", d->planes[i].bc_sel);
        log.add(p + " correction", d->planes[i].cg0_x);
        for (size_t f = 0; f < m.bc0_blocks[i].size(); f++)
            log.add(p + " file " + std::to_string(f), m.bc0_blocks[i][f]);
    }
    log_state(log, d, m, what);
}

double g_bc_decrease = 0.0;
long long g_bc_improved = 0;

// Stage C7: under --l0 bc8, the seed pack of the plane block (c') left, two refinement passes, then the pack continued
// against the same level 1 and decoder and one more pass - main.cpp's post-fit step and its --bc-refine-after - each
// call's figures and state logged in full. Every pass must lower the block objective or leave it (its accepted decrease
// is never positive), which is the refinement's own acceptance rule observed from outside.
void run_bc_kernels(CpuModel* d, Model& m, const KernelCase& kc, int threads, ByteLog& log)
{
    if (!kc.bc8)
        return;
    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < m.planes.size(); i++)
        level0_download(d, m, (int)i, m.k0[i]);
    double psnr = 0.0;
    size_t texels = 0;
    bc_pack_prepare(d, m, kc.k, true, psnr, texels);
    log_bc(log, d, m, "bc_pack_prepare seed", psnr, texels);
    for (int call = 0; call < 2; call++)
    {
        std::vector<BcRefineReport> rep;
        bc_refine(d, m, call == 0 ? 2 : 1, rep, psnr, texels);
        const std::string what = "bc_refine call " + std::to_string(call);
        for (const BcRefineReport& r : rep)
        {
            const double fields[] = { (double)r.pass, (double)r.blocks, (double)r.improved, r.delta_e };
            log.add(what + " pass " + std::to_string(r.pass), fields, sizeof(fields) / sizeof(fields[0]));
            if (!(r.delta_e <= 0.0))
                fail(std::string(kc.name) + " -j" + std::to_string(threads) + ": a refinement pass accepted a rise of " +
                     std::to_string(r.delta_e));
            if (threads == 1)
            {
                g_bc_decrease += r.delta_e;
                g_bc_improved += r.improved;
            }
        }
        log_bc(log, d, m, what, psnr, texels);
        if (call == 0)
        {
            bc_pack_prepare(d, m, kc.k, false, psnr, texels);
            log_bc(log, d, m, "bc_pack_prepare continued", psnr, texels);
            log_workspace(log, d, m, true, "bc_pack_prepare continued");
        }
    }
}

ByteLog run_kernel_case(const KernelCase& kc, int threads)
{
    Model m = kernel_model(kc);
    const std::vector<Image> chain = kernel_chain(m);
    CpuModel* d = device_create(m, chain, threads);
    kernel_latents(d, m);
    ByteLog log;
    run_objective_kernels(d, m, kc, threads, log);
    run_init_kernels(d, m, kc, log);
    run_decoder_kernels(d, m, kc, log);
    run_level1_kernels(d, m, kc, threads, log);
    run_level0_kernels(d, m, kc, log);
    run_bc_kernels(d, m, kc, threads, log);
    run_objective_kernels(d, m, kc, threads, log);
    device_destroy(d);
    return log;
}

void check_kernels()
{
    const KernelCase cases[] = {
        { "bc8 C0 2 C1 4 K 4, residual over the chain", 1, 2, 4, 5, 4, true, true, false },
        { "palette C0 3 C1 3 K 2 two textures, texture over the base", 2, 3, 3, 4, 2, false, false, true },
        { "bc8 C0 1 C1 2 K 0, residual over the base", 1, 1, 2, 6, 0, true, false, false },
        { "palette C0 2 C1 4 K 4 two textures, residual over the chain", 2, 2, 4, 5, 4, false, true, false },
        { "bc8 C0 3 C1 4 K 2 three textures, texture over the chain", 3, 3, 4, 4, 2, true, true, true },
    };
    const int counts[] = { 4, 16, 0 };
    size_t sections = 0, bytes = 0;
    for (const KernelCase& kc : cases)
    {
        const ByteLog one = run_kernel_case(kc, 1);
        for (int t : counts)
        {
            const ByteLog r = run_kernel_case(kc, t);
            if (r.names != one.names)
            {
                fail(std::string(kc.name) + " -j" + std::to_string(t) + ": a different set of outputs from -j1");
                continue;
            }
            for (size_t s = 0; s < r.data.size(); s++)
                if (r.data[s] != one.data[s])
                    fail(std::string(kc.name) + " -j" + std::to_string(t) + ": " + r.names[s] +
                         " is not the same bits as at -j1");
        }
        sections += one.data.size();
        for (const std::vector<unsigned char>& s : one.data)
            bytes += s.size();
    }
    std::printf("  kernels: %zu outputs (%zu bytes) over %zu layouts, the same bits at -j1, -j4, -j16 and -j0; the "
                "objective against its brute-force twin: worst relative %.3g (the bar is 1e-9)\n", sections, bytes,
                sizeof(cases) / sizeof(cases[0]), g_objective_twin_worst);
    std::printf("  block (b)'s arms on the deepest plane: CG residual %.3g at worst after at most %d iterations (the bar "
                "is 1e-10), dense direct solve %.3g of the range (%.0e), fd %.3g (1e-5)\n", g_level1_resid_worst,
                g_level1_iterations, g_level1_dense_worst, DENSE_BAR, g_level1_fd_worst);
    std::printf("  the BC refinement at -j1: %lld blocks improved over its passes, the blocks' own E fell by %.6e, no "
                "pass accepted a rise\n", g_bc_improved, -g_bc_decrease);
}

// ---------------------------------------------------------------------------------------------------------------------
// The two measurements
// ---------------------------------------------------------------------------------------------------------------------

double now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void bench()
{
    std::printf("Pool::run per call (chunks = the thread count, so every worker is woken; the median of 5 batches of "
                "20000 runs):\n");
    std::printf("  %-6s %14s %14s\n", "-j", "empty body", "tiny body");
    const int counts[] = { 1, 4, 16, 32 };
    for (int t : counts)
    {
        Pool pool(t);
        std::vector<double> slot((size_t)t, 0.0);
        double med[2];
        for (int kind = 0; kind < 2; kind++)
        {
            std::vector<double> batch;
            for (int b = 0; b < 5; b++)
            {
                const int runs = 20000;
                const double t0 = now_us();
                for (int r = 0; r < runs; r++)
                {
                    if (kind == 0)
                        pool.run(t, [](int) {});
                    else
                        pool.run(t, [&](int c) {
                            double s = 0.0;
                            for (int i = 0; i < 64; i++)
                                s += (double)(i ^ c);
                            slot[(size_t)c] += s;
                        });
                }
                batch.push_back((now_us() - t0) / runs);
            }
            std::sort(batch.begin(), batch.end());
            med[kind] = batch[2];
        }
        std::printf("  -j%-4d %11.2f us %11.2f us\n", t, med[0], med[1]);
    }

    std::printf("\nthe CPU backend's fixed-chunk reduction (a double sum, %zu-element chunks folded in chunk order), per "
                "call, against a plain serial loop:\n", CPU_CHUNK);
    struct Size
    {
        const char* what;
        size_t n;
    };
    const Size sizes[] = {
        { "base, level 0 of 2048x2048, C0 2 (block (c') CG vector)", (size_t)2048 * 2048 * 2 },
        { "base, level 1 of 2048x2048, C1 4 (block (b) CG vector)", (size_t)512 * 512 * 4 },
        { "deep, level 0 of the 16x16 mip, C0 2", (size_t)16 * 16 * 2 },
        { "deep, level 1 of the 8x8 mip, C1 4", (size_t)2 * 2 * 4 },
    };
    std::printf("  %-58s %8s %12s %12s %12s %12s %12s\n", "plane", "chunks", "serial", "-j1", "-j4", "-j16", "-j32");
    for (const Size& s : sizes)
    {
        const std::vector<double> v = wide_values(s.n);
        const int reps = s.n > 100000 ? 40 : 20000;
        double serial_us;
        {
            volatile double sink = 0.0;
            std::vector<double> batch;
            for (int b = 0; b < 5; b++)
            {
                const double t0 = now_us();
                for (int r = 0; r < reps; r++)
                {
                    double acc = 0.0;
                    for (double x : v)
                        acc += x;
                    sink = sink + acc;
                }
                batch.push_back((now_us() - t0) / reps);
            }
            std::sort(batch.begin(), batch.end());
            serial_us = batch[2];
        }
        std::printf("  %-58s %8d %9.2f us", s.what, chunks_for(s.n, CPU_CHUNK), serial_us);
        for (int t : counts)
        {
            Pool pool(t);
            volatile double sink = 0.0;
            std::vector<double> batch;
            for (int b = 0; b < 5; b++)
            {
                const double t0 = now_us();
                for (int r = 0; r < reps; r++)
                    sink = sink + chunked_sum(pool, v);
                batch.push_back((now_us() - t0) / reps);
            }
            std::sort(batch.begin(), batch.end());
            std::printf(" %9.2f us", batch[2]);
        }
        std::printf("\n");
    }
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        bench();
        return 0;
    }
    if (argc > 1)
    {
        std::fprintf(stderr, "ERROR: unknown option '%s' (nntc_cpu_check takes --bench or nothing)\n", argv[1]);
        return 1;
    }
    std::printf("nntc_cpu_check: the CPU backend's pool contract, its model's transfers and its kernels\n");
    check_partition();
    const int counts[] = { 1, 2, 3, 7, 16, 32, 64, 0 };
    for (int t : counts)
        check_visits(t);
    std::printf("  rule 1-2: every chunk body runs exactly once at -j1, 2, 3, 7, 16, 32, 64 and 0, for 0..1000 chunks\n");
    check_reduction();
    check_more_threads_than_chunks();
    check_resolution();
    check_churn();
    check_transfers();
    check_kernels();
    if (g_failures)
    {
        std::fprintf(stderr, "ERROR: %d check%s failed\n", g_failures, g_failures == 1 ? "" : "s");
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
