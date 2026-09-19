// fp_contract_check.cpp: does this toolchain fuse the one multiply-add the format depends on?
//
// WHY THIS EXISTS. model.h's level1_value and refine_bc.cu's bc0_value both compute
//
//     (float)((double)lo + frac * span)
//
// and both spell the device half out as __dadd_rn(__dmul_rn(..)) so that the host and the device round the multiply
// and the add separately and land on the same bits. verify_level1_on_grid depends on that agreement: every value the
// writer stores must be exactly the grid value of its own index, and one fused multiply-add moves a last bit and turns
// a correct asset into a refusal.
//
// The host side of that promise is a COMPILER SETTING, not a line of code. CMakeLists.txt puts -ffp-contract=off on the
// gcc and clang branch; the MSVC branch relies on /fp:precise not contracting, which is well founded on x64 (a double
// FMA needs /arch:AVX2 to be emitted at all) and has never been checked on MSVC ARM64, where fmadd is a baseline
// instruction and the compiler has every reason to reach for it. If it contracts there, the encoder fails on the
// Snapdragon laptop - the machine the CPU backend most exists for - on an asset that is correct.
//
// The remedy if this program refuses is one line in CMakeLists.txt: /fp:contract- on the MSVC branch, the counterpart
// of the gcc option already there. Re-run this and it passes.
//
// HOW IT ANSWERS, and the trap it was written around. The obvious design - sweep a spread of grid ends, dump the bits,
// diff the dumps between machines - DOES NOT WORK, and finding that out is most of what this file is. The two
// expressions differ in the last bit of a DOUBLE, and the result is then rounded to a FLOAT, which throws most of that
// difference away: a readable sweep of a few thousand well-conditioned ranges produced a byte-identical dump from a
// deliberately fused build, so it could not have caught a contracting compiler either. The rate at which it does
// survive to the float was measured while writing this: about one value in 5,000, and the cases are those where lo and
// frac * span nearly cancel, so the sum is small and the product's last bit is large relative to it. That is not a
// corner: a 2048x2048 level-1 plane holds a million values.
//
// So the check has three parts, in increasing order of how much they prove:
//
//   1. the readable sweep, over the grid ends a real asset carries. It is what a reader wants to look at, and on its
//      own it proves nothing;
//   2. the WITNESSES: forty triples, found by search, at which a fused multiply-add provably gives a different float.
//      Each carries BOTH answers, so this program decides on its own machine, with no second machine to diff against
//      and nothing to compare by eye. That is what makes it runnable on the Arm laptop by itself;
//   3. a large pseudorandom sweep folded into one hash, which is the belt to the witnesses' braces: it would catch a
//      contraction in some shape the witnesses happen to miss, and it is the number two machines compare.
//
// The witnesses' expected values assume IEEE double arithmetic rounded to nearest-even, which every target here has.
// A toolchain evaluating doubles in x87's 80-bit registers would fail them for a different reason, and should.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// The real thing, not a copy: level1_value, level1_index and replicate_index are read from the header the encoder and
// the writer both use, so what this program measures is what ships.
#include "model.h"

// bc0_value's HOST HALF, copied. refine_bc.cu cannot be included from a host-only program - it includes device.cuh,
// which includes <cuda_runtime.h> on its first line - and it is one of the eight frozen files, so it is not to be
// edited into something that could be. The copy is the three lines of the #else branch, verbatim.
static float bc0_value_host(float lo, float hi, float byte)
{
    const double frac = (double)byte / 255.0;
    const double span = (double)hi - (double)lo;
    return (float)((double)lo + frac * span);
}

// A float's bits, because that is the whole point: a decimal rendering would hide the difference being looked for.
static uint32_t bits_of(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static float float_of(uint32_t u)
{
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// One 64-bit FNV-1a over every value produced, so that two machines can be compared by one line before anyone reaches
// for a diff of two million.
static uint64_t g_hash = 1469598103934665603ull;

static void feed(uint32_t u)
{
    for (int b = 0; b < 4; b++)
    {
        g_hash ^= (uint64_t)((u >> (8 * b)) & 0xFFu);
        g_hash *= 1099511628211ull;
    }
}

// The grid ends a real asset carries: the fixed [-1,1] of the palette mode, the tight and the wide ranges a fitted
// level-1 channel lands on, an asymmetric pair, a degenerate one, and a pair whose span is small enough that the
// multiply's rounding is the whole story.
static const float RANGES[][2] = {
    { -1.0f, 1.0f },
    { 0.0f, 1.0f },
    { -0.5f, 0.5f },
    { 0.123456f, 0.876543f },
    { -3.7f, 12.9f },
    { 1.0f, 1.0f },
    { -1e-4f, 1e-4f },
    { -0.0009765625f, 0.0009765625f },
    { 0.3333333f, 0.6666667f },
    { -12345.678f, 98765.432f },
};
static const int RANGE_COUNT = (int)(sizeof(RANGES) / sizeof(RANGES[0]));

// THE WITNESSES. lo, hi, bits, k, then the float a separately rounded multiply and add gives, then the float a fused
// multiply-add gives. They were found by searching random grid ends with the product forced through a volatile double
// on one side and std::fma on the other; the search hit rate was about one in 5,000.
struct Level1Witness
{
    uint32_t lo, hi;
    int bits, k;
    uint32_t separate, fused;
};
static const Level1Witness L1_WITNESS[] = {
    { 0xc1816c15u, 0xbfb2d043u, 2, 2, 0xc0ca5d7cu, 0xc0ca5d7du },      // -16.1777744 .. -1.39698064
    { 0xb3be817bu, 0x34d41b39u, 2, 2, 0x347b0eb8u, 0x347b0eb7u },      // -8.87111682e-08 .. 3.95078843e-07
    { 0x4148ae5cu, 0xbe426228u, 2, 2, 0x4081bcdcu, 0x4081bcddu },      // 12.5425682 .. -0.189827561
    { 0xba6fd492u, 0xb8824f13u, 2, 2, 0xb9b59ae4u, 0xb9b59ae5u },      // -0.000914880191 .. -6.21361178e-05
    { 0xbbac92b9u, 0x3950a132u, 2, 2, 0xbad4b632u, 0xbad4b633u },      // -0.00526651414 .. 0.000198964757
    { 0x362cf9e3u, 0xb36c5db8u, 4, 11, 0x352daca6u, 0x352daca7u },     // 2.57754505e-06 .. -5.50332686e-08
    { 0xc30b1f64u, 0xbf915f74u, 8, 180, 0xc226e164u, 0xc226e163u },    // -139.12262 .. -1.1357255
    { 0x3d178a1fu, 0x3a54d6b3u, 4, 14, 0x3b534df6u, 0x3b534df5u },     // 0.0369969569 .. 0.000811915088
    { 0xb9d76204u, 0x37c3c87du, 2, 2, 0xb8fe8bf0u, 0xb8fe8bf1u },      // -0.000410810229 .. 2.33391711e-05
    { 0x3e78bf33u, 0xbcce08e1u, 2, 2, 0x3d837dfcu, 0x3d837dfdu },      // 0.242916867 .. -0.0251507182
    { 0xbf63b295u, 0x3d137732u, 7, 88, 0xbe7d014cu, 0xbe7d014du },     // -0.889443696 .. 0.0360023454
    { 0x3f8a6af7u, 0x3d50c2beu, 4, 10, 0x3ec9f42eu, 0x3ec9f42fu },     // 1.08138931 .. 0.0509669706
    { 0xc3dd264bu, 0x44baf080u, 2, 2, 0x445464f4u, 0x445464f3u },      // -442.299164 .. 1495.51562
    { 0x3909831bu, 0xb7629571u, 2, 2, 0x381195e6u, 0x381195e7u },      // 0.000131141787 .. -1.35054443e-05
    { 0xba67e6f2u, 0x3859e07au, 2, 2, 0xb98871ecu, 0xb98871edu },      // -0.000884636422 .. 5.19458918e-05
    { 0xc4d4e8e6u, 0xc1b14b68u, 4, 11, 0xc3eb3ad0u, 0xc3eb3ad1u },     // -1703.27808 .. -22.1618195
    { 0xc08b639cu, 0x3ea212a1u, 2, 2, 0xbf9ed70au, 0xbf9ed70bu },      // -4.35590935 .. 0.316548377
    { 0xc14ccfcau, 0x3f7fc461u, 6, 42, 0xc0667452u, 0xc0667453u },     // -12.8007298 .. 0.999090254
    { 0xbfa2570cu, 0x3d95cb05u, 6, 42, 0xbebf7ce4u, 0xbebf7ce5u },     // -1.26828146 .. 0.0731411353
    { 0xc1b26c50u, 0xbf814bb2u, 6, 48, 0xc0c0a55au, 0xc0c0a55bu },     // -22.302887 .. -1.01012254
};
static const int L1_WITNESS_COUNT = (int)(sizeof(L1_WITNESS) / sizeof(L1_WITNESS[0]));

struct Bc0Witness
{
    uint32_t lo, hi;
    int byte;
    uint32_t separate, fused;
};
static const Bc0Witness BC0_WITNESS[] = {
    { 0xc340dda3u, 0x44314d92u, 180, 0x43ddf290u, 0x43ddf291u },       // -192.865768 .. 709.212036
    { 0x32ea3573u, 0xb3626f02u, 204, 0xb31db9dcu, 0xb31db9ddu },       // 2.72654912e-08 .. -5.27206865e-08
    { 0x37e2db4eu, 0x34a7c436u, 204, 0x36bddfa8u, 0x36bddfa7u },       // 2.70434211e-05 .. 3.12489362e-07
    { 0x3f7f0f17u, 0xbd56299du, 210, 0x3e07f306u, 0x3e07f307u },       // 0.996324003 .. -0.0522857793
    { 0xbed73419u, 0x3b86e8e2u, 204, 0xbda56ad6u, 0xbda56ad5u },       // -0.420319349 .. 0.00411711726
    { 0xbb63f03fu, 0xb886d7ceu, 198, 0xba58e43au, 0xba58e439u },       // -0.0034780649 .. -6.42981468e-05
    { 0x35013839u, 0xb29d0059u, 156, 0x343caa1cu, 0x343caa1bu },       // 4.81380596e-07 .. -1.82773636e-08
    { 0x4393944fu, 0xc07790e8u, 170, 0x42bf9d64u, 0x42bf9d65u },       // 295.158661 .. -3.86821938
    { 0x3bcb45acu, 0xb8f1589bu, 228, 0x3a1135a2u, 0x3a1135a1u },       // 0.00620337389 .. -0.000115082796
    { 0xb522ce13u, 0x38274628u, 51, 0x36fb5ba4u, 0x36fb5ba5u },        // -6.06495803e-07 .. 3.98812408e-05
    { 0xbce255e6u, 0xbac48189u, 180, 0xbc167a32u, 0xbc167a31u },       // -0.0276288502 .. -0.00149922178
    { 0x3abc9a6eu, 0x36cd03b0u, 210, 0x3987c50eu, 0x3987c50fu },       // 0.00143892854 .. 6.10990537e-06
    { 0x40932a2du, 0x3ebc4693u, 170, 0x3fe39954u, 0x3fe39955u },       // 4.59889841 .. 0.367725939
    { 0x37c12687u, 0xb643833du, 153, 0x36fa6312u, 0x36fa6313u },       // 2.30253336e-05 .. -2.91336551e-06
    { 0x40751ad5u, 0x3b84a519u, 240, 0x3e6a9694u, 0x3e6a9695u },       // 3.8297627 .. 0.00404800149
    { 0x44a42375u, 0xbf07d040u, 204, 0x4383193eu, 0x4383193du },       // 1313.10803 .. -0.530521393
    { 0x36c59b1cu, 0xb297a644u, 242, 0x34983084u, 0x34983085u },       // 5.88911462e-06 .. -1.76543224e-08
    { 0xb8157360u, 0x36107ff2u, 185, 0xb709e4f8u, 0xb709e4f9u },       // -3.56318196e-05 .. 2.15321461e-06
    { 0xb882fe63u, 0x365edb53u, 180, 0xb7867248u, 0xb7867247u },       // -6.24626628e-05 .. 3.32082413e-06
    { 0x44f975cdu, 0x41569f28u, 180, 0x44151bc2u, 0x44151bc1u },       // 1995.68127 .. 13.4138565
};
static const int BC0_WITNESS_COUNT = (int)(sizeof(BC0_WITNESS) / sizeof(BC0_WITNESS[0]));

// The same fixed linear congruential sequence the encoder's own probes use, so the random sweep is reproducible run to
// run and machine to machine and there is no seed to pass.
static uint64_t g_state = 0x9E3779B97F4A7C15ull;

static uint32_t next32()
{
    g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(g_state >> 32);
}

// A finite float in the magnitude band a fitted grid end actually occupies, about 2^-27 to 2^13.
static float rand_float()
{
    const uint32_t m = next32() & 0x7FFFFFu;
    const uint32_t e = 100 + (next32() % 40);
    const uint32_t s = next32() & 1u;
    return float_of((s << 31) | (e << 23) | m);
}

static const long long RANDOM_VALUES = 1000000;

int main()
{
    std::printf("fp_contract_check: level1_value, level1_index and bc0_value, bit for bit\n");

    // ---- 1. the readable sweep: every index of every depth on the ends an asset carries, and the round trip
    // level1_index promises.
    long long values = 0, trips = 0, trip_bad = 0;
    for (int r = 0; r < RANGE_COUNT; r++)
    {
        const float lo = RANGES[r][0], hi = RANGES[r][1];
        for (int bits = 1; bits <= 8; bits++)
        {
            const int levels = (1 << bits) - 1;
            for (int k = 0; k <= levels; k++)
            {
                const float v = level1_value(lo, hi, bits, k);
                feed(bits_of(v));
                values++;
                std::printf("L1 %2d %3d %08x %08x %08x\n", bits, k, bits_of(lo), bits_of(hi), bits_of(v));
                // A span of zero has one grid point and every index reads back as 0, which is the function's own
                // documented answer and not a failure.
                if (hi > lo)
                {
                    trips++;
                    if (level1_index(lo, hi, bits, v) != k)
                    {
                        trip_bad++;
                        std::printf("ROUNDTRIP bits %d k %d lo %08x hi %08x came back %d\n", bits, k, bits_of(lo),
                                    bits_of(hi), level1_index(lo, hi, bits, v));
                    }
                }
            }
        }
    }
    long long bytes = 0;
    for (int r = 0; r < RANGE_COUNT; r++)
    {
        const float lo = RANGES[r][0], hi = RANGES[r][1];
        for (int b = 0; b <= 255; b++)
        {
            const uint32_t u = bits_of(bc0_value_host(lo, hi, (float)b));
            feed(u);
            bytes++;
            std::printf("B0 %3d %08x %08x %08x\n", b, bits_of(lo), bits_of(hi), u);
        }
    }

    // ---- 2. the witnesses, which are what this program actually decides on.
    int fused_here = 0, neither = 0;
    for (int i = 0; i < L1_WITNESS_COUNT; i++)
    {
        const Level1Witness& w = L1_WITNESS[i];
        const uint32_t got = bits_of(level1_value(float_of(w.lo), float_of(w.hi), w.bits, w.k));
        feed(got);
        const char* verdict = got == w.separate ? "separate" : (got == w.fused ? "FUSED" : "NEITHER");
        if (got == w.fused)
            fused_here++;
        else if (got != w.separate)
            neither++;
        std::printf("W1 %2d %08x %08x %2d %3d got %08x sep %08x fma %08x %s\n", i, w.lo, w.hi, w.bits, w.k, got,
                    w.separate, w.fused, verdict);
    }
    for (int i = 0; i < BC0_WITNESS_COUNT; i++)
    {
        const Bc0Witness& w = BC0_WITNESS[i];
        const uint32_t got = bits_of(bc0_value_host(float_of(w.lo), float_of(w.hi), (float)w.byte));
        feed(got);
        const char* verdict = got == w.separate ? "separate" : (got == w.fused ? "FUSED" : "NEITHER");
        if (got == w.fused)
            fused_here++;
        else if (got != w.separate)
            neither++;
        std::printf("W0 %2d %08x %08x %3d got %08x sep %08x fma %08x %s\n", i, w.lo, w.hi, w.byte, got, w.separate,
                    w.fused, verdict);
    }
    const int witnesses = L1_WITNESS_COUNT + BC0_WITNESS_COUNT;

    // ---- 3. the random sweep, folded into the hash and not printed: a million of each shape, which at the measured
    // rate covers a few hundred values where a fusion would show.
    for (long long t = 0; t < RANDOM_VALUES; t++)
    {
        const float lo = rand_float(), hi = rand_float();
        const int bits = 1 + (int)(next32() % 8u);
        const int k = (int)(next32() % (uint32_t)((1 << bits)));
        feed(bits_of(level1_value(lo, hi, bits, k)));
    }
    for (long long t = 0; t < RANDOM_VALUES; t++)
    {
        const float lo = rand_float(), hi = rand_float();
        feed(bits_of(bc0_value_host(lo, hi, (float)(next32() % 256u))));
    }

    std::printf("sweep values %lld  bytes %lld  round trips %lld  round trips wrong %lld\n", values, bytes, trips,
                trip_bad);
    std::printf("witnesses %d: %d separately rounded, %d fused, %d neither\n", witnesses,
                witnesses - fused_here - neither, fused_here, neither);
    std::printf("random values %lld each shape\n", RANDOM_VALUES);
    std::printf("hash %016llx\n", (unsigned long long)g_hash);

    if (fused_here != 0 || neither != 0)
    {
        std::fprintf(stderr, "ERROR: this toolchain does not compute lo + frac * span the way the format needs "
                             "(%d of %d witnesses came out fused, %d came out as neither answer): add /fp:contract- "
                             "to the MSVC branch of CMakeLists.txt, or -ffp-contract=off to the other one, and build "
                             "the encoder again - without it verify_level1_on_grid will refuse assets that are "
                             "correct\n", fused_here, witnesses, neither);
        return 1;
    }
    if (trip_bad != 0)
    {
        std::fprintf(stderr, "ERROR: %lld of %lld level-1 round trips did not come back to their own index on this "
                             "toolchain: level1_value and level1_index disagree, and the writer's assertion rests on "
                             "them agreeing\n", trip_bad, trips);
        return 1;
    }
    return 0;
}
