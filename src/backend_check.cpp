// backend_check.cpp: the per-kernel comparison harness, --backend check (docs/CPU_BACKEND_PLAN.md section 4).
//
// WHAT IT IS FOR. An end-to-end PSNR that comes out 0.3 dB low says the CPU port is wrong somewhere and not where. This
// says where. It runs a whole CUDA encode and, at every dispatched call whose CPU twin exists, gives the CPU arm the
// CUDA arm's exact pre-call state, runs both, and compares what both produced - so each CPU function is judged on
// IDENTICAL inputs, with no drift carried in from earlier calls, and a transcription mistake (a wrong index, a loop
// bound, a tie-break, a swapped channel) shows up as a difference many orders of magnitude outside rounding, at the
// call that made it.
//
// HOW ONE CALL GOES:
//   1. if the CPU arm does not implement the function yet, the CUDA arm runs it and the call is counted
//      `not implemented`; nothing else happens, so the run carries on and later calls are still checked;
//   2. otherwise the CPU model is MIRRORED from the CUDA one - every plane's values and indices, level 1's
//      correction, the BC state under --l0 bc8, and the decoder and the grids from the host model - and the CPU arm
//      gets its own copy of the host Model as it stood before the call;
//   3. the CUDA arm runs first, on the real Model and the real outputs: the encode's trajectory is always CUDA's, so one
//      divergence cannot cascade into fifty;
//   4. the CPU arm runs on its copies, and every observable output is compared: both models' planes (read back through
//      each arm's own downloads), the host Model's fields, and the call's own outputs and reports.
//
// THE BARS (section 0a point 3). CUDA and the CPU are held to agree on PSNR, not on bits, so nothing here asks a
// floating-point kernel for identical bits. Each output is compared by its kind:
//   * TRANSFERS, and every integer output that no near-tie can move: exact, bit for bit.
//   * an INDEX (a level-0 or level-1 grid index, a BC byte): exact, except that an index one step away is a FLIP - a
//     near-tie decision the contraction of an fp32 multiply-add can legitimately move - which is counted and reported
//     and is not a failure. The value beside a flipped index is not compared (it is the other grid value, by design).
//     More flips than FLIP_SHARE of an array is not a near-tie any more and is a MISMATCH.
//   * a COUNT of decisions (moved texels, improved blocks): exact, or a FLIP when the same call flipped an index.
//   * a per-ENTRY float or double (a latent value, a correction, a decoder weight, a grid end): BAR_ENTRY relative to
//     the largest magnitude in its own array, so an entry near zero is judged on the array's scale and not its own -
//     or BAR_ENTRY_DIRECT, ten times tighter, for a kernel with no solve between its inputs and its outputs.
//   * a SUMMED scalar (E and its companions, a covariance trace, a square norm): BAR_SUM relative.
// A difference past a bar is a MISMATCH, the only verdict that fails the run. The measurement behind each bar is the
// comment above its constant.
//
// WHAT IT CANNOT SEE (section 4.3, stated rather than discovered). The CUDA arm's workspaces - the assembled stencils,
// the gradients, the preconditioner, the partials - have no download on the seam, and adding one to the kernel file
// that owns the buffer would be an edit to a frozen file. What the harness CAN do without one is read device memory
// itself, from a file of its own (backend_check_probe.cu), after the CUDA call has finished writing it: that is how
// block (a)'s normal equations are compared, the intermediate result a transcription error in the accumulation shows
// up in first, and how blocks (b) and (c')'s monomial matrices, stencils, gradients, preconditioners and level 0's
// correction are compared. What it still cannot do is WRITE a workspace: a CPU function that READS a workspace another call wrote (the BC refinement reads the stencil
// the pack's assembly built; the quantised sweeps read the assembly of their own call) cannot be mirrored into from the
// CUDA side; the stage that ports such a function runs the CPU side's own producer as part of its mirror. And
// upload_decoder_to_device / upload_grids_to_device write buffers the seam cannot read back on the CUDA side: their CPU
// twins are held to the Model they were handed, and their CUDA twins are checked the first time a CPU kernel that reads
// the decoder (stage C2's objective) is compared against its CUDA twin.
//
// It is also the proof that the COPIES of section 1.8 agree with their originals: the constants, the sampling rule and
// the host math the CPU backend copied are exercised by CPU kernels that are compared, call by call, against the CUDA
// kernels that use the originals.

#include "backend_check.h"

#include "backend_check_probe.h"
#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_refine_bc.h"
#include "cpu/cpu_solve_level0.h"
#include "cuda_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nntc_check
{

namespace
{

// ---------------------------------------------------------------------------------------------------------------------
// The bars
// ---------------------------------------------------------------------------------------------------------------------

// A summed scalar: E, its companions, a trace, a square norm. MEASURED: the CUDA objective against its own fp64 host
// twin (objective_check_host - fp32 features and a two-stage device reduction against fp64 features and a serial sum,
// which is a LARGER difference than the CPU port's, whose features are fp32 too) differs by at worst 7.1e-14 relative
// over every pass of eight tests/tiny.png layouts, on MSVC and on gcc alike, and the harness re-measures that pair on
// every run it makes and prints the worst figure in its summary. 1e-9 is the bar the gate already holds that same pair
// to (objective_checks): four orders above the measurement, and still seven below the smallest transcription error worth finding (one misplaced site in a plane of
// 4096 texels moves E by about 1e-4 of itself).
const double BAR_SUM = 1e-9;

// A per-entry value relative to its array's largest magnitude. MEASURED: the CUDA conjugate-gradient correction
// against the independent host dense direct solve of the same system (--check's `block (b) dense` line) differs by
// at most 1.8e-7 of the plane's range on tests/tiny.png (1.0e-7 under --l0 palette), with the CG's own 1e-10 residual bar in between; the gate holds that
// pair to 1e-6. An fp32 contraction difference in one feature is an ulp, 6e-8 relative, and it reaches an entry
// through a conditioned solve. 1e-5 sits fifty times above the measured known-good pair, which is the plan's proposed
// ceiling (section 4.2) confirmed rather than guessed; a transcription error moves an entry by a large fraction of the
// range, not a hundred-thousandth of it. It is the bar for everything DOWNSTREAM OF AN ITERATIVE OR ILL-CONDITIONED
// SOLVE (blocks (a), (b), (c'), (c) and the BC refinement): two correct conjugate-gradient runs that stop at the same
// 1e-10 residual from arithmetic a rounding apart land as far apart as that measured pair, not a rounding apart.
// RE-MEASURED in stage C5 against the real CPU conjugate gradients: given the identical stencil the harness now holds
// bit for bit, the two arms stop on the same iteration on every call measured, and their corrections differ by at most
// 2.0e-12 of the plane's scale (level 1) and 6.3e-13 (level 0) - the summation order of the inner products and nothing
// else - while the CPU arm's own CG against its own dense direct solve differs by 1.1e-7 to 2.8e-7 of the range, the
// same place the CUDA pair sits. The bar stays where it is: a stop one probe apart, which the iteration count's slack
// allows, lands the two corrections that second distance apart, not the first.
const double BAR_ENTRY = 1e-5;

// The per-entry bar for a DIRECT kernel - one that computes its outputs from its inputs with no solve in between: the
// objective, the decode, both latents' inits and the grid snaps (direct_kernel() names them). RE-MEASURED on the real
// CPU kernels (stages C2 and C3, with the CUDA kernels built -fmad=false): over every call of 23 command lines on
// tests/tiny.png and the m1_m4 material, both level-0 modes and all three --init0 policies at both scopes, the worst
// per-entry difference is 2.3e-13 of its array's scale (a seeded level-0 value one float rounding apart, from a
// direction that differs by 3.7e-14), and every float the pca init and the snaps write is identical. That leaves the
// data room for far less than 1e-5; the floor is set instead by what a correct float output may legitimately do: a
// double a rounding apart can round to the neighbouring float, which at the top of an array's scale is up to 1.2e-7 of
// it, and a grid end one float apart moves every value snapped onto it by the same. 1e-6 is eight such roundings: ten
// times tighter than BAR_ENTRY and still clear of a legitimate flip.
const double BAR_ENTRY_DIRECT = 1e-6;

// An ASSEMBLED quantity - block (a)'s normal equations: fp32 features, formed exactly as the CUDA kernel forms them
// from identical inputs, multiplied and summed in fp64, with no solve between the inputs and the sums. It is held to the
// direct kernels' bar whichever function assembled it, because the solve that follows is what BAR_ENTRY exists for and
// an assembly has none. MEASURED in stage C4 (the comment on the table in docs/CPU_BACKEND_PLAN.md section 6 has the
// figures): with the CUDA kernels built -fmad=false and the CPU walking k_ls_accumulate's own tiles, the normal
// equations come out bit-identical on every call measured.
const double BAR_ASSEMBLED = BAR_ENTRY_DIRECT;

bool direct_kernel(const char* name)
{
    static const char* const direct[] = {
        "objective_eval",  "objective_check_host", "decode_plane",       "init_level0",
        "init_level1",     "init0_residual_channel", "quantise_level1",  "freeze_level1_grid",
        "quantise_level0", "freeze_level0_grid",   "snap_level0_on_grid",
    };
    for (const char* d : direct)
        if (std::strcmp(d, name) == 0)
            return true;
    return false;
}

// The conjugate gradients' relative-residual stop, solve_level1.cu's tolerance: the one number the harness needs to
// know which residuals are below it (compare_level1_reports).
const double CG_STOP = 1e-10;

// More index flips than this share of an array (and more than FLIP_FLOOR of them) is not a scatter of near-ties but a
// systematic disagreement, and is a MISMATCH. A near-tie is the rare index whose continuous value sat within an fp32
// rounding of the midpoint between two grid values; the share of such indices is set by the grid spacing against an
// ulp, far below 1 %.
const double FLIP_SHARE = 0.01;
const long long FLIP_FLOOR = 8;

// ---------------------------------------------------------------------------------------------------------------------
// The tally
// ---------------------------------------------------------------------------------------------------------------------

struct Tally
{
    std::string name;
    long long calls = 0, ok = 0, flips = 0, mismatches = 0, not_implemented = 0;
    double worst = 0.0;   // the largest relative difference any toleranced compare of this function saw
    bool toleranced = false;
};

std::deque<Tally> g_tally;    // in the order the functions were first called, so the summary reads like the run
std::string g_only;           // --kcheck-only: print every call of this one function
bool g_stop = false;          // --kcheck-stop: end the run at the first MISMATCH
bool g_quiet = false;         // --quiet: no summary table (the MISMATCH lines and the ERROR always print)
const Model* g_model = nullptr;   // main's own Model, for the one seam call that is not handed it (level1_poke)
int g_threads = 0;

// The calibration pair the summed-scalar bar is anchored on, re-measured every run: the CUDA objective against its
// fp64 host twin.
long long g_calib_passes = 0;
double g_calib_worst = 0.0;

// And the CPU arm's own pair, the CPU objective against the CPU copy of the twin (stage C2's acceptance).
long long g_calib_cpu_passes = 0;
double g_calib_cpu_worst = 0.0;

// THE BLOCK (b) ARMS ON THE CPU ARM'S OWN TERMS (stage C5's acceptance). The gate's --check runs three arms on the
// round-1 block (b) of its own backend: the conjugate gradients' residual under the bar, the correction against the
// independent dense direct solve of the same system, and the finite-difference gradient of E at the answer. Under this
// harness main runs those arms on the CUDA arm (the trajectory is CUDA's), so the harness runs them a second time on the
// CPU arm's own solve: the last solve_level1_all call keeps the planes the CPU arm started from, the planes it ended at,
// its correction and its report, and when main asks for the dense twin of a plane - which it does under --check only,
// right after that block (b) - the CPU arm's own dense twin solves the CPU arm's own system and its own objective is
// probed at its own answer. Everything the arms read is the CPU arm's; nothing is compared with CUDA.
struct OwnSolve
{
    bool have = false;
    std::vector<std::vector<float>> c_prev, v1;
    std::vector<std::vector<double>> delta;
    std::vector<Level1Report> rep;
};
OwnSolve g_own;
long long g_own_planes = 0, g_own_dense = 0, g_own_passes = 0;
double g_own_resid = 0.0, g_own_dense_worst = 0.0, g_own_fd = 0.0;
int g_own_iterations = 0;

// The gate's bars for the three arms (tests/run_checks.py's objective_checks and the CG's own stop): a residual under
// 1e-10 unless the iteration ran into its cap (where the gate's 1e-6 applies), the dense solve within 1e-6 of the
// correction's range, and |dE/dx| h / E under 1e-5.
const double OWN_RESIDUAL_CAPPED = 1e-6, OWN_DENSE = 1e-6, OWN_FD = 1e-5;

Tally& tally(const char* name)
{
    for (Tally& t : g_tally)
        if (t.name == name)
            return t;
    g_tally.push_back(Tally());
    g_tally.back().name = name;
    return g_tally.back();
}

// ---------------------------------------------------------------------------------------------------------------------
// The comparison of one call
// ---------------------------------------------------------------------------------------------------------------------

class Diff
{
public:
    // entry_bar is the per-entry bar of the function under test: BAR_ENTRY_DIRECT for a direct kernel, BAR_ENTRY
    // otherwise.
    explicit Diff(bool exact, double entry_bar = BAR_ENTRY) : exact_(exact), entry_bar_(entry_bar) {}

    long long flips() const { return flips_; }
    long long mismatches() const { return mismatches_; }

    // Set by a call's own output comparison when the two arms legitimately took different discrete paths through a
    // decision whose far sides differ by far more than any bar - block (a)'s ridge ladder on a near-singular normal
    // matrix - so that the decoder the two paths shipped is not then compared as if they had taken the same one.
    bool skip_decoder = false;
    double worst() const { return worst_; }
    bool toleranced() const { return toleranced_; }
    const std::string& first_mismatch() const { return first_mismatch_; }
    const std::string& first_flip() const { return first_flip_; }
    // Every toleranced compare of the call with its own relative difference, for --kcheck-only's per-quantity lines.
    const std::vector<std::pair<std::string, double>>& detail() const { return detail_; }

    void mismatch(const std::string& what)
    {
        if (mismatches_++ == 0)
            first_mismatch_ = what;
    }

    void flip(const std::string& what, long long n)
    {
        if (flips_ == 0)
            first_flip_ = what;
        flips_ += n;
    }

    // Bit for bit, any element type: memcmp per element, so a negative zero or a NaN's payload counts as a difference.
    template <class T>
    void bits(const std::string& what, const std::vector<T>& a, const std::vector<T>& b)
    {
        if (a.size() != b.size())
        {
            mismatch(what + ": sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
            return;
        }
        size_t first = a.size(), count = 0;
        for (size_t i = 0; i < a.size(); i++)
            if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0)
            {
                if (count++ == 0)
                    first = i;
            }
        if (count)
            mismatch(what + ": " + std::to_string(count) + " of " + std::to_string(a.size()) +
                     " entries differ in their bits, the first at " + std::to_string(first));
    }

    // Indices: exact, a one-step difference a FLIP, anything more a MISMATCH. Exact mode (a transfer) admits no flip.
    void indices(const std::string& what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
    {
        if (exact_)
        {
            bits(what, a, b);
            return;
        }
        if (a.size() != b.size())
        {
            mismatch(what + ": sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
            return;
        }
        long long one = 0, far = 0;
        size_t first_far = 0;
        for (size_t i = 0; i < a.size(); i++)
        {
            const int d = std::abs((int)a[i] - (int)b[i]);
            if (d == 1)
                one++;
            else if (d > 1 && far++ == 0)
                first_far = i;
        }
        if (far)
            mismatch(what + ": " + std::to_string(far) + " indices differ by more than one step, the first at " +
                     std::to_string(first_far) + " (" + std::to_string(a[first_far]) + " against " +
                     std::to_string(b[first_far]) + ")");
        if (one > std::max(FLIP_FLOOR, (long long)(FLIP_SHARE * (double)a.size())))
            mismatch(what + ": " + std::to_string(one) + " of " + std::to_string(a.size()) +
                     " indices one step apart, too many to be near-ties");
        else if (one)
            flip(what + ": " + std::to_string(one) + " of " + std::to_string(a.size()) + " indices one step apart",
                 one);
    }

    // Per-entry values against the entry bar relative to the array's largest magnitude. ka / kb, when given, are the
    // indices laid out beside the values: an entry whose index differs has already been judged by indices() and its
    // value is the other grid value, so it is skipped here. Exact mode compares the bits.
    template <class T>
    void values(const std::string& what, const std::vector<T>& a, const std::vector<T>& b,
                const std::vector<uint8_t>* ka = nullptr, const std::vector<uint8_t>* kb = nullptr)
    {
        if (exact_)
        {
            bits(what, a, b);
            return;
        }
        if (a.size() != b.size())
        {
            mismatch(what + ": sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size()));
            return;
        }
        const bool coupled = ka && kb && ka->size() == a.size() && kb->size() == a.size();
        double scale = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            if (std::isfinite((double)a[i]))
                scale = std::max(scale, std::fabs((double)a[i]));
            if (std::isfinite((double)b[i]))
                scale = std::max(scale, std::fabs((double)b[i]));
        }
        toleranced_ = true;
        size_t count = 0, first = 0;
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            if (coupled && (*ka)[i] != (*kb)[i])
                continue;
            const double r = relative((double)a[i], (double)b[i], scale);
            worst = std::max(worst, r);
            if (r > entry_bar_ && count++ == 0)
                first = i;
        }
        worst_ = std::max(worst_, worst);
        detail_.emplace_back(what, worst);
        if (count)
            mismatch(what + ": " + std::to_string(count) + " of " + std::to_string(a.size()) +
                     " entries past the bar, the first at " + std::to_string(first) + " (" + number((double)a[first]) +
                     " against " + number((double)b[first]) + ", relative " + number(worst) + " of the scale " +
                     number(scale) + ")");
    }

    // An assembled array (BAR_ASSEMBLED above): per entry, relative to the array's largest magnitude, whatever the
    // call's own entry bar is.
    template <class T>
    void assembled(const std::string& what, const std::vector<T>& a, const std::vector<T>& b)
    {
        const double keep = entry_bar_;
        entry_bar_ = BAR_ASSEMBLED;
        values(what, a, b);
        entry_bar_ = keep;
    }

    // A figure printed under --kcheck-only and judged by nothing: a quantity whose difference between the arms is
    // not a measure of the port (the CG residual below its stop bar, compare_level1_reports says why).
    void note(const std::string& what, double figure) { detail_.emplace_back(what, figure); }

    // One summed scalar against BAR_SUM relative to itself; an extreme or other single entry uses the entry bar.
    void scalar(const std::string& what, double a, double b, bool summed = true)
    {
        if (exact_)
        {
            if (std::memcmp(&a, &b, sizeof(double)) != 0)
                mismatch(what + ": " + number(a) + " against " + number(b));
            return;
        }
        toleranced_ = true;
        const double bar = summed ? BAR_SUM : entry_bar_;
        const double r = relative(a, b, std::max(std::fabs(a), std::fabs(b)));
        worst_ = std::max(worst_, r);
        detail_.emplace_back(what, r);
        if (r > bar)
            mismatch(what + ": " + number(a) + " against " + number(b) + ", relative " + number(r) + " (the bar is " +
                     number(bar) + ")");
    }

    // An integer that no near-tie can move: exact.
    void exact(const std::string& what, long long a, long long b)
    {
        if (a != b)
            mismatch(what + ": " + std::to_string(a) + " against " + std::to_string(b));
    }

    // A count of decisions: exact, or a FLIP when this call has already seen an index flip, or when the two are within
    // `slack` of each other for a reason the caller states.
    void count(const std::string& what, long long a, long long b, long long slack = 0)
    {
        if (a == b)
            return;
        if (!exact_ && (flips_ > 0 || std::llabs(a - b) <= slack))
            flip(what + ": " + std::to_string(a) + " against " + std::to_string(b), 1);
        else
            mismatch(what + ": " + std::to_string(a) + " against " + std::to_string(b));
    }

private:
    static double relative(double a, double b, double scale)
    {
        if (std::isnan(a) || std::isnan(b))
            return std::isnan(a) && std::isnan(b) ? 0.0 : std::numeric_limits<double>::infinity();
        if (a == b)
            return 0.0;
        if (!(scale > 0.0) || !std::isfinite(scale))
            return std::numeric_limits<double>::infinity();
        return std::fabs(a - b) / scale;
    }

    static std::string number(double x)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", x);
        return buf;
    }

    bool exact_;
    double entry_bar_;
    bool toleranced_ = false;
    long long flips_ = 0, mismatches_ = 0;
    double worst_ = 0.0;
    std::string first_mismatch_, first_flip_;
    std::vector<std::pair<std::string, double>> detail_;
};

// The verdict of one call, counted, and printed when it is not a plain ok (or when --kcheck-only names the function).
void record(Tally& t, const Diff& diff)
{
    const char* verdict = diff.mismatches() ? "MISMATCH" : (diff.flips() ? "FLIP" : "ok");
    if (diff.mismatches())
        t.mismatches++;
    else if (diff.flips())
        t.flips++;
    else
        t.ok++;
    t.worst = std::max(t.worst, diff.worst());
    t.toleranced = t.toleranced || diff.toleranced();
    const bool named = !g_only.empty() && g_only == t.name;
    if (diff.mismatches() || diff.flips() || named)
    {
        std::string detail = diff.mismatches() ? diff.first_mismatch() : diff.first_flip();
        std::printf("kcheck  %-26s call %-5lld %s%s%s\n", t.name.c_str(), t.calls, verdict,
                    detail.empty() ? "" : "  ", detail.c_str());
        // Under --kcheck-only, every toleranced quantity of the call with its own figure: which of a call's outputs
        // is exact and which carries the summation-order noise is what an implementer reads the harness for.
        if (named)
            for (const std::pair<std::string, double>& q : diff.detail())
                std::printf("kcheck      %-40s relative %.3g\n", q.first.c_str(), q.second);
        std::fflush(stdout);
    }
    if (diff.mismatches() && g_stop)
    {
        finish();
        std::fprintf(stderr, "ERROR: --kcheck-stop: %s call %lld is a MISMATCH: %s\n", t.name.c_str(), t.calls,
                     diff.first_mismatch().c_str());
        std::exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The two models' observable state
// ---------------------------------------------------------------------------------------------------------------------

// Everything the seam can read back from a model: every plane's two latents as values and as indices, level 1's
// correction, and under --l0 bc8 the BC state.
struct State
{
    std::vector<std::vector<float>> v0, v1;
    std::vector<std::vector<uint8_t>> k0, k1;
    std::vector<std::vector<double>> delta;
    std::vector<std::vector<uint8_t>> ep, sel;
};

void snapshot_cuda(DeviceModel* d, const Model& m, State& s)
{
    const size_t np = m.planes.size();
    s.v0.assign(np, {});
    s.v1.assign(np, {});
    s.k0.assign(np, {});
    s.k1.assign(np, {});
    s.delta.assign(np, {});
    for (size_t i = 0; i < np; i++)
    {
        nntc_cuda::level0_download_values(d, m, (int)i, s.v0[i]);
        nntc_cuda::level0_download(d, m, (int)i, s.k0[i]);
        nntc_cuda::level1_download(d, m, (int)i, s.v1[i]);
        nntc_cuda::level1_download_indices(d, m, (int)i, s.k1[i]);
        nntc_cuda::level1_delta(d, m, (int)i, s.delta[i]);
    }
    if (m.l0_bc8)
        nntc_cuda::bc_state_save(d, m, s.ep, s.sel);
}

void snapshot_cpu(nntc_cpu::CpuModel* d, const Model& m, State& s)
{
    const size_t np = m.planes.size();
    s.v0.assign(np, {});
    s.v1.assign(np, {});
    s.k0.assign(np, {});
    s.k1.assign(np, {});
    s.delta.assign(np, {});
    for (size_t i = 0; i < np; i++)
    {
        nntc_cpu::level0_download_values(d, m, (int)i, s.v0[i]);
        nntc_cpu::level0_download(d, m, (int)i, s.k0[i]);
        nntc_cpu::level1_download(d, m, (int)i, s.v1[i]);
        nntc_cpu::level1_download_indices(d, m, (int)i, s.k1[i]);
        nntc_cpu::level1_delta(d, m, (int)i, s.delta[i]);
    }
    if (m.l0_bc8)
        nntc_cpu::bc_state_save(d, m, s.ep, s.sel);
}

// The CPU model made to hold exactly what the CUDA model holds, through the transfers: the planes and the BC state
// downloaded from CUDA and uploaded to the CPU, level 1's correction written straight into the CPU model (the seam has a
// download for it and no upload), and the decoder and the grids from the host Model, which is where the CUDA side's
// came from.
void mirror(const be::Device* d, const Model& m)
{
    State s;
    snapshot_cuda(d->cuda, m, s);
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        nntc_cpu::level0_upload(d->cpu, m, (int)i, s.v0[i]);
        nntc_cpu::level0_upload_indices(d->cpu, m, (int)i, s.k0[i]);
        nntc_cpu::level1_upload(d->cpu, m, (int)i, s.v1[i]);
        nntc_cpu::level1_upload_indices(d->cpu, m, (int)i, s.k1[i]);
        std::vector<double>& x = d->cpu->planes[i].cg_x;
        if (s.delta[i].size() == x.size())
            std::copy(s.delta[i].begin(), s.delta[i].end(), x.begin());
    }
    if (m.l0_bc8)
        nntc_cpu::bc_state_restore(d->cpu, m, s.ep, s.sel);
    nntc_cpu::upload_decoder_to_device(d->cpu, m);
    nntc_cpu::upload_grids_to_device(d->cpu, m);
}

std::string plane_label(size_t i, const char* what)
{
    return "M" + std::to_string(i) + " " + what;
}

// Both models' observable state, compared. The indices first, so that a value beside a flipped index is skipped.
void compare_state(Diff& diff, const be::Device* d, const Model& m)
{
    State a, b;
    snapshot_cuda(d->cuda, m, a);
    snapshot_cpu(d->cpu, m, b);
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        diff.indices(plane_label(i, "k0"), a.k0[i], b.k0[i]);
        diff.values(plane_label(i, "v0"), a.v0[i], b.v0[i], &a.k0[i], &b.k0[i]);
        diff.indices(plane_label(i, "k1"), a.k1[i], b.k1[i]);
        diff.values(plane_label(i, "v1"), a.v1[i], b.v1[i], &a.k1[i], &b.k1[i]);
        diff.values(plane_label(i, "level-1 correction"), a.delta[i], b.delta[i]);
    }
    for (size_t i = 0; i < a.ep.size() && i < b.ep.size(); i++)
    {
        diff.indices(plane_label(i, "BC endpoints"), a.ep[i], b.ep[i]);
        diff.indices(plane_label(i, "BC selectors"), a.sel[i], b.sel[i]);
    }
    if (a.ep.size() != b.ep.size())
        diff.mismatch("the BC state's plane count");
}

// The host Model's fields a seam function may write, CUDA's call against the CPU's.
void compare_model(Diff& diff, const Model& a, const Model& b)
{
    diff.exact("the decoder's nin", a.dec.nin, b.dec.nin);
    if (!diff.skip_decoder)
    {
        diff.values("the decoder's weights", a.dec.w, b.dec.w);
        diff.values("the decoder's bias", a.dec.b, b.dec.b);
    }
    diff.values("lo0", a.lo0, b.lo0);
    diff.values("hi0", a.hi0, b.hi0);
    diff.values("lo1", a.lo1, b.lo1);
    diff.values("hi1", a.hi1, b.hi1);
    diff.exact("level 0's palette count", (long long)a.palette0.size(), (long long)b.palette0.size());
    for (size_t c = 0; c < a.palette0.size() && c < b.palette0.size(); c++)
        diff.bits("level 0's palette " + std::to_string(c), a.palette0[c], b.palette0[c]);
    diff.exact("the host k0 plane count", (long long)a.k0.size(), (long long)b.k0.size());
    for (size_t i = 0; i < a.k0.size() && i < b.k0.size(); i++)
        diff.indices(plane_label(i, "host k0"), a.k0[i], b.k0[i]);
    diff.exact("the host k1 plane count", (long long)a.k1.size(), (long long)b.k1.size());
    for (size_t i = 0; i < a.k1.size() && i < b.k1.size(); i++)
        diff.indices(plane_label(i, "host k1"), a.k1[i], b.k1[i]);
    diff.exact("the BC block plane count", (long long)a.bc0_blocks.size(), (long long)b.bc0_blocks.size());
    for (size_t i = 0; i < a.bc0_blocks.size() && i < b.bc0_blocks.size(); i++)
    {
        diff.exact(plane_label(i, "BC file count"), (long long)a.bc0_blocks[i].size(),
                   (long long)b.bc0_blocks[i].size());
        for (size_t f = 0; f < a.bc0_blocks[i].size() && f < b.bc0_blocks[i].size(); f++)
            diff.indices(plane_label(i, "BC block bytes"), a.bc0_blocks[i][f], b.bc0_blocks[i][f]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The report structs
// ---------------------------------------------------------------------------------------------------------------------

void compare_objective(Diff& diff, const Objective& a, const Objective& b)
{
    diff.scalar("E", a.e, b.e);
    diff.scalar("centre mse", a.centre_mse, b.centre_mse);
    diff.scalar("sampled mse", a.sampled_mse, b.sampled_mse);
    diff.exact("the per-plane E count", (long long)a.e_plane.size(), (long long)b.e_plane.size());
    for (size_t i = 0; i < a.e_plane.size() && i < b.e_plane.size(); i++)
        diff.scalar(plane_label(i, "E"), a.e_plane[i], b.e_plane[i]);
    diff.exact("the per-plane centre count", (long long)a.centre_plane.size(), (long long)b.centre_plane.size());
    for (size_t i = 0; i < a.centre_plane.size() && i < b.centre_plane.size(); i++)
        diff.scalar(plane_label(i, "centre mse"), a.centre_plane[i], b.centre_plane[i]);
}

void compare_level1_init(Diff& diff, const Level1InitReport& a, const Level1InitReport& b)
{
    diff.exact("pca", a.pca, b.pca);
    diff.exact("components", a.components, b.components);
    for (int c = 0; c < 4; c++)
    {
        diff.scalar("eigenvalue " + std::to_string(c), a.eigenvalue[c], b.eigenvalue[c]);
        diff.scalar("peak " + std::to_string(c), a.peak[c], b.peak[c], false);
    }
    diff.scalar("total variance", a.total_variance, b.total_variance);
}

void compare_init0(Diff& diff, const Init0ChannelReport& a, const Init0ChannelReport& b)
{
    diff.scalar("eigenvalue", a.eigenvalue, b.eigenvalue);
    diff.scalar("total variance", a.total_variance, b.total_variance);
    diff.scalar("share", a.share, b.share);
    diff.scalar("peak", a.peak, b.peak, false);
    diff.scalar("divisor", a.divisor, b.divisor, false);
    // The chain-wide histogram's percentile, as the bin the divisor was read at. Under --l0 palette the divisor IS
    // peak (bin + 1) / PROJ_HIST_BINS, so the bin is recovered from the two numbers the report carries (under --l0 bc8
    // the divisor is the peak and the bin reads as the last one on both arms). The counts behind it are integers and
    // the bin is a discrete choice, so it is held exact.
    auto bin = [](const Init0ChannelReport& r) {
        return r.peak > 0.0 ? std::llround(r.divisor / r.peak * (double)PROJ_HIST_BINS) - 1 : -1ll;
    };
    diff.exact("the percentile's histogram bin", bin(a), bin(b));
    diff.exact("nout", a.nout, b.nout);
    diff.exact("texture", a.texture, b.texture);
    diff.values("direction", std::vector<double>(a.dir, a.dir + 3 * MAX_TEXTURES),
                std::vector<double>(b.dir, b.dir + 3 * MAX_TEXTURES));
}

void compare_level1_reports(Diff& diff, const std::vector<Level1Report>& a, const std::vector<Level1Report>& b)
{
    diff.exact("the report's plane count", (long long)a.size(), (long long)b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        const Level1Report& x = a[i];
        const Level1Report& y = b[i];
        // The conjugate gradients' stop is probed every fourth iteration (solve_level1.cu's CG_PROBE), so a residual
        // that crosses its bar within a rounding of it moves the count by one probe.
        diff.count(plane_label(i, "CG iterations"), x.iterations, y.iterations, 4);
        // The residual the iteration stopped at. Where both arms stopped on the same probe with the residual under
        // the stop bar, its value is NOT a measure of the port: under the bar the recurrence's residual is the
        // rounding of its own inner products, and two summation orders of the same doubles put it anywhere below the
        // bar - MEASURED in stage C5 on identical stencils, identical preconditioners and the same iteration count,
        // 3.71279e-11 against 3.71289e-11 on one plane and a third apart on a plane whose residual was 1e-12. What
        // the solve produced is the correction, which compare_state holds per entry; here the residual is held to
        // being under the bar on both arms, and its difference in units of the bar is printed. Anywhere else - an arm
        // that ran into the iteration cap, or a different stop - it is held to the entry bar like any other extreme.
        if (x.iterations == y.iterations && x.residual < CG_STOP && y.residual < CG_STOP)
            diff.note(plane_label(i, "CG residual, both under the stop bar, |difference| / bar"),
                      std::fabs(x.residual - y.residual) / CG_STOP);
        else
            diff.scalar(plane_label(i, "CG residual"), x.residual, y.residual, false);
        diff.scalar(plane_label(i, "lambda"), x.lambda, y.lambda);
        diff.scalar(plane_label(i, "smallest diagonal"), x.min_diag, y.min_diag, false);
        diff.scalar(plane_label(i, "largest diagonal"), x.max_diag, y.max_diag, false);
        diff.count(plane_label(i, "values moved"), x.moved_values, y.moved_values);
        diff.scalar(plane_label(i, "|delta|^2"), x.delta2, y.delta2);
        diff.scalar(plane_label(i, "|plane|^2"), x.plane2, y.plane2);
        for (int c = 0; c < 4; c++)
        {
            diff.scalar(plane_label(i, "range lo"), x.lo[c], y.lo[c], false);
            diff.scalar(plane_label(i, "range hi"), x.hi[c], y.hi[c], false);
        }
        diff.exact(plane_label(i, "quantised"), x.quantised, y.quantised);
        diff.exact(plane_label(i, "sweeps"), x.sweeps, y.sweeps);
        diff.count(plane_label(i, "indices moved"), x.moved, y.moved);
    }
}

// Blocks (b) and (c')'s workspaces, read out of both arms after the call (the CUDA side through the probe): the
// monomial matrices the assembly expanded against and every plane's stencil and gradient, held to BAR_ASSEMBLED - they
// are assemblies, from identical planes and an identical decoder - and, where the call wrote them, the preconditioner
// (from the ridge, whose mean is a reduction in each arm's own order, so to the entry bar), the plane the quantised
// sweeps measure from (a copy, so exact) and level 0's correction (downstream of the solve, the entry bar; level 1's
// is already in compare_state).
void compare_workspace(Diff& diff, const be::Device* d, const Model& m, bool level0, bool solved, bool swept)
{
    const std::string lat = level0 ? "level-0 " : "level-1 ";
    std::vector<double> ma;
    nntc_probe::monomials(d->cuda, m, level0, ma);
    const std::vector<double>& cm = level0 ? d->cpu->mono0 : d->cpu->mono;
    diff.assembled(lat + "monomial matrices", ma,
                   std::vector<double>(cm.begin(), cm.begin() + (std::ptrdiff_t)std::min(ma.size(), cm.size())));
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        nntc_probe::Workspace a;
        nntc_probe::workspace(d->cuda, m, (int)i, level0, a);
        const nntc_cpu::PlaneWork w = nntc_cpu::plane_work(d->cpu, m, d->cpu->planes[i], level0);
        const size_t n = w.values(), texels = w.texels(), c = (size_t)w.c;
        diff.assembled(plane_label(i, (lat + "stencil").c_str()), a.stencil,
                       std::vector<double>(w.stencil, w.stencil + texels * nntc_cpu::STENCIL_BLOCKS * c * c));
        diff.assembled(plane_label(i, (lat + "gradient").c_str()), a.grad, std::vector<double>(w.grad, w.grad + n));
        if (solved)
            diff.values(plane_label(i, (lat + "preconditioner").c_str()), a.precond,
                        std::vector<float>(w.precond, w.precond + texels * c * c));
        if (swept)
            diff.bits(plane_label(i, (lat + "plane the sweeps measure from").c_str()), a.prev,
                      std::vector<float>(w.prev, w.prev + n));
        if (level0)
            diff.values(plane_label(i, "level-0 correction"), a.cg_x, std::vector<double>(w.cg_x, w.cg_x + n));
    }
}

void compare_level0_reports(Diff& diff, const std::vector<Level0Report>& a, const std::vector<Level0Report>& b)
{
    diff.exact("the report's plane count", (long long)a.size(), (long long)b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        for (int c = 0; c < 4; c++)
            diff.count(plane_label(i, "moved in colour ") + std::to_string(c), a[i].moved[c], b[i].moved[c]);
        diff.count(plane_label(i, "moved"), a[i].moved_total, b[i].moved_total);
        diff.exact(plane_label(i, "joint"), a[i].joint, b[i].joint);
        diff.exact(plane_label(i, "states"), a[i].states, b[i].states);
    }
}

void compare_bc_reports(Diff& diff, const std::vector<BcRefineReport>& a, const std::vector<BcRefineReport>& b)
{
    diff.exact("the refinement's pass count", (long long)a.size(), (long long)b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
    {
        diff.exact("pass", a[i].pass, b[i].pass);
        diff.exact("blocks", a[i].blocks, b[i].blocks);
        diff.count("blocks improved", a[i].improved, b[i].improved);
        diff.scalar("accepted decrease", a[i].delta_e, b[i].delta_e);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// One call under the harness
// ---------------------------------------------------------------------------------------------------------------------

// cuda() is the CUDA arm's call on the real Model and outputs; cpu(mc) the CPU arm's on its own copy of the Model and
// its own outputs; outputs(diff, mc) compares the call's own outputs, and runs before the host Model is compared so
// that it can say a legitimate discrete divergence happened (Diff::skip_decoder). `exact` is true for a transfer, which
// moves bits and computes nothing.
//
// The host Model is compared only for a function that takes it MUTABLY (M is Model, not const Model): one that takes it
// const cannot write it, and the caller may hand such a function an output that IS a Model field - main downloads
// straight into m.k0[i] - which the CUDA call then changes under the harness's pre-call copy while the CPU call writes
// its own vector. Comparing the Model there would compare a download with the state before it.
template <class M, class Cuda, class Cpu, class Outputs>
void harness(const char* name, const be::Device* d, M& m, bool exact, const Cuda& cuda, const Cpu& cpu,
             const Outputs& outputs)
{
    Tally& t = tally(name);
    t.calls++;
    if (!nntc_cpu::implemented(name))
    {
        cuda();
        t.not_implemented++;
        return;
    }
    mirror(d, m);
    Model mc = m;
    cuda();
    cpu(mc);
    Diff diff(exact, direct_kernel(name) ? BAR_ENTRY_DIRECT : BAR_ENTRY);
    compare_state(diff, d, m);
    outputs(diff, mc);
    if constexpr (!std::is_const<M>::value)
        compare_model(diff, m, mc);
    record(t, diff);
}

// The same for the few functions that take no model handle at all: both arms are called and their answers compared.
void handleless_verdict(const char* name, bool agree, const std::string& detail)
{
    Tally& t = tally(name);
    t.calls++;
    Diff diff(true);
    if (!agree)
        diff.mismatch(detail);
    record(t, diff);
}

// ---------------------------------------------------------------------------------------------------------------------
// The start-up round trip
// ---------------------------------------------------------------------------------------------------------------------

// Values a careless copy would mangle, among ordinary ones: a negative zero, the smallest subnormal and both extremes.
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

uint8_t pattern_byte(size_t i, unsigned salt)
{
    return (uint8_t)((i * 37u + salt) & 0xFFu);
}

// Every plane transfer, once, on both arms, through the harness's own wrappers so that each is compared like any other
// call - and then both models are put back to exactly what the CUDA model held before, so the encode that follows runs
// on the state it would have run on without the harness. Uploads of the CUDA model's own downloaded bytes restore it bit
// for bit (they are the bytes it held, whatever an allocation left in them). The decoder and grid uploads are not
// probed: the seam has no CUDA download to restore them from, and they are compared whenever the encode calls them.
void startup_round_trip(be::Device* d, const Model& m)
{
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const int p = (int)i;
        std::vector<float> keep_v0, keep_v1, pat, out_a;
        std::vector<uint8_t> keep_k0, keep_k1, idx, idx_out;
        nntc_cuda::level0_download_values(d->cuda, m, p, keep_v0);
        nntc_cuda::level1_download(d->cuda, m, p, keep_v1);
        nntc_cuda::level0_download(d->cuda, m, p, keep_k0);
        nntc_cuda::level1_download_indices(d->cuda, m, p, keep_k1);

        pat.resize(keep_v0.size());
        for (size_t j = 0; j < pat.size(); j++)
            pat[j] = pattern_value(j + i);
        nntc_check::level0_upload(d, m, p, pat);
        nntc_check::level0_download_values(d, m, p, out_a);

        pat.resize(keep_v1.size());
        for (size_t j = 0; j < pat.size(); j++)
            pat[j] = pattern_value(j + 3 * i + 5);
        nntc_check::level1_upload(d, m, p, pat);
        nntc_check::level1_download(d, m, p, out_a);
        std::vector<float> lo, hi;
        nntc_check::level1_range(d, m, p, lo, hi);
        if (!pat.empty())
        {
            nntc_check::level1_poke(d, p, pat.size() / 2, 0.25f);
            nntc_check::level1_range(d, m, p, lo, hi);
        }

        idx.resize(keep_k0.size());
        for (size_t j = 0; j < idx.size(); j++)
            idx[j] = pattern_byte(j, 11u + (unsigned)i);
        nntc_check::level0_upload_indices(d, m, p, idx);
        nntc_check::level0_download(d, m, p, idx_out);
        idx.resize(keep_k1.size());
        for (size_t j = 0; j < idx.size(); j++)
            idx[j] = pattern_byte(j, 97u + (unsigned)i);
        nntc_check::level1_upload_indices(d, m, p, idx);
        nntc_check::level1_download_indices(d, m, p, idx_out);

        std::vector<double> delta;
        nntc_check::level1_delta(d, m, p, delta);

        // Back to the CUDA model's own bytes, on both arms.
        nntc_cuda::level0_upload(d->cuda, m, p, keep_v0);
        nntc_cuda::level1_upload(d->cuda, m, p, keep_v1);
        nntc_cuda::level0_upload_indices(d->cuda, m, p, keep_k0);
        nntc_cuda::level1_upload_indices(d->cuda, m, p, keep_k1);
    }
    if (m.l0_bc8)
    {
        std::vector<std::vector<uint8_t>> keep_ep, keep_sel, ep, sel, out_ep, out_sel;
        nntc_cuda::bc_state_save(d->cuda, m, keep_ep, keep_sel);
        nntc_check::bc_state_save(d, m, out_ep, out_sel);
        ep = keep_ep;
        sel = keep_sel;
        for (size_t i = 0; i < ep.size(); i++)
        {
            for (size_t j = 0; j < ep[i].size(); j++)
                ep[i][j] = pattern_byte(j, 5u + (unsigned)i);
            for (size_t j = 0; j < sel[i].size(); j++)
                sel[i][j] = pattern_byte(j, 201u + (unsigned)i);
        }
        nntc_check::bc_state_restore(d, m, ep, sel);
        nntc_check::bc_state_save(d, m, out_ep, out_sel);
        nntc_cuda::bc_state_restore(d->cuda, m, keep_ep, keep_sel);
    }
    mirror(d, m);
}

}   // namespace

// ---------------------------------------------------------------------------------------------------------------------
// The harness's own entry points
// ---------------------------------------------------------------------------------------------------------------------

void options(const std::string& only, bool stop_at_mismatch, bool quiet)
{
    g_only = only;
    g_stop = stop_at_mismatch;
    g_quiet = quiet;
}

namespace
{

// The summary table, one line per dispatched function in the order the run first called it.
void print_summary(long long calls, long long mismatches, long long flips, long long missing)
{
    std::printf("\nbackend check: %lld dispatched calls; the CUDA arm ran the encode, and at every call the CPU arm "
                "implements it was given the CUDA arm's state and compared (cpu workers %d)\n", calls, g_threads);
    std::printf("  bars: transfers and integers exact; an index one step off a FLIP (counted, not failed, unless "
                "more than %.0f %% of an array);\n", 100.0 * FLIP_SHARE);
    std::printf("        a per-entry value %.0e of its array's scale (%.0e for a direct kernel: the objective, the decode, "
                "the inits, the snaps);\n        a summed scalar %.0e relative\n", BAR_ENTRY, BAR_ENTRY_DIRECT, BAR_SUM);
    if (g_calib_passes > 0)
        std::printf("  calibration: the CUDA objective against its fp64 host twin over %lld passes: worst relative "
                    "difference %.3g (the summed-scalar bar is %.0e)\n", g_calib_passes, g_calib_worst, BAR_SUM);
    if (g_calib_cpu_passes > 0)
        std::printf("               the CPU objective against its own copy of the twin over %lld passes: worst "
                    "relative difference %.3g\n", g_calib_cpu_passes, g_calib_cpu_worst);
    if (g_own_planes > 0)
        std::printf("  own terms: the CPU arm's round-1 block (b) over %lld planes: CG residual %.3g at worst (the bar "
                    "is %.0e), %d iterations at most; dense direct solve against its CG over %lld planes %.3g of the "
                    "range (the bar is %.0e); fd %.3g (the bar is %.0e)\n", g_own_planes, g_own_resid, CG_STOP,
                    g_own_iterations, g_own_dense, g_own_dense_worst, OWN_DENSE, g_own_fd, OWN_FD);
    std::printf("  %-28s %7s %7s %7s %11s %16s  %s\n", "function", "calls", "ok", "flips", "mismatches",
                "not implemented", "worst relative");
    for (const Tally& t : g_tally)
    {
        char worst[32] = "-";
        if (t.toleranced)
            std::snprintf(worst, sizeof(worst), "%.3g", t.worst);
        std::printf("  %-28s %7lld %7lld %7lld %11lld %16lld  %s\n", t.name.c_str(), t.calls, t.ok, t.flips,
                    t.mismatches, t.not_implemented, worst);
    }
    std::printf("  total: %lld mismatches, %lld flips, %lld calls not implemented on the cpu arm yet\n", mismatches,
                flips, missing);
    std::fflush(stdout);
}

}   // namespace

bool finish()
{
    long long calls = 0, mismatches = 0, flips = 0, missing = 0;
    for (const Tally& t : g_tally)
    {
        calls += t.calls;
        mismatches += t.mismatches;
        flips += t.flips;
        missing += t.not_implemented;
    }
    if (!g_quiet)
        print_summary(calls, mismatches, flips, missing);
    // The CPU objective and its own brute-force twin are one backend's two derivations of the same number, so they
    // are held to the bar the gate's objective_checks holds the CUDA pair to, and a pair past it fails the run.
    if (g_calib_cpu_worst > BAR_SUM)
    {
        std::fprintf(stderr, "ERROR: backend check: the CPU objective and its own brute-force twin differ by %.3g "
                             "relative, past the %.0e bar\n", g_calib_cpu_worst, BAR_SUM);
        return false;
    }
    if (mismatches > 0)
    {
        std::fprintf(stderr, "ERROR: backend check: %lld call%s disagreed past the bar (the MISMATCH lines above name "
                             "each one)\n", mismatches, mismatches == 1 ? "" : "s");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// The lifecycle
// ---------------------------------------------------------------------------------------------------------------------

bool device_select(int index, DeviceInfo& info)
{
    if (!nntc_cuda::device_select(index, info))
        return false;
    DeviceInfo cpu;
    const bool agree = nntc_cpu::device_select(index, cpu);
    handleless_verdict("device_select", agree, "the cpu arm refused its device");
    return true;
}

void device_create(be::Device* d, const Model& m, const std::vector<Image>& source_chain, int threads)
{
    g_model = &m;
    g_threads = threads;
    d->cuda = nntc_cuda::device_create(m, source_chain);
    d->cpu = nntc_cpu::device_create(m, source_chain, threads);
    handleless_verdict("device_create", d->cuda && d->cpu, "an arm returned no model");

    // stencil_monomials is a pure function of one small integer, and plane_work (device.cuh) calls the CUDA arm's
    // unqualified while the CPU model's twin calls the CPU arm's: assert once that they agree for every channel count
    // either latent can have (plan section 1.7).
    for (int c = 1; c <= nntc_cpu::MAX_CHANNELS; c++)
        stencil_monomials(c);
    nntc_check::device_memory(d);
    startup_round_trip(d, m);
}

void device_destroy(be::Device* d)
{
    nntc_cuda::device_destroy(d->cuda);
    nntc_cpu::device_destroy(d->cpu);
    d->cuda = nullptr;
    d->cpu = nullptr;
}

// The CPU model allocates init.cu's buffer list buffer for buffer, so the two figures are the same number.
size_t device_memory(const be::Device* d)
{
    const size_t a = nntc_cuda::device_memory(d->cuda);
    const size_t b = nntc_cpu::device_memory(d->cpu);
    handleless_verdict("device_memory", a == b,
                       "cuda " + std::to_string(a) + " bytes against cpu " + std::to_string(b));
    return a;
}

int stencil_monomials(int c0)
{
    const int a = nntc_cuda::stencil_monomials(c0);
    const int b = nntc_cpu::stencil_monomials(c0);
    handleless_verdict("stencil_monomials", a == b,
                       "c0 " + std::to_string(c0) + ": cuda " + std::to_string(a) + " against cpu " + std::to_string(b));
    return a;
}

// ---------------------------------------------------------------------------------------------------------------------
// The kernels. Every one is the same harness() call: the two arms' calls and the outputs to compare.
// ---------------------------------------------------------------------------------------------------------------------

void init_level0(be::Device* d, Model& m, const float* luma_weights, bool luma_channel0)
{
    harness("init_level0", d, m, false, [&] { nntc_cuda::init_level0(d->cuda, m, luma_weights, luma_channel0); },
            [&](Model& mc) { nntc_cpu::init_level0(d->cpu, mc, luma_weights, luma_channel0); },
            [](Diff&, const Model&) {});
}

void init_level1(be::Device* d, Model& m, bool pca, Level1InitReport& rep)
{
    Level1InitReport rc = rep;
    harness("init_level1", d, m, false, [&] { nntc_cuda::init_level1(d->cuda, m, pca, rep); },
            [&](Model& mc) { nntc_cpu::init_level1(d->cpu, mc, pca, rc); },
            [&](Diff& diff, const Model&) { compare_level1_init(diff, rep, rc); });
}

void init0_residual_channel(be::Device* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep)
{
    Init0ChannelReport rc = rep;
    harness("init0_residual_channel", d, m, false,
            [&] { nntc_cuda::init0_residual_channel(d->cuda, m, channel, plane_weight, rep); },
            [&](Model& mc) { nntc_cpu::init0_residual_channel(d->cpu, mc, channel, plane_weight, rc); },
            [&](Diff& diff, const Model&) { compare_init0(diff, rep, rc); });
}

void quantise_level1(be::Device* d, Model& m, const std::string& range_policy)
{
    harness("quantise_level1", d, m, false, [&] { nntc_cuda::quantise_level1(d->cuda, m, range_policy); },
            [&](Model& mc) { nntc_cpu::quantise_level1(d->cpu, mc, range_policy); }, [](Diff&, const Model&) {});
}

void freeze_level1_grid(be::Device* d, Model& m, const std::string& range_policy)
{
    harness("freeze_level1_grid", d, m, false, [&] { nntc_cuda::freeze_level1_grid(d->cuda, m, range_policy); },
            [&](Model& mc) { nntc_cpu::freeze_level1_grid(d->cpu, mc, range_policy); }, [](Diff&, const Model&) {});
}

void quantise_level0(be::Device* d, Model& m, const std::string& range_policy)
{
    harness("quantise_level0", d, m, false, [&] { nntc_cuda::quantise_level0(d->cuda, m, range_policy); },
            [&](Model& mc) { nntc_cpu::quantise_level0(d->cpu, mc, range_policy); }, [](Diff&, const Model&) {});
}

void freeze_level0_grid(be::Device* d, Model& m, const std::string& range_policy)
{
    harness("freeze_level0_grid", d, m, false, [&] { nntc_cuda::freeze_level0_grid(d->cuda, m, range_policy); },
            [&](Model& mc) { nntc_cpu::freeze_level0_grid(d->cpu, mc, range_policy); }, [](Diff&, const Model&) {});
}

void snap_level0_on_grid(be::Device* d, Model& m)
{
    harness("snap_level0_on_grid", d, m, false, [&] { nntc_cuda::snap_level0_on_grid(d->cuda, m); },
            [&](Model& mc) { nntc_cpu::snap_level0_on_grid(d->cpu, mc); }, [](Diff&, const Model&) {});
}

// Block (a)'s rung disagreements over the run: the one discrete decision of the block, taken by a float comparison
// against a threshold (solve_decoder.cu's acceptance bar), so on a near-singular normal matrix identical-to-rounding
// normal equations can land on the two sides of it. Each one is a FLIP of its call and is carried into the tally's
// comparison below, which would otherwise read the same flip a second time as a mismatch.
namespace
{

long long g_rung_flips = 0;

// The rung one call ended on, from the tally before and after it: 0 standard, 1 reduced, 2 none, 3 refused.
int rung_taken(const long long before[4], const long long after[4])
{
    for (int r = 0; r < 4; r++)
        if (after[r] != before[r])
            return r;
    return -1;
}

}   // namespace

// The block solves return their own kernel time, which is never compared: the report is CUDA's.
//
// Block (a) is compared at three depths. The NORMAL EQUATIONS, read out of both arms' workspaces after the call, to
// BAR_ASSEMBLED: they are the accumulation's entire output, so a transcription error there is named there. The RUNG
// the ridge ladder took, exact - it is copied host code on the same normal equations - except that a disagreement
// after normal equations that agreed is the near-tie the plan allows for (section 7.4: the degenerate dot) and is a
// FLIP, after which the two decoders are not compared, because they are the far sides of the ladder's decision. And
// the DECODER itself, per weight, through compare_model.
double solve_decoder(be::Device* d, Model& m, int k, const std::vector<double>& per_site)
{
    double ms = 0.0;
    long long cuda_before[4] = {}, cuda_after[4] = {}, cpu_before[4] = {}, cpu_after[4] = {};
    harness("solve_decoder", d, m, false,
            [&] {
                nntc_cuda::decoder_ridge_tally(cuda_before);
                ms = nntc_cuda::solve_decoder(d->cuda, m, k, per_site);
                nntc_cuda::decoder_ridge_tally(cuda_after);
            },
            [&](Model& mc) {
                nntc_cpu::decoder_ridge_tally(cpu_before);
                nntc_cpu::solve_decoder(d->cpu, mc, k, per_site);
                nntc_cpu::decoder_ridge_tally(cpu_after);
            },
            [&](Diff& diff, const Model&) {
                std::vector<double> a, b;
                nntc_probe::normal_equations(d->cuda, m, a);
                b.assign(d->cpu->reduction.begin(), d->cpu->reduction.begin() + (std::ptrdiff_t)a.size());
                const long long before = diff.mismatches();
                diff.assembled("the normal equations", a, b);
                const int ra = rung_taken(cuda_before, cuda_after), rb = rung_taken(cpu_before, cpu_after);
                if (ra != rb)
                {
                    diff.skip_decoder = true;
                    if (diff.mismatches() == before)
                    {
                        g_rung_flips++;
                        diff.flip("the ridge ladder's rung: cuda " + std::to_string(ra) + " against cpu " +
                                  std::to_string(rb) + " on normal equations that agree", 1);
                    }
                    else
                        diff.mismatch("the ridge ladder's rung: cuda " + std::to_string(ra) + " against cpu " +
                                      std::to_string(rb));
                }
            });
    return ms;
}

// The tally counts the CPU arm's own block (a) calls, which under the harness are exactly the CUDA arm's calls, so
// the two four-rung counts are compared outright once the CPU arm has them - exact, except by the rung flips
// solve_decoder above has already counted, each of which moves one call from one rung to another.
void decoder_ridge_tally(long long out[4])
{
    nntc_cuda::decoder_ridge_tally(out);
    Tally& t = tally("decoder_ridge_tally");
    t.calls++;
    if (!nntc_cpu::implemented("decoder_ridge_tally"))
    {
        t.not_implemented++;
        return;
    }
    long long cpu[4] = {};
    nntc_cpu::decoder_ridge_tally(cpu);
    Diff diff(false);
    for (int r = 0; r < 4; r++)
        diff.count("rung " + std::to_string(r), out[r], cpu[r], g_rung_flips);
    record(t, diff);
}

double solve_level1_all(be::Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    double ms = 0.0;
    std::vector<Level1Report> rc;
    harness("solve_level1_all", d, m, false, [&] { ms = nntc_cuda::solve_level1_all(d->cuda, m, k, ridge, rep); },
            [&](Model& mc) {
                // The CPU arm's own solve, kept for the arms on its own terms (g_own above).
                const size_t np = d->cpu->planes.size();
                g_own.c_prev.resize(np);
                g_own.v1.resize(np);
                g_own.delta.resize(np);
                for (size_t i = 0; i < np; i++)
                    g_own.c_prev[i] = d->cpu->planes[i].v1;
                nntc_cpu::solve_level1_all(d->cpu, mc, k, ridge, rc);
                for (size_t i = 0; i < np; i++)
                {
                    g_own.v1[i] = d->cpu->planes[i].v1;
                    g_own.delta[i] = d->cpu->planes[i].cg_x;
                }
                g_own.rep = rc;
                g_own.have = true;
            },
            [&](Diff& diff, const Model&) {
                compare_level1_reports(diff, rep, rc);
                compare_workspace(diff, d, m, false, true, false);
            });
    return ms;
}

double sweep_level1_all(be::Device* d, const Model& m, int k, double ridge, int sweeps,
                        std::vector<Level1Report>& rep)
{
    double ms = 0.0;
    std::vector<Level1Report> rc;
    harness("sweep_level1_all", d, m, false,
            [&] { ms = nntc_cuda::sweep_level1_all(d->cuda, m, k, ridge, sweeps, rep); },
            [&](Model& mc) { nntc_cpu::sweep_level1_all(d->cpu, mc, k, ridge, sweeps, rc); },
            [&](Diff& diff, const Model&) {
                compare_level1_reports(diff, rep, rc);
                compare_workspace(diff, d, m, false, false, true);
            });
    return ms;
}

double solve_level0_cont_all(be::Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    double ms = 0.0;
    std::vector<Level1Report> rc;
    harness("solve_level0_cont_all", d, m, false,
            [&] { ms = nntc_cuda::solve_level0_cont_all(d->cuda, m, k, ridge, rep); },
            [&](Model& mc) { nntc_cpu::solve_level0_cont_all(d->cpu, mc, k, ridge, rc); },
            [&](Diff& diff, const Model&) {
                compare_level1_reports(diff, rep, rc);
                compare_workspace(diff, d, m, true, true, false);
            });
    return ms;
}

double sweep_level0_cont_all(be::Device* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep)
{
    double ms = 0.0;
    std::vector<Level1Report> rc;
    harness("sweep_level0_cont_all", d, m, false,
            [&] { ms = nntc_cuda::sweep_level0_cont_all(d->cuda, m, k, ridge, sweeps, rep); },
            [&](Model& mc) { nntc_cpu::sweep_level0_cont_all(d->cpu, mc, k, ridge, sweeps, rc); },
            [&](Diff& diff, const Model&) {
                compare_level1_reports(diff, rep, rc);
                compare_workspace(diff, d, m, true, false, true);
            });
    return ms;
}

double assemble_level0_for_refine(be::Device* d, const Model& m, int k)
{
    double ms = 0.0;
    harness("assemble_level0_for_refine", d, m, false,
            [&] { ms = nntc_cuda::assemble_level0_for_refine(d->cuda, m, k); },
            [&](Model& mc) { nntc_cpu::assemble_level0_for_refine(d->cpu, mc, k); },
            [&](Diff& diff, const Model&) { compare_workspace(diff, d, m, true, false, true); });
    return ms;
}

// ---------------------------------------------------------------------------------------------------------------------
// The transfers: exact, and the output of a download compared bit for bit.
// ---------------------------------------------------------------------------------------------------------------------

void level1_download_indices(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& k1)
{
    std::vector<uint8_t> kc;
    harness("level1_download_indices", d, m, true, [&] { nntc_cuda::level1_download_indices(d->cuda, m, plane, k1); },
            [&](Model& mc) { nntc_cpu::level1_download_indices(d->cpu, mc, plane, kc); },
            [&](Diff& diff, const Model&) { diff.bits("the downloaded k1", k1, kc); });
}

void level1_upload(be::Device* d, const Model& m, int plane, const std::vector<float>& v1)
{
    harness("level1_upload", d, m, true, [&] { nntc_cuda::level1_upload(d->cuda, m, plane, v1); },
            [&](Model& mc) { nntc_cpu::level1_upload(d->cpu, mc, plane, v1); }, [](Diff&, const Model&) {});
}

void level1_upload_indices(be::Device* d, const Model& m, int plane, const std::vector<uint8_t>& k1)
{
    harness("level1_upload_indices", d, m, true, [&] { nntc_cuda::level1_upload_indices(d->cuda, m, plane, k1); },
            [&](Model& mc) { nntc_cpu::level1_upload_indices(d->cpu, mc, plane, k1); }, [](Diff&, const Model&) {});
}

void level0_upload_indices(be::Device* d, const Model& m, int plane, const std::vector<uint8_t>& k0)
{
    harness("level0_upload_indices", d, m, true, [&] { nntc_cuda::level0_upload_indices(d->cuda, m, plane, k0); },
            [&](Model& mc) { nntc_cpu::level0_upload_indices(d->cpu, mc, plane, k0); }, [](Diff&, const Model&) {});
}

// The CUDA side of the two uploads below cannot be read back through the seam, so the CPU side is held to the Model it
// was handed, laid out as device.cuh lays it out; the file's header says where the CUDA side is checked instead.
void upload_decoder_to_device(be::Device* d, const Model& m)
{
    harness("upload_decoder_to_device", d, m, true, [&] { nntc_cuda::upload_decoder_to_device(d->cuda, m); },
            [&](Model& mc) { nntc_cpu::upload_decoder_to_device(d->cpu, mc); },
            [&](Diff& diff, const Model&) {
                std::vector<float> w(d->cpu->weights.begin(),
                                     d->cpu->weights.begin() + (std::ptrdiff_t)std::min(m.dec.w.size(),
                                                                                        d->cpu->weights.size()));
                std::vector<float> b(d->cpu->bias.begin(),
                                     d->cpu->bias.begin() + (std::ptrdiff_t)std::min(m.dec.b.size(),
                                                                                     d->cpu->bias.size()));
                diff.bits("the cpu model's weights against the Model's", m.dec.w, w);
                diff.bits("the cpu model's bias against the Model's", m.dec.b, b);
            });
}

void upload_grids_to_device(be::Device* d, const Model& m)
{
    harness("upload_grids_to_device", d, m, true, [&] { nntc_cuda::upload_grids_to_device(d->cuda, m); },
            [&](Model& mc) { nntc_cpu::upload_grids_to_device(d->cpu, mc); },
            [&](Diff& diff, const Model&) {
                std::vector<float> pal((size_t)nntc_cpu::MAX_CHANNELS * 16, 0.0f);
                for (int c = 0; c < m.c0 && c < (int)m.palette0.size(); c++)
                    for (size_t k = 0; k < m.palette0[(size_t)c].size() && k < 16; k++)
                        pal[(size_t)c * 16 + k] = m.palette0[(size_t)c][k];
                diff.bits("the cpu model's palette against the Model's", pal, d->cpu->palette);
                std::vector<float> lo((size_t)nntc_cpu::MAX_CHANNELS, 0.0f), hi = lo;
                for (int c = 0; c < m.c1 && c < (int)m.lo1.size(); c++)
                {
                    lo[(size_t)c] = m.lo1[(size_t)c];
                    hi[(size_t)c] = m.hi1[(size_t)c];
                }
                diff.bits("the cpu model's lo1 against the Model's", lo, d->cpu->lo1);
                diff.bits("the cpu model's hi1 against the Model's", hi, d->cpu->hi1);
                if (!d->cpu->lo0.empty())
                {
                    std::fill(lo.begin(), lo.end(), 0.0f);
                    std::fill(hi.begin(), hi.end(), 0.0f);
                    for (int c = 0; c < m.c0 && c < (int)m.lo0.size(); c++)
                    {
                        lo[(size_t)c] = m.lo0[(size_t)c];
                        hi[(size_t)c] = m.hi0[(size_t)c];
                    }
                    diff.bits("the cpu model's lo0 against the Model's", lo, d->cpu->lo0);
                    diff.bits("the cpu model's hi0 against the Model's", hi, d->cpu->hi0);
                }
            });
}

void level1_download(be::Device* d, const Model& m, int plane, std::vector<float>& v)
{
    std::vector<float> vc;
    harness("level1_download", d, m, true, [&] { nntc_cuda::level1_download(d->cuda, m, plane, v); },
            [&](Model& mc) { nntc_cpu::level1_download(d->cpu, mc, plane, vc); },
            [&](Diff& diff, const Model&) { diff.bits("the downloaded v1", v, vc); });
}

void level1_delta(be::Device* d, const Model& m, int plane, std::vector<double>& delta)
{
    std::vector<double> dc;
    harness("level1_delta", d, m, true, [&] { nntc_cuda::level1_delta(d->cuda, m, plane, delta); },
            [&](Model& mc) { nntc_cpu::level1_delta(d->cpu, mc, plane, dc); },
            [&](Diff& diff, const Model&) { diff.bits("the downloaded correction", delta, dc); });
}

void level1_poke(be::Device* d, int plane, size_t index, float value)
{
    harness("level1_poke", d, *g_model, true, [&] { nntc_cuda::level1_poke(d->cuda, plane, index, value); },
            [&](Model&) { nntc_cpu::level1_poke(d->cpu, plane, index, value); }, [](Diff&, const Model&) {});
}

void level1_range(be::Device* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi)
{
    std::vector<float> loc, hic;
    harness("level1_range", d, m, true, [&] { nntc_cuda::level1_range(d->cuda, m, plane, lo, hi); },
            [&](Model& mc) { nntc_cpu::level1_range(d->cpu, mc, plane, loc, hic); },
            [&](Diff& diff, const Model&) {
                diff.bits("the range's lo", lo, loc);
                diff.bits("the range's hi", hi, hic);
            });
}

namespace
{

// main.cpp's finite_difference_probe, on the CPU arm alone: a central difference of the plane's own E in one level-1
// value, the realised float step as the step, the same fixed sequence of probed values. Its objective passes are counted
// so that the pass-count comparison (objective_passes) can leave them out: the CUDA arm makes none of them.
double own_fd_probe(nntc_cpu::CpuModel* c, const Model& m, int k, int plane, double h, int samples)
{
    const std::vector<double> per_site(m.planes.size(), 1.0);   // a plane's own E does not read the mip weights
    Objective base, up, down;
    nntc_cpu::objective_eval(c, m, k, per_site, base);
    g_own_passes++;
    const double scale = base.e_plane[(size_t)plane];
    std::vector<float> cur;
    nntc_cpu::level1_download(c, m, plane, cur);
    if (cur.empty() || !(scale > 0.0))
        return 0.0;
    uint64_t state = 0x9E3779B97F4A7C15ull;
    double worst = 0.0;
    for (int i = 0; i < samples; i++)
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const size_t index = (size_t)((state >> 16) % (uint64_t)cur.size());
        const float base_v = cur[index];
        const float plus = (float)((double)base_v + h), minus = (float)((double)base_v - h);
        const double step = (double)plus - (double)minus;
        nntc_cpu::level1_poke(c, plane, index, plus);
        nntc_cpu::objective_eval(c, m, k, per_site, up);
        nntc_cpu::level1_poke(c, plane, index, minus);
        nntc_cpu::objective_eval(c, m, k, per_site, down);
        nntc_cpu::level1_poke(c, plane, index, base_v);
        g_own_passes += 2;
        if (!(step > 0.0))
            continue;
        const double derivative = (up.e_plane[(size_t)plane] - down.e_plane[(size_t)plane]) / step;
        worst = std::max(worst, std::fabs(derivative) * h / scale);
    }
    return worst;
}

// The three block (b) arms on one plane of the CPU arm's own last solve (g_own above), counted as a call of their own.
// The CPU model is put at the CPU arm's own answer for the plane first; the next harnessed call mirrors every plane
// from the CUDA arm again, so nothing of this outlives it.
void own_terms(be::Device* d, const Model& m, int k, int plane, int max_unknowns)
{
    if (!g_own.have || (size_t)plane >= g_own.rep.size())
        return;
    Tally& t = tally("block (b) arms, cpu only");
    t.calls++;
    Diff diff(false);
    const Level1Report& r = g_own.rep[(size_t)plane];
    const std::string p = plane_label((size_t)plane, "");
    g_own_planes++;
    g_own_resid = std::max(g_own_resid, r.residual);
    g_own_iterations = std::max(g_own_iterations, r.iterations);
    const double bar = r.iterations >= 200 ? OWN_RESIDUAL_CAPPED : CG_STOP;
    if (!(r.residual < bar))
        diff.mismatch(p + "the CG residual " + std::to_string(r.residual) + " after " +
                      std::to_string(r.iterations) + " iterations is not under its bar");

    std::vector<double> dense;
    double worst = 0.0, lo = 1e300, hi = -1e300;
    const bool solved = nntc_cpu::solve_level1_dense_host(d->cpu, m, k, plane, r.lambda, g_own.c_prev[(size_t)plane],
                                                          dense, max_unknowns);
    if (solved)
    {
        const std::vector<double>& cg = g_own.delta[(size_t)plane];
        for (size_t i = 0; i < dense.size() && i < cg.size(); i++)
        {
            lo = std::min(lo, dense[i]);
            hi = std::max(hi, dense[i]);
            worst = std::max(worst, std::fabs(cg[i] - dense[i]));
        }
        const double rel = hi > lo ? worst / (hi - lo) : 0.0;
        g_own_dense++;
        g_own_dense_worst = std::max(g_own_dense_worst, rel);
        if (!(rel < OWN_DENSE))
            diff.mismatch(p + "the dense direct solve differs from the CG by " + std::to_string(rel) +
                          " of the range");
    }

    nntc_cpu::level1_upload(d->cpu, m, plane, g_own.v1[(size_t)plane]);
    const double fd = own_fd_probe(d->cpu, m, k, plane, 1e-3, 20);
    g_own_fd = std::max(g_own_fd, fd);
    if (!(fd < OWN_FD))
        diff.mismatch(p + "the finite-difference gradient is " + std::to_string(fd));
    char dense_text[128] = "skipped, the plane has too many unknowns";
    if (solved)
        std::snprintf(dense_text, sizeof(dense_text), "|delta cg - delta dense| %.3e over a range of %.6e", worst,
                      hi - lo);
    std::printf("own terms  M%d: CG residual %.3e after %d iterations; dense %s; fd %.3e\n", plane, r.residual,
                r.iterations, dense_text, fd);
    record(t, diff);
}

}   // namespace

bool solve_level1_dense_host(be::Device* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns)
{
    bool solved = false, solved_cpu = false;
    std::vector<double> dc;
    harness("solve_level1_dense_host", d, m, false,
            [&] { solved = nntc_cuda::solve_level1_dense_host(d->cuda, m, k, plane, lambda, c_prev, delta, max_unknowns); },
            [&](Model& mc) {
                solved_cpu = nntc_cpu::solve_level1_dense_host(d->cpu, mc, k, plane, lambda, c_prev, dc, max_unknowns);
            },
            [&](Diff& diff, const Model&) {
                diff.exact("solved", solved, solved_cpu);
                diff.values("the dense correction", delta, dc);
            });
    if (nntc_cpu::implemented("solve_level1_all") && nntc_cpu::implemented("solve_level1_dense_host"))
        own_terms(d, m, k, plane, max_unknowns);
    return solved;
}

namespace
{

// The colour class of texel (x, y) - or of block (x, y) on the block grid - in the four-colour schemes: bit 0 is the
// column's parity and bit 1 the row's, the (colour & 1, colour >> 1) of every colour pass.
int colour_of(int x, int y)
{
    return (x & 1) | ((y & 1) << 1);
}

// Block (c) one colour pass at a time (docs/CPU_BACKEND_PLAN.md stage C6: "k0 and v0 after each of the four colour
// passes"). A pass writes only the texels of its colour, so the state CUDA's pass q started from is the state before
// the call with the texels of the colours before q taken from the state after it, and what pass q produced is the same
// with colour q taken too. So each CPU pass is run from exactly CUDA's input to that pass - whatever an earlier pass of
// either arm did - and compared with exactly CUDA's output of it: the texels and their indices, and the pass's moved
// count against CUDA's own count for that colour. Counted under a tally of its own, one call per plane and colour.
void level0_colour_passes(be::Device* d, const Model& m, int k, int sweeps,
                          const std::vector<std::vector<float>>& v0_before,
                          const std::vector<std::vector<uint8_t>>& k0_before, const std::vector<Level0Report>& rep)
{
    Tally& t = tally("block (c) colour passes");
    const bool joint = nntc_cpu::level0_search_joint(m);
    for (size_t i = 0; i < m.planes.size() && i < rep.size(); i++)
    {
        const int w0 = m.planes[i].w0, h0 = m.planes[i].h0, c0 = m.c0;
        std::vector<float> v_after;
        std::vector<uint8_t> k_after;
        nntc_cuda::level0_download_values(d->cuda, m, (int)i, v_after);
        nntc_cuda::level0_download(d->cuda, m, (int)i, k_after);
        std::vector<float> v_in = v0_before[i], v_out, v_cpu;
        std::vector<uint8_t> k_in = k0_before[i], k_out, k_cpu;
        for (int q = 0; q < 4; q++)
        {
            t.calls++;
            v_out = v_in;
            k_out = k_in;
            for (int y = 0; y < h0; y++)
                for (int x = 0; x < w0; x++)
                    if (colour_of(x, y) == q)
                        for (int c = 0; c < c0; c++)
                        {
                            const size_t e = ((size_t)y * w0 + x) * (size_t)c0 + (size_t)c;
                            v_out[e] = v_after[e];
                            k_out[e] = k_after[e];
                        }
            nntc_cpu::level0_upload(d->cpu, m, (int)i, v_in);
            nntc_cpu::level0_upload_indices(d->cpu, m, (int)i, k_in);
            const unsigned int moved = nntc_cpu::level0_search_pass(d->cpu, m, k, sweeps, joint, (int)i, q);
            nntc_cpu::level0_download_values(d->cpu, m, (int)i, v_cpu);
            nntc_cpu::level0_download(d->cpu, m, (int)i, k_cpu);
            Diff diff(false);
            const std::string what = plane_label(i, "colour ") + std::to_string(q);
            diff.indices(what + " k0", k_out, k_cpu);
            diff.values(what + " v0", v_out, v_cpu, &k_out, &k_cpu);
            diff.count(what + " moved", rep[i].moved[q], (long long)moved);
            record(t, diff);
            v_in = v_out;
            k_in = k_out;
        }
    }
}

}   // namespace

double solve_level0_all(be::Device* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep)
{
    double ms = 0.0;
    std::vector<Level0Report> rc;
    // The planes as the call found them, for the colour-by-colour comparison after it.
    const bool passes = nntc_cpu::implemented("solve_level0_all");
    std::vector<std::vector<float>> v0_before(m.planes.size());
    std::vector<std::vector<uint8_t>> k0_before(m.planes.size());
    for (size_t i = 0; passes && i < m.planes.size(); i++)
    {
        nntc_cuda::level0_download_values(d->cuda, m, (int)i, v0_before[i]);
        nntc_cuda::level0_download(d->cuda, m, (int)i, k0_before[i]);
    }
    harness("solve_level0_all", d, m, false, [&] { ms = nntc_cuda::solve_level0_all(d->cuda, m, k, sweeps, rep); },
            [&](Model& mc) { nntc_cpu::solve_level0_all(d->cpu, mc, k, sweeps, rc); },
            [&](Diff& diff, const Model&) { compare_level0_reports(diff, rep, rc); });
    if (passes)
        level0_colour_passes(d, m, k, sweeps, v0_before, k0_before, rep);
    return ms;
}

void level0_download(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& k0)
{
    std::vector<uint8_t> kc;
    harness("level0_download", d, m, true, [&] { nntc_cuda::level0_download(d->cuda, m, plane, k0); },
            [&](Model& mc) { nntc_cpu::level0_download(d->cpu, mc, plane, kc); },
            [&](Diff& diff, const Model&) { diff.bits("the downloaded k0", k0, kc); });
}

void level0_upload(be::Device* d, const Model& m, int plane, const std::vector<float>& v0)
{
    harness("level0_upload", d, m, true, [&] { nntc_cuda::level0_upload(d->cuda, m, plane, v0); },
            [&](Model& mc) { nntc_cpu::level0_upload(d->cpu, mc, plane, v0); }, [](Diff&, const Model&) {});
}

void level0_download_values(be::Device* d, const Model& m, int plane, std::vector<float>& v0)
{
    std::vector<float> vc;
    harness("level0_download_values", d, m, true, [&] { nntc_cuda::level0_download_values(d->cuda, m, plane, v0); },
            [&](Model& mc) { nntc_cpu::level0_download_values(d->cpu, mc, plane, vc); },
            [&](Diff& diff, const Model&) { diff.bits("the downloaded v0", v0, vc); });
}

// ---------------------------------------------------------------------------------------------------------------------
// The objective and the decode
// ---------------------------------------------------------------------------------------------------------------------

// A decoded byte one step off is a rounding of a value that sat on a half, which is a FLIP; indices() says exactly that.
void decode_plane(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& rgb8)
{
    std::vector<uint8_t> rc;
    harness("decode_plane", d, m, false, [&] { nntc_cuda::decode_plane(d->cuda, m, plane, rgb8); },
            [&](Model& mc) { nntc_cpu::decode_plane(d->cpu, mc, plane, rc); },
            [&](Diff& diff, const Model&) { diff.indices("the decoded bytes", rgb8, rc); });
}

// Every CUDA objective pass is also measured against its own fp64 host twin: the calibration the summed-scalar bar is
// anchored on, re-taken on every run the harness makes (the twin only reads the device, so the encode cannot notice).
void objective_eval(be::Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    Objective rc;
    harness("objective_eval", d, m, false, [&] { nntc_cuda::objective_eval(d->cuda, m, k, per_site, r); },
            [&](Model& mc) { nntc_cpu::objective_eval(d->cpu, mc, k, per_site, rc); },
            [&](Diff& diff, const Model&) { compare_objective(diff, r, rc); });

    // The twin reads the decoder from the host Model, which is empty until the first block (a) has filled it (the
    // device's decoder starts at zeros, the Model's at nothing), so the passes before that are not measured.
    if (m.dec.w.size() != (size_t)m.dec.nin * m.nout || m.dec.b.size() != (size_t)m.nout)
        return;
    auto worst_of = [](const Objective& a, const Objective& b, double worst) {
        auto rel = [](double x, double y) {
            const double s = std::max(std::fabs(x), std::fabs(y));
            return s > 0.0 ? std::fabs(x - y) / s : 0.0;
        };
        worst = std::max(worst, rel(a.e, b.e));
        worst = std::max(worst, rel(a.centre_mse, b.centre_mse));
        worst = std::max(worst, rel(a.sampled_mse, b.sampled_mse));
        for (size_t i = 0; i < a.e_plane.size() && i < b.e_plane.size(); i++)
            worst = std::max(worst, rel(a.e_plane[i], b.e_plane[i]));
        return worst;
    };
    Objective twin;
    nntc_cuda::objective_check_host(d->cuda, m, k, per_site, twin);
    g_calib_worst = worst_of(r, twin, g_calib_worst);
    g_calib_passes++;

    // The same pair on the CPU arm: its objective (rc, run on the CUDA arm's state) against its own copy of the twin on
    // that same state. Stage C2's acceptance holds this pair to the bar the gate holds the CUDA pair to, and it is
    // re-measured here on every real image the harness runs, beside the synthetic one nntc_cpu_check holds it on.
    if (!nntc_cpu::implemented("objective_eval") || !nntc_cpu::implemented("objective_check_host"))
        return;
    Objective twin_cpu;
    nntc_cpu::objective_check_host(d->cpu, m, k, per_site, twin_cpu);
    g_calib_cpu_worst = worst_of(rc, twin_cpu, g_calib_cpu_worst);
    g_calib_cpu_passes++;
}

// The run's objective time is a report figure on another clock, and the report is CUDA's: it is never compared, and it
// counts as ok once the CPU arm has one to give.
double objective_total_ms()
{
    Tally& t = tally("objective_total_ms");
    t.calls++;
    if (nntc_cpu::implemented("objective_total_ms"))
        t.ok++;
    else
        t.not_implemented++;
    return nntc_cuda::objective_total_ms();
}

// The pass count is the CPU arm's own passes, which under the harness are exactly the CUDA arm's.

long long objective_passes()
{
    const long long a = nntc_cuda::objective_passes();
    Tally& t = tally("objective_passes");
    t.calls++;
    if (!nntc_cpu::implemented("objective_passes"))
    {
        t.not_implemented++;
        return a;
    }
    Diff diff(true);
    diff.exact("passes", a, nntc_cpu::objective_passes() - g_own_passes);
    record(t, diff);
    return a;
}

void objective_check_host(be::Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    Objective rc;
    harness("objective_check_host", d, m, false, [&] { nntc_cuda::objective_check_host(d->cuda, m, k, per_site, r); },
            [&](Model& mc) { nntc_cpu::objective_check_host(d->cpu, mc, k, per_site, rc); },
            [&](Diff& diff, const Model&) { compare_objective(diff, r, rc); });
}

// ---------------------------------------------------------------------------------------------------------------------
// The BC refinement
// ---------------------------------------------------------------------------------------------------------------------

// The pack's call compares, besides the planes, the BC state and the Model's blocks, the level-0 workspace the pack's
// own assembly built (the stencil and gradient to BAR_ASSEMBLED, the plane the correction is measured from exactly)
// and the correction k_bc_decode wrote. The CPU arm assembles from the mirrored planes itself: the assembly is part of
// the call on both arms.
double bc_pack_prepare(be::Device* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels)
{
    double ms = 0.0;
    double psnr_c = pack_psnr;
    size_t texels_c = pack_texels;
    harness("bc_pack_prepare", d, m, false,
            [&] { ms = nntc_cuda::bc_pack_prepare(d->cuda, m, k, seed, pack_psnr, pack_texels); },
            [&](Model& mc) { nntc_cpu::bc_pack_prepare(d->cpu, mc, k, seed, psnr_c, texels_c); },
            [&](Diff& diff, const Model&) {
                diff.scalar("the packing psnr", pack_psnr, psnr_c, false);
                diff.exact("the packing texels", (long long)pack_texels, (long long)texels_c);
                if (nntc_cpu::implemented("bc_pack_prepare"))
                    compare_workspace(diff, d, m, true, false, true);
            });
    return ms;
}

namespace
{

// The BC refinement's inputs and outputs on one arm, per plane: the endpoints and selectors, and the plane's values,
// indices and correction (the correction is level 0's conjugate-gradient buffer, cg0_x, which k_bc_decode and
// k_bc_refine write; the probe reads it on the CUDA arm).
struct BcPlanes
{
    std::vector<std::vector<uint8_t>> ep, sel, k0;
    std::vector<std::vector<float>> v0;
    std::vector<std::vector<double>> delta;
};

void bc_planes_cuda(DeviceModel* d, const Model& m, BcPlanes& s)
{
    const size_t np = m.planes.size();
    nntc_cuda::bc_state_save(d, m, s.ep, s.sel);
    s.v0.assign(np, {});
    s.k0.assign(np, {});
    s.delta.assign(np, {});
    for (size_t i = 0; i < np; i++)
    {
        nntc_cuda::level0_download_values(d, m, (int)i, s.v0[i]);
        nntc_cuda::level0_download(d, m, (int)i, s.k0[i]);
        nntc_probe::Workspace w;
        nntc_probe::workspace(d, m, (int)i, true, w);
        s.delta[i] = w.cg_x;
    }
}

// The CUDA arm's level-0 workspace written into the CPU model: the stencil, the gradient and the plane the correction
// is measured from, which the refinement reads and bc_pack_prepare's assembly wrote. The harness's mirror carries the
// planes and the BC state; a refinement also reads these, and they are CUDA's own, so the CPU refinement is judged on
// exactly the quadratic the CUDA one worked on.
void mirror_level0_workspace(const be::Device* d, const Model& m, const std::vector<nntc_probe::Workspace>& ws)
{
    for (size_t i = 0; i < m.planes.size() && i < ws.size(); i++)
    {
        nntc_cpu::CpuPlane& p = d->cpu->planes[i];
        if (ws[i].stencil.size() == p.stencil0.size())
            std::copy(ws[i].stencil.begin(), ws[i].stencil.end(), p.stencil0.begin());
        if (ws[i].grad.size() == p.grad0.size())
            std::copy(ws[i].grad.begin(), ws[i].grad.end(), p.grad0.begin());
        if (ws[i].prev.size() == p.q0_prev.size())
            std::copy(ws[i].prev.begin(), ws[i].prev.end(), p.q0_prev.begin());
        if (ws[i].cg_x.size() == p.cg0_x.size())
            std::copy(ws[i].cg_x.begin(), ws[i].cg_x.end(), p.cg0_x.begin());
    }
}

// One pass of the BC refinement one colour at a time (stage C7: "the endpoints and selectors after every refinement
// pass", taken one step finer). A block of one colour writes only its own endpoints, selectors and texels, so the state
// CUDA's colour q started from is the state before the pass with the blocks of the colours before q taken from the
// state after it; each CPU colour is run from exactly that and compared with the same state with colour q taken too.
// Counted under a tally of its own, one call per plane and colour.
void bc_colour_passes(const be::Device* d, const Model& m, const BcPlanes& before, const BcPlanes& after,
                      const std::vector<nntc_probe::Workspace>& ws)
{
    Tally& t = tally("bc refinement colour passes");
    const int c0 = m.c0;
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const int w0 = m.planes[i].w0, h0 = m.planes[i].h0;
        const int bx = (w0 + 3) / 4, by = (h0 + 3) / 4;
        BcPlanes in, out;
        in.ep = { before.ep[i] };
        in.sel = { before.sel[i] };
        in.v0 = { before.v0[i] };
        in.k0 = { before.k0[i] };
        in.delta = { before.delta[i] };
        for (int q = 0; q < 4; q++)
        {
            t.calls++;
            out = in;
            for (int gy = 0; gy < by; gy++)
                for (int gx = 0; gx < bx; gx++)
                {
                    if (colour_of(gx, gy) != q)
                        continue;
                    const size_t gi = (size_t)gy * bx + gx;
                    for (size_t e = gi * (size_t)c0 * 2; e < (gi + 1) * (size_t)c0 * 2; e++)
                        out.ep[0][e] = after.ep[i][e];
                    for (size_t e = gi * (size_t)c0 * 16; e < (gi + 1) * (size_t)c0 * 16; e++)
                        out.sel[0][e] = after.sel[i][e];
                    for (int y = gy * 4; y < gy * 4 + 4 && y < h0; y++)
                        for (int x = gx * 4; x < gx * 4 + 4 && x < w0; x++)
                            for (int c = 0; c < c0; c++)
                            {
                                const size_t e = ((size_t)y * w0 + x) * (size_t)c0 + (size_t)c;
                                out.v0[0][e] = after.v0[i][e];
                                out.k0[0][e] = after.k0[i][e];
                                out.delta[0][e] = after.delta[i][e];
                            }
                }
            // The CPU plane put at CUDA's input to this colour: the workspace as CUDA's pass found it, then the
            // blocks, the plane and the correction.
            mirror_level0_workspace(d, m, ws);
            nntc_cpu::CpuPlane& p = d->cpu->planes[i];
            std::copy(in.ep[0].begin(), in.ep[0].end(), p.bc_ep.begin());
            std::copy(in.sel[0].begin(), in.sel[0].end(), p.bc_sel.begin());
            std::copy(in.v0[0].begin(), in.v0[0].end(), p.v0.begin());
            std::copy(in.k0[0].begin(), in.k0[0].end(), p.k0.begin());
            std::copy(in.delta[0].begin(), in.delta[0].end(), p.cg0_x.begin());
            double taken = 0.0, improved = 0.0;
            nntc_cpu::bc_refine_colour(d->cpu, m, (int)i, q, taken, improved);

            Diff diff(false);
            const std::string what = plane_label(i, "colour ") + std::to_string(q);
            const std::vector<uint8_t> ep_c(p.bc_ep.begin(), p.bc_ep.begin() + (std::ptrdiff_t)out.ep[0].size());
            const std::vector<uint8_t> sel_c(p.bc_sel.begin(), p.bc_sel.begin() + (std::ptrdiff_t)out.sel[0].size());
            diff.indices(what + " BC endpoints", out.ep[0], ep_c);
            diff.indices(what + " BC selectors", out.sel[0], sel_c);
            diff.indices(what + " k0", out.k0[0], p.k0);
            diff.values(what + " v0", out.v0[0], p.v0, &out.k0[0], &p.k0);
            diff.values(what + " level-0 correction", out.delta[0], p.cg0_x);
            record(t, diff);
            in = out;
        }
    }
}

}   // namespace

// The refinement one pass at a time, on both arms. main asks for `passes` passes in one call; the CUDA arm's bc_refine
// runs the same kernels in the same order whether it is asked for N passes once or for one pass N times (each pass is
// four launches per plane from the state the last left, and the figures are per pass), so here it is asked N times,
// and every pass is a harnessed call of its own: the CPU arm is handed CUDA's state before the pass - the planes and the
// BC state through the mirror, the level-0 workspace through mirror_level0_workspace - runs one pass, and is compared
// with CUDA's state after it; then the same pass is taken one colour at a time (bc_colour_passes). The report CUDA
// hands back is the one main would have had, pass numbers included.
double bc_refine(be::Device* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels)
{
    if (!nntc_cpu::implemented("bc_refine") || passes <= 0)
    {
        double ms = 0.0;
        std::vector<BcRefineReport> rc;
        double psnr_c = pack_psnr;
        size_t texels_c = pack_texels;
        harness("bc_refine", d, m, false,
                [&] { ms = nntc_cuda::bc_refine(d->cuda, m, passes, rep, pack_psnr, pack_texels); },
                [&](Model& mc) { nntc_cpu::bc_refine(d->cpu, mc, passes, rc, psnr_c, texels_c); },
                [&](Diff& diff, const Model&) {
                    compare_bc_reports(diff, rep, rc);
                    diff.scalar("the packing psnr", pack_psnr, psnr_c, false);
                    diff.exact("the packing texels", (long long)pack_texels, (long long)texels_c);
                });
        return ms;
    }
    double ms = 0.0;
    rep.clear();
    const size_t np = m.planes.size();
    for (int pass = 0; pass < passes; pass++)
    {
        std::vector<nntc_probe::Workspace> ws(np);
        for (size_t i = 0; i < np; i++)
            nntc_probe::workspace(d->cuda, m, (int)i, true, ws[i]);
        BcPlanes before, after;
        bc_planes_cuda(d->cuda, m, before);

        std::vector<BcRefineReport> one, rc;
        double psnr_c = pack_psnr;
        size_t texels_c = pack_texels;
        harness("bc_refine", d, m, false,
                [&] { ms += nntc_cuda::bc_refine(d->cuda, m, 1, one, pack_psnr, pack_texels); },
                [&](Model& mc) {
                    mirror_level0_workspace(d, m, ws);
                    nntc_cpu::bc_refine(d->cpu, mc, 1, rc, psnr_c, texels_c);
                },
                [&](Diff& diff, const Model&) {
                    compare_bc_reports(diff, one, rc);
                    diff.scalar("the packing psnr", pack_psnr, psnr_c, false);
                    diff.exact("the packing texels", (long long)pack_texels, (long long)texels_c);
                    for (size_t i = 0; i < np; i++)
                    {
                        nntc_probe::Workspace w;
                        nntc_probe::workspace(d->cuda, m, (int)i, true, w);
                        const std::vector<double>& x = d->cpu->planes[i].cg0_x;
                        diff.values(plane_label(i, "level-0 correction"), w.cg_x, x);
                    }
                });
        bc_planes_cuda(d->cuda, m, after);
        bc_colour_passes(d, m, before, after, ws);
        for (BcRefineReport& r : one)
        {
            r.pass = pass + 1;
            rep.push_back(r);
        }
    }
    return ms;
}

void bc_blocks_from_device(be::Device* d, Model& m, double& pack_psnr, size_t& pack_texels)
{
    double psnr_c = pack_psnr;
    size_t texels_c = pack_texels;
    harness("bc_blocks_from_device", d, m, false,
            [&] { nntc_cuda::bc_blocks_from_device(d->cuda, m, pack_psnr, pack_texels); },
            [&](Model& mc) { nntc_cpu::bc_blocks_from_device(d->cpu, mc, psnr_c, texels_c); },
            [&](Diff& diff, const Model&) {
                diff.scalar("the packing psnr", pack_psnr, psnr_c, false);
                diff.exact("the packing texels", (long long)pack_texels, (long long)texels_c);
            });
}

void bc_state_save(const be::Device* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel)
{
    std::vector<std::vector<uint8_t>> ec, sc;
    harness("bc_state_save", d, m, true, [&] { nntc_cuda::bc_state_save(d->cuda, m, ep, sel); },
            [&](Model& mc) { nntc_cpu::bc_state_save(d->cpu, mc, ec, sc); },
            [&](Diff& diff, const Model&) {
                diff.exact("the saved plane count", (long long)ep.size(), (long long)ec.size());
                for (size_t i = 0; i < ep.size() && i < ec.size(); i++)
                {
                    diff.bits(plane_label(i, "saved endpoints"), ep[i], ec[i]);
                    diff.bits(plane_label(i, "saved selectors"), sel[i], sc[i]);
                }
            });
}

void bc_state_restore(be::Device* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel)
{
    harness("bc_state_restore", d, m, true, [&] { nntc_cuda::bc_state_restore(d->cuda, m, ep, sel); },
            [&](Model& mc) { nntc_cpu::bc_state_restore(d->cpu, mc, ep, sel); }, [](Diff&, const Model&) {});
}

}   // namespace nntc_check
