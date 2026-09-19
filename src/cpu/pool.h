// cpu/pool.h: the CPU backend's thread pool, and the contract that makes its results independent of the thread count.
//
// THE CONTRACT (docs/CPU_BACKEND_PLAN.md section 3.1). Every rule is checkable by reading a call site:
//
//   1. `chunks` is a property of the KERNEL, never of the thread count. A kernel passes chunks_for(n, CPU_CHUNK) or a
//      fixed grid; Pool::run never sees threads() in its first argument.
//   2. The chunk-to-element map is fixed: chunk c covers the same elements at every -j, on every run, on every machine.
//   3. Static partition, no work stealing. Of P participants (the caller and P - 1 workers, P = min(threads, chunks)),
//      participant p takes the contiguous chunk range partition() gives it, computed once per run.
//   4. Every reduction is folded over chunks in increasing c, AFTER the join (fold_chunks below). No partial is written
//      by one thread and read by another while the run is live.
//   5. -j1 is not a separate code path. With one thread there are no workers, P is 1, and the calling thread walks
//      [0, chunks) through the SAME walk() every worker uses. The review of the plan found that an inline fast path
//      for -j1 is the adversarial case - different control flow is where a divergence hides - so there is none.
//   6. No nesting. One run is live at a time; a body that calls run is refused by name.
//
// So a kernel's arithmetic cannot notice how many threads ran it: the chunk bodies are the same, each writes only its
// own outputs, and the fold order is the chunk order. Determinism WITHIN the CPU backend (repeat runs byte-identical,
// -j1 and -j32 byte-identical) is therefore a property of the call sites, not of the scheduler.
//
// THE HAND-OFF: SPIN, THEN PARK. The first version parked every worker on a condition variable between runs, and stage
// C1 measured what that costs: a run at -j32 took 68 us of wake-ups before any work, against the "microseconds" section
// 3.1 asks for, at a hundred thousand runs an encode. So this is the fallback the plan named - inside this file and
// nowhere else. Each worker has its own job slot and its own generation counter. The caller writes a participant's slot
// and then publishes it by storing the new generation (release); the worker, which has been watching the counter
// (acquire), reads the slot and walks. Workers that are not participants are not touched at all. A worker that sees no
// new generation for a while parks on its own condition variable, and the caller wakes it only if it said it was
// parked; the finish goes the other way, through one countdown the caller watches and parks on the same way. So a run
// between busy kernels costs a few atomic operations per worker, and an idle pool costs no CPU.
//
// ZERO DATA RACES. A slot's plain fields are written by the caller only before it publishes the generation and only
// after the slot's worker has counted itself finished (which the caller observes with acquire), and read by the worker
// only between those two events: a release/acquire pair orders each hand-off in both directions. A body's writes happen
// before its worker's release decrement of the countdown, and the caller's acquire load that sees zero happens after
// them, so everything a body wrote is visible when run returns. The parked flags use sequentially consistent atomics,
// because the park is the classic two-flag handshake (publish the generation, then read `parked`; set `parked`, then
// read the generation) and only sequential consistency guarantees that at least one side sees the other. Every shared
// variable is either an atomic or ordered by one; ThreadSanitizer is the permanent check (tests/cpu_check.cpp).
//
// AN EXCEPTION IN A BODY. A body can throw - a scratch std::vector that cannot be had throws std::bad_alloc - and one
// escaping a worker thread would end the process through std::terminate with no ERROR line. So every walk catches what
// its body throws and stops its own share there; the other participants finish theirs, and after the join run rethrows
// on the calling thread, to reach main's catch like any other exception. When several threw, the one rethrown is the
// lowest-numbered participant's, whatever the order they finished in. A worker's exception is a slot field it writes before its release
// decrement of the countdown and the caller reads after the acquire that sees zero, the same pair that publishes a
// body's writes, so it is ordered like them.
//
// Plain C++17, std::thread and std::atomic: no intrinsics, no pause instruction, nothing per-ISA.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace nntc_cpu
{

// The -j cap, which is the dispatcher's (backend.cpp's backend_threads) and the option parser's too.
static const int POOL_MAX_THREADS = 1024;

// How long a waiting thread watches its counter before it parks, in wall-clock time rather than in iterations, because
// a load's cost differs by an order of magnitude between machines. MEASURED on the 32-thread development machine: a
// budget of a few tens of microseconds lost the race - one late worker parked, its wake-up took tens of microseconds,
// the others' budgets ran out meanwhile, and a run at -j32 settled at 67 us, no better than parking always; a
// millisecond keeps the pool awake across the gaps between the kernels of a conjugate-gradient iteration and brings the
// run to about 3 us. An idle pool still parks within a millisecond. It changes when a thread sleeps, never what it
// computes.
static const std::chrono::microseconds POOL_SPIN{ 1000 };

// Watch `done` until it says true or the spin budget is spent; the clock is read once every 1024 looks, so the watch
// itself stays a loop of loads.
template <class Done>
inline bool pool_spin(const Done& done)
{
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 1;; i++)
    {
        if (done())
            return true;
        if ((i & 1023u) == 0 && std::chrono::steady_clock::now() - start > POOL_SPIN)
            return false;
    }
}

// -j N resolved to a thread count: 0 asks the machine, a machine that cannot answer counts as one, and the result is
// clamped into 1..POOL_MAX_THREADS. The same rule backend_threads() applies, so the pool is correct even if it is ever
// handed an unresolved count.
inline int pool_resolve_threads(int requested)
{
    int t = requested;
    if (t <= 0)
        t = (int)std::thread::hardware_concurrency();
    if (t < 1)
        t = 1;
    if (t > POOL_MAX_THREADS)
        t = POOL_MAX_THREADS;
    return t;
}

class Pool
{
public:
    explicit Pool(int threads) : threads_(pool_resolve_threads(threads))
    {
        slots_.reserve((size_t)threads_ - 1);
        for (int i = 1; i < threads_; i++)
            slots_.push_back(std::unique_ptr<Slot>(new Slot()));
        workers_.reserve((size_t)threads_ - 1);
        for (int i = 1; i < threads_; i++)
        {
            try
            {
                workers_.emplace_back(&Pool::worker_main, this, i - 1);
            }
            catch (const std::system_error& e)
            {
                // Reported, not survived: the process exits with the started workers still parked.
                std::fprintf(stderr, "ERROR: could not create worker thread %d of %d (%s): try a smaller -j\n", i + 1,
                             threads_, e.what());
                std::exit(EXIT_FAILURE);
            }
        }
    }

    ~Pool()
    {
        for (std::unique_ptr<Slot>& s : slots_)
        {
            s->quit.store(true);
            {
                std::lock_guard<std::mutex> lk(s->mu);
            }
            s->cv.notify_one();
        }
        for (std::thread& t : workers_)
            t.join();
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int threads() const { return threads_; }

    // The participants of a run of `chunks` chunks: never more than there are chunks, so a deep mip plane of one chunk
    // wakes nobody and the rest of the pool is not disturbed.
    int participants(int chunks) const { return chunks < threads_ ? chunks : threads_; }

    // The static partition: participant p of P takes [begin, end). A pure function of (chunks, P, p), contiguous,
    // covering [0, chunks) exactly once; public so that the unit exercise can check it by enumeration.
    static void partition(int chunks, int participants, int p, int& begin, int& end)
    {
        begin = (int)((long long)chunks * p / participants);
        end = (int)((long long)chunks * (p + 1) / participants);
    }

    // body(c) for every c in [0, chunks), each exactly once, spread over up to threads() threads; returns after all of
    // them have finished. A body writes only what chunk c owns.
    template <class Body>
    void run(int chunks, const Body& body)
    {
        run_erased(chunks, &body, [](const void* f, int c) { (*static_cast<const Body*>(f))(c); });
    }

private:
    using Trampoline = void (*)(const void*, int);

    struct Slot
    {
        // The job, plain fields: written by the caller before `generation` is published, read by the worker after.
        int begin = 0, end = 0;
        const void* body = nullptr;
        Trampoline call = nullptr;
        std::exception_ptr error;   // written by the worker during its walk, read by the caller after the join

        std::atomic<unsigned> generation{ 0 };   // bumped once per job handed to this worker
        std::atomic<bool> parked{ false };       // the worker is (about to be) asleep on cv
        std::atomic<bool> quit{ false };
        std::mutex mu;
        std::condition_variable cv;
    };

    // Set on a thread while it is inside a body, so that a body calling run - the nesting rule 6 forbids, and a
    // deadlock if it were allowed from a worker - is refused rather than hung. thread_local, so it is never shared.
    static bool& in_body()
    {
        static thread_local bool flag = false;
        return flag;
    }

    // The ONE chunk walk. The caller and every worker go through it, at every thread count including one. What a body
    // throws ends this participant's share and is handed back, for run to rethrow after the join.
    static std::exception_ptr walk(int begin, int end, const void* body, Trampoline call)
    {
        std::exception_ptr error;
        in_body() = true;
        try
        {
            for (int c = begin; c < end; c++)
                call(body, c);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        in_body() = false;
        return error;
    }

    void run_erased(int chunks, const void* body, Trampoline call)
    {
        if (chunks <= 0)
            return;
        if (in_body())
        {
            std::fprintf(stderr, "ERROR: the CPU backend's pool was entered from inside one of its own chunk bodies; "
                                 "runs do not nest (docs/CPU_BACKEND_PLAN.md section 3.1, rule 6): no encode\n");
            std::exit(EXIT_FAILURE);
        }
        const int np = participants(chunks);
        remaining_.store(np - 1, std::memory_order_relaxed);   // published by the generation stores below
        for (int p = 1; p < np; p++)
        {
            Slot& s = *slots_[(size_t)p - 1];
            partition(chunks, np, p, s.begin, s.end);
            s.body = body;
            s.call = call;
            if (s.error)
                s.error = nullptr;
            s.generation.fetch_add(1);   // seq_cst: publishes the job, and orders the parked check after it
            if (s.parked.load())
            {
                {
                    std::lock_guard<std::mutex> lk(s.mu);
                }
                s.cv.notify_one();
            }
        }
        // The caller is participant 0 and walks its own share while the workers walk theirs.
        int begin = 0, end = 0;
        partition(chunks, np, 0, begin, end);
        std::exception_ptr error = walk(begin, end, body, call);
        if (np > 1 && !pool_spin([this] { return remaining_.load(std::memory_order_acquire) == 0; }))
        {
            std::unique_lock<std::mutex> lk(done_mu_);
            caller_parked_.store(true);
            done_cv_.wait(lk, [this] { return remaining_.load() == 0; });
            caller_parked_.store(false);
        }
        // Joined: every worker's walk has ended, so an exception may leave now without a body still running on it.
        for (int p = 1; p < np && !error; p++)
            error = slots_[(size_t)p - 1]->error;
        if (error)
            std::rethrow_exception(error);
    }

    void worker_main(int slot)
    {
        Slot& s = *slots_[(size_t)slot];
        unsigned seen = 0;
        for (;;)
        {
            unsigned g = seen;
            pool_spin([&] {
                g = s.generation.load(std::memory_order_acquire);
                return g != seen || s.quit.load(std::memory_order_relaxed);
            });
            if (g == seen && s.quit.load())
                return;
            if (g == seen)
            {
                std::unique_lock<std::mutex> lk(s.mu);
                s.parked.store(true);   // seq_cst: set before the generation is looked at again
                s.cv.wait(lk, [&] { return s.generation.load() != seen || s.quit.load(); });
                s.parked.store(false);
                g = s.generation.load();
            }
            if (g == seen)
                return;   // woken with no job: the pool is being destroyed
            seen = g;
            s.error = walk(s.begin, s.end, s.body, s.call);
            if (remaining_.fetch_sub(1) == 1 && caller_parked_.load())   // seq_cst, the other two-flag handshake
            {
                {
                    std::lock_guard<std::mutex> lk(done_mu_);
                }
                done_cv_.notify_one();
            }
        }
    }

    int threads_;
    std::vector<std::unique_ptr<Slot>> slots_;   // one per worker, index = participant - 1
    std::vector<std::thread> workers_;
    std::atomic<int> remaining_{ 0 };            // participating workers of the live run still walking
    std::atomic<bool> caller_parked_{ false };
    std::mutex done_mu_;
    std::condition_variable done_cv_;
};

// A reduction under rule 4: chunk(c) returns chunk c's partial and writes nothing shared; the partials land in one
// slot each (disjoint writes) and are folded in increasing c on the calling thread after the join. The chunk count is
// the caller's and must obey rule 1, which is what makes the result the same at every -j.
template <class T, class Chunk, class Fold>
T fold_chunks(Pool& pool, int chunks, T init, const Chunk& chunk, const Fold& fold)
{
    if (chunks <= 0)
        return init;
    std::vector<T> partial((size_t)chunks, init);
    pool.run(chunks, [&](int c) { partial[(size_t)c] = chunk(c); });
    T acc = init;
    for (int c = 0; c < chunks; c++)
        acc = fold(acc, partial[(size_t)c]);
    return acc;
}

}   // namespace nntc_cpu
