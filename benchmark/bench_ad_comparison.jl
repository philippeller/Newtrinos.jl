using BenchmarkTools
using DensityInterface
using Newtrinos
using Printf
import ForwardDiff
import PolyesterForwardDiff
import Mooncake

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

# ── Configure experiments ──────────────────────────────────────────

println("Configuring experiments...")

db = Newtrinos.dayabay.configure()
dc = Newtrinos.deepcore.configure()
or = Newtrinos.orca.configure()
sk = Newtrinos.super_k.configure()

configs = [
    ("Daya Bay (simple)",
     (dayabay=db,)),
    ("DeepCore (intermediate)",
     (deepcore=dc,)),
    ("DeepCore+ORCA+Super-K (combined)",
     (deepcore=dc, orca=or, super_k=sk)),
]

# ── Benchmark helpers ──────────────────────────────────────────────

function bench_config(name, experiments)
    params = Newtrinos.get_params(experiments)
    likelihood = Newtrinos.generate_likelihood(experiments)
    n = length(params)
    param_keys = keys(params)
    x0 = collect(Float64, values(params))

    println()
    println("=" ^ 72)
    println("  $name  —  $n parameters")
    println("=" ^ 72)

    # ── Likelihood evaluation ──────────────────────────────────────
    llh_val = logdensityof(likelihood, params)
    b_llh = @benchmark logdensityof($likelihood, $params)
    t_llh = median(b_llh).time / 1e6
    m_llh = b_llh.memory

    # ── ForwardDiff gradient ───────────────────────────────────────
    fd_f(p) = logdensityof(likelihood, p)
    t_fd_compile = @elapsed ForwardDiff.gradient(fd_f, params)
    b_fd = @benchmark ForwardDiff.gradient($fd_f, $params)
    t_fd = median(b_fd).time / 1e6
    m_fd = b_fd.memory

    # ── PolyesterForwardDiff gradient ──────────────────────────────
    poly_f(x) = logdensityof(likelihood, NamedTuple{param_keys}(Tuple(x)))
    grad_buf = similar(x0)
    chunk = ForwardDiff.Chunk{min(12, n)}()
    t_pfd_compile = @elapsed PolyesterForwardDiff.threaded_gradient!(poly_f, grad_buf, x0, chunk)
    b_pfd = @benchmark PolyesterForwardDiff.threaded_gradient!($poly_f, $grad_buf, $x0, $chunk)
    t_pfd = median(b_pfd).time / 1e6
    m_pfd = b_pfd.memory

    # ── Mooncake gradient ──────────────────────────────────────────
    mc_f(x) = Float64(-logdensityof(likelihood, NamedTuple{param_keys}(Tuple(x))))
    t_mc_compile = @elapsed begin
        rule = Mooncake.build_rrule(mc_f, x0)
    end
    # warmup eval
    Mooncake.value_and_gradient!!(rule, mc_f, x0)
    b_mc = @benchmark Mooncake.value_and_gradient!!($rule, $mc_f, $x0)
    t_mc = median(b_mc).time / 1e6
    m_mc = b_mc.memory

    # ── Print results ──────────────────────────────────────────────
    println()
    @printf("  %-28s %12s %12s %12s\n", "", "Time (ms)", "Memory (KB)", "vs LLH")
    println("  ", "-"^68)
    @printf("  %-28s %12.2f %12.1f %12s\n", "Likelihood eval", t_llh, m_llh/1024, "1.0×")
    @printf("  %-28s %12.2f %12.1f %12.1f×\n", "ForwardDiff gradient", t_fd, m_fd/1024, t_fd/t_llh)
    @printf("  %-28s %12.2f %12.1f %12.1f×\n", "PolyesterForwardDiff grad.", t_pfd, m_pfd/1024, t_pfd/t_llh)
    @printf("  %-28s %12.2f %12.1f %12.1f×\n", "Mooncake gradient", t_mc, m_mc/1024, t_mc/t_llh)

    println()
    @printf("  %-28s %12s\n", "", "Compile (s)")
    println("  ", "-"^42)
    @printf("  %-28s %12.1f\n", "ForwardDiff 1st gradient", t_fd_compile)
    @printf("  %-28s %12.1f\n", "PolyesterForwardDiff 1st", t_pfd_compile)
    @printf("  %-28s %12.1f\n", "Mooncake build_rrule", t_mc_compile)

    println()
    best_grad = min(t_fd, t_pfd, t_mc)
    @printf("  Fastest gradient: %.2f ms  ", best_grad)
    if best_grad == t_mc
        println("(Mooncake)")
    elseif best_grad == t_pfd
        println("(PolyesterForwardDiff)")
    else
        println("(ForwardDiff)")
    end
    @printf("  Overhead vs likelihood: %.1f×\n", best_grad / t_llh)
    @printf("  Theoretical FD minimum: %.1f× (ceil(%d/12) = %d passes)\n",
        ceil(n/12), n, Int(ceil(n/12)))

    return (; name, n, t_llh, t_fd, t_pfd, t_mc, m_llh, m_fd, m_pfd, m_mc,
              t_fd_compile, t_pfd_compile, t_mc_compile)
end

# ── Run all benchmarks ─────────────────────────────────────────────

results = []
for (name, exps) in configs
    r = bench_config(name, exps)
    push!(results, r)
end

# ── Final summary table ────────────────────────────────────────────

println()
println()
println("=" ^ 90)
println("  SUMMARY TABLE")
println("=" ^ 90)
println()
@printf("  %-35s %6s %10s %10s %10s %10s\n",
    "Configuration", "Params", "LLH (ms)", "FD (ms)", "PFD (ms)", "MC (ms)")
println("  ", "-"^85)
for r in results
    @printf("  %-35s %6d %10.2f %10.2f %10.2f %10.2f\n",
        r.name, r.n, r.t_llh, r.t_fd, r.t_pfd, r.t_mc)
end

println()
@printf("  %-35s %6s %10s %10s %10s\n",
    "Compile / startup cost (s)", "", "FD", "PFD", "Mooncake")
println("  ", "-"^72)
for r in results
    @printf("  %-35s %6d %10.1f %10.1f %10.1f\n",
        r.name, r.n, r.t_fd_compile, r.t_pfd_compile, r.t_mc_compile)
end

println()
@printf("  %-35s %6s %10s %10s %10s\n",
    "Memory per gradient (KB)", "", "FD", "PFD", "Mooncake")
println("  ", "-"^72)
for r in results
    @printf("  %-35s %6d %10.1f %10.1f %10.1f\n",
        r.name, r.n, r.m_fd/1024, r.m_pfd/1024, r.m_mc/1024)
end

println()
println("  Threads: $(Threads.nthreads())")
println()
