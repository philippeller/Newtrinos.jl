using ArgParse
using BenchmarkTools
using DensityInterface
using Newtrinos
import ForwardDiff
import PolyesterForwardDiff

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments"
        help = "List of experiments to run"
        nargs = '+'
        required = true
    end

    return parse_args(s)
end

args = parse_command_line()

println("Configuring experiments...")
function configure_experiments(experiment_list)
    pairs = (Symbol(lowercase(exp)) => getproperty(getproperty(Newtrinos, Symbol(lowercase(exp))), :configure)() for exp in experiment_list)
    return (; pairs...)
end

experiments = configure_experiments(args["experiments"])

params = Newtrinos.get_params(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

println()
println("=" ^60)
println("LIKELIHOOD BENCHMARK")
println("=" ^60)
println("  Experiments: ", join(args["experiments"], ", "))
println("  Parameters:  ", length(keys(params)))
println()

# Warmup
l = logdensityof(likelihood, params)

println("llh =", l)

println("=== Likelihood evaluation ===")
b_llh = @benchmark logdensityof($likelihood, $params)
display(b_llh)
println()

# Gradient via ForwardDiff
println("=== Gradient (ForwardDiff) ===")
grad_f(p) = logdensityof(likelihood, p)
ForwardDiff.gradient(grad_f, params)  # warmup

b_grad = @benchmark ForwardDiff.gradient($grad_f, $params)
display(b_grad)
println()

# Gradient via PolyesterForwardDiff (threaded chunked ForwardDiff)
println("=== Gradient (PolyesterForwardDiff) ===")
param_keys = keys(params)
poly_f(x) = logdensityof(likelihood, NamedTuple{param_keys}(Tuple(x)))
params_vec = collect(Float64, values(params))
grad_buf = similar(params_vec)
PolyesterForwardDiff.threaded_gradient!(poly_f, grad_buf, params_vec, ForwardDiff.Chunk{12}())  # warmup

b_poly = @benchmark PolyesterForwardDiff.threaded_gradient!($poly_f, $grad_buf, $params_vec, ForwardDiff.Chunk{12}())
display(b_poly)
println()

# Summary
t_llh = median(b_llh).time / 1e6
t_grad = median(b_grad).time / 1e6
t_poly = median(b_poly).time / 1e6
n_params = length(keys(params))

println("=== Summary ===")
println("  Likelihood:          $(round(t_llh, digits=2)) ms (median)")
println("  ForwardDiff:         $(round(t_grad, digits=2)) ms (median)")
println("  PolyesterForwardDiff: $(round(t_poly, digits=2)) ms (median)")
println("  Ratio FD/llh:        $(round(t_grad / t_llh, digits=1))×")
println("  Ratio Poly/llh:      $(round(t_poly / t_llh, digits=1))×")
println("  Speedup Poly vs FD:  $(round(t_grad / t_poly, digits=2))×")
println("  Parameters:          $n_params")
println("  Threads:             $(Threads.nthreads())")
