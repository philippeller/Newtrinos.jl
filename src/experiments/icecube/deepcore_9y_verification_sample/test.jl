using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (deepcore = Newtrinos.deepcore.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 11
vars_to_scan[:Δm²₃₁] = 11

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.δCP = p.δCP
@reset priors.xsec_nutau_cc_norm = p.xsec_nutau_cc_norm
@reset priors.θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18)
@reset priors.Δm²₃₁ = Uniform(0.0022, 0.0027)
@reset priors.θ₂₃ = Uniform(0.685, 0.9)

result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end

FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(result.axes.θ₂₃).^2, Δm²₃₂ = result.axes.Δm²₃₁ .- p.Δm²₂₁), values=result.values);

official = CSV.read("DeepCore_oscNext_verification_sample__sin2_theta23_dm2_32__90pc_result_bugfix.csv", DataFrame; header=false)

fig = Figure()
ax = Axis(fig[1,1])
ax.title = "DeepCore NO 90% C.L. contours"
plot!(ax, converted, levels=[0.9,], label="ours")

lines!(ax, official.Column1, official.Column2, color=:red, label="official")

ax.xlabel = "sin²θ₂₃"
ax.ylabel = "Δm²₃₂ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)

fig = experiments.deepcore.plot(bestfit)
save("test_output/datamc.png", fig)

open("README.md", "w") do io
    write(io, "# IceCube DeepCore 9y Verification Sample\n ## Resources\n")
    write(io, """
data source: https://icecube.wisc.edu/data-releases/2025/07/measurement-of-atmospheric-neutrino-mixing-with-improved-icecube-deepcore-calibration-and-data-processing/
IceCube Collaboration, 2025, "Replication Data for: Measurement of atmospheric neutrino mixing with improved IceCube DeepCore calibration and data processing", https://doi.org/10.7910/DVN/B4RITM, Harvard Dataverse, V1, UNF:6:EqPPIAlmbhWU7MUgQgQVCw== [fileUNF]

""")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
