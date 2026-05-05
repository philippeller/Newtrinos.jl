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

experiments = (orca = Newtrinos.orca.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₂₃] = 7
vars_to_scan[:Δm²₃₁] = 7

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.θ₁₂ = p.θ₁₂
@reset priors.δCP = p.δCP
@reset priors.θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18)

@reset priors.Δm²₃₁ = Uniform(0.0016, 0.0027)
@reset priors.θ₂₃ = Uniform(pi/4-0.15, pi/4+0.15)
@reset priors.atm_flux_updown_sigma = 0.0
@reset priors.atm_flux_nuenuebar_sigma_lo = 0.
@reset priors.atm_flux_nuenumu_sigma_lo = 0.
@reset priors.atm_flux_numunumubar_sigma_lo = 0.



result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="test")

if !isdir("test_output")
    mkdir("test_output")
end

FileIO.save("test_output/test.jld2", Dict("result" => result))

converted = Newtrinos.NewtrinosResult(axes = (sin2theta23 = sin.(result.axes.θ₂₃).^2, Δm²₃₁ = result.axes.Δm²₃₁), values=result.values);

official_raw = CSV.read("../../km3net/orca6_433kton/chi2_landscape_NO.csv", DataFrame)
official = NewtrinosResult(axes=(sin2theta23=sin.(reshape(official_raw.theta23[2:end], 41,50)[:, 1] ./180. * pi).^2, Δm²₃₁ = reshape(official_raw.dm31[2:end], 41,50)[1, :]), values = (log_posterior=-0.5 .* reshape(official_raw.chi2[2:end], 41,50),))

fig = Figure()
ax = Axis(fig[1,1])
ax.title = "ORCA NO 68%, 90% C.L. contours"
plot!(ax, official, levels=[0.68, 0.9,], label="official", color=:red)
plot!(ax, converted, levels=[0.68, 0.9,], label="ours")

ax.xlabel = "sin²θ₂₃"
ax.ylabel = "Δm²₃₁ (eV²)"
axislegend(ax)
save("test_output/contours.png", fig)

bestfit = Newtrinos.bestfit(result)

fig = experiments.orca.plot(bestfit)
save("test_output/datamc.png", fig)

open("README.md", "w") do io
    write(io, "# KM3NeT ORCA6 433 k-ton Sample\n ## Resources\n")
    write(io, "Data source: https://opendata.km3net.de/dataset.xhtml?persistentId=doi:10.5072/FK2/Y0UXVW\n")
    write(io, "\n## Test output plots\n")
    write(io, "![Comparison](test_output/contours.png)\n")
    write(io, "![DataMC](test_output/datamc.png)\n")
    write(io, "## Meta Information\n")
    for (key, value) in result.meta
        write(io, "- **$(key)**: $(value)\n")
    end
end
