module minos

using LinearAlgebra
using Distributions
using HDF5
using BAT
using DataStructures
using CairoMakie
using Logging
import ..Newtrinos


@kwdef struct Minos <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return Minos(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading minos data"

    h5file = h5open(joinpath(datadir, "dataRelease.h5"), "r")

    channels = ["FDCC", "FDNC", "NDCC", "NDNC"]
    experiments = ["minos", "minosPlus"]

    L=735.0

    ch_data = Dict([
        channel => (
            observed = sum([read(h5file["data$(channel)_$(ex)_hist"]) for ex in experiments]),
            bin_edges = read(h5file["data$(channel)_minosPlus_bins"]),
            smearings = (
                E=(x -> L ./((x[1:end-1] .+ x[2:end]) ./ 2))(read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins1"])),
                energy_edges=read(h5file["hRecoToTrue$(channel)SelectedNuMu_minos_bins2"]),
                NuMu=sum([read(h5file["hRecoToTrue$(channel)SelectedNuMu_$(ex)_hist"]) for ex in experiments]),
                TrueNC=sum([read(h5file["hRecoToTrue$(channel)SelectedTrueNC_$(ex)_hist"]) for ex in experiments]),
                BeamNue=sum([read(h5file["hRecoToTrue$(channel)SelectedBeamNue_$(ex)_hist"]) for ex in experiments]),
                AppNue=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNue_$(ex)_hist"]) for ex in experiments]),
                AppNuTau=sum([read(h5file["hRecoToTrue$(channel)SelectedAppNuTau_$(ex)_hist"]) for ex in experiments])
            ),
            L=L,
        )
        for channel in channels
    ])

   assets = (
        ch_data = ch_data,
        TotalCCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalCCCovar"])),
        TotalNCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalNCCovar"])),
        observed = (
            CC = ch_data["FDCC"].observed,
            NC = ch_data["FDNC"].observed,
        ),
    )

end


function get_expected_per_channel(params, physics, assets)
    # Minos baseline:
    s = assets.smearings
    p = physics.osc.osc_prob(s.E, [assets.L], params)
    NuMu = s.NuMu * p[:,[1],2,2]
    TrueNC = s.TrueNC * dropdims(sum(p[:,[1],2,1:3], dims=3), dims=3)
    BeamNue = s.BeamNue * p[:,[1],1,1]
    AppNue = s.AppNue * p[:,[1],2,1]
    AppNuTau = s.AppNuTau * p[:,[1],2,3]
    dropdims(NuMu + TrueNC + BeamNue + AppNue + AppNuTau, dims=2)
end

function forward_model_per_channel(params, physics, channel, assets)
   
    observed_far = assets.ch_data["FD"*channel].observed
    expected_far = get_expected_per_channel(params, physics, assets.ch_data["FD"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    observed_near = assets.ch_data["ND"*channel].observed
    expected_near = get_expected_per_channel(params, physics, assets.ch_data["ND"*channel]) * physics.xsec.scale(:any, Symbol(channel), params)
    
    cov = channel == "CC" ? assets.TotalCCCovar : assets.TotalNCCovar

    tot = vcat(expected_far, expected_near)
    cov = (tot * tot') .* cov + diagm(tot)
        
    cov11 = cov[1:length(observed_far), 1:length(observed_far)]
    cov12 = cov[1:length(observed_far), end-length(observed_near)+1:end]
    cov21 = cov[end-length(observed_near)+1:end, 1:length(observed_far)]
    cov22 = cov[end-length(observed_near)+1:end, end-length(observed_near)+1:end]

    cov22inv = inv(cov22)

    x = cov12 * (cov22inv * (observed_near - expected_near))
    expected = expected_far + x

    cov = cov11 - cov12 * cov22inv * cov21

    Distributions.MvNormal(expected, (cov+cov')/2)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        distprod(
            CC = forward_model_per_channel(params, physics, "CC", assets),
            NC = forward_model_per_channel(params, physics, "NC", assets),
        )
    end
end

function get_plot(physics, assets)

    function plot(params, d=assets.observed)
        f = Figure()
    
        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
        
        for (i, ch) in enumerate([:CC, :NC])
        
            ax = Axis(f[1,i])
            energy_bins = assets.ch_data["FD"*String(ch)].bin_edges
            energy = 0.5 .* (energy_bins[1:end-1] .+ energy_bins[2:end])
            
            plot!(ax, energy, d[ch] ./ diff(energy_bins), color=:black, label="Observed")
            stephist!(ax, energy, weights=m[ch] ./ diff(energy_bins), bins=energy_bins, label="Expected")
            barplot!(ax, energy, (m[ch] .+ sqrt.(v[ch])) ./ diff(energy_bins), width=diff(energy_bins), gap=0, fillto= (m[ch] .- sqrt.(v[ch])) ./ diff(energy_bins), alpha=0.5, label="Standard Deviation")
            
            ax.ylabel="Counts / GeV"
            ax.title="MINOS/MINOS+ Far Detector "*String(ch)
            axislegend(ax, framevisible = false)
            
            
            ax2 = Axis(f[2,i])
            plot!(ax2, energy, d[ch] ./ m[ch], color=:black, label="Observed")
            hlines!(ax2, 1, label="Expected")
            barplot!(ax2, energy, 1 .+ sqrt.(v[ch]) ./ m[ch], width=diff(energy_bins), gap=0, fillto= 1 .- sqrt.(v[ch])./m[ch], alpha=0.5, label="Standard Deviation")
            ylims!(ax2, 0.7, 1.3)
            
            ax.xticksvisible = false
            ax.xticklabelsvisible = false
            
            rowsize!(f.layout, 1, Relative(3/4))
            rowgap!(f.layout, 1, 0)
            
            ax2.xlabel="Reconstructed Energy (GeV)"
            ax2.ylabel="Counts/Expected"
            
            xlims!(ax, minimum(energy_bins), maximum(energy_bins))
            xlims!(ax2, minimum(energy_bins), maximum(energy_bins))
        
        end
        
        ylims!(f.content[1], 0, 800)
        ylims!(f.content[4], 0, 600)
        
        f
    end
end

end
