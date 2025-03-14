module minos

using LinearAlgebra
using Distributions
using HDF5
using BAT
using DataStructures

function prepare_data(datadir = @__DIR__)

    h5file = h5open(joinpath(datadir, "dataRelease.h5"), "r")

    data = Dict()

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

   return (
        ch_data = ch_data,
        TotalCCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalCCCovar"])),
        TotalNCCovar = (x->reshape(x, fill(Int(sqrt(length(x))), 2)...))(read(h5file["TotalNCCovar"])),
    ), (
            CC = ch_data["FDCC"].observed,
            NC = ch_data["FDNC"].observed,
        )
end

const data, observed = prepare_data()
params = OrderedDict()
priors = OrderedDict()


function get_expected_per_channel(params, osc_prob, data)
    # Minos baseline:
    s = data.smearings
    p = osc_prob(s.E, [data.L], params)
    NuMu = s.NuMu * p[:,[1],2,2]
    TrueNC = s.TrueNC * dropdims(sum(p[:,[1],2,1:3], dims=3), dims=3)
    BeamNue = s.BeamNue * p[:,[1],1,1]
    AppNue = s.AppNue * p[:,[1],2,1]
    AppNuTau = s.AppNuTau * p[:,[1],2,3]
    dropdims(NuMu + TrueNC + BeamNue + AppNue + AppNuTau, dims=2)
end

function forward_model_per_channel(params, osc_prob, channel, data)
   
    observed_far = data.ch_data["FD"*channel].observed
    expected_far = get_expected_per_channel(params, osc_prob, data.ch_data["FD"*channel])
    observed_near = data.ch_data["ND"*channel].observed
    expected_near = get_expected_per_channel(params, osc_prob, data.ch_data["ND"*channel])
    
    cov = channel == "CC" ? data.TotalCCCovar : data.TotalNCCovar

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

function forward_model(osc_prob)
    model = let this_data = data
        params -> distprod(
            CC = forward_model_per_channel(params, osc_prob, "CC", this_data),
            NC = forward_model_per_channel(params, osc_prob, "NC", this_data),
        )
        end
end


end
