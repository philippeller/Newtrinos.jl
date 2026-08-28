"""
    Newtrinos

A Julia package for global analysis of neutrino data. Provides modular physics models,
experimental likelihoods, and inference tools (profile likelihood, Bayesian sampling).
"""
module Newtrinos

"""
    Physics

Abstract type for physics modules. Subtypes must have `params::NamedTuple` and `priors::NamedTuple` fields.
"""
abstract type Physics end

"""
    Experiment

Abstract type for experiment modules. Subtypes must have `physics`, `params`, `priors`, `assets`,
`forward_model`, and `plot` fields.
"""
abstract type Experiment end

export Physics, Experiment
export NewtrinosResult, plot
export bin, rebin
export make_init_samples, make_prior_samples, whack_a_moles, whack_many_moles

include("physics/osc.jl")
using .osc
include("physics/barger_eigen.jl")
include("physics/earth_layers.jl")
include("physics/atm_flux.jl")
include("physics/xsec.jl")
include("physics/cevns_xsec.jl")
include("physics/sns_flux.jl")
include("analysis/analysis_tools.jl")
include("analysis/molewhacker.jl")
include("utils/plotting.jl")
include("utils/plotting_bat.jl")
include("utils/autodiff.jl")
include("utils/helpers.jl")

include("experiments/daya_bay/daya_bay_3158days/dayabay.jl")
include("experiments/minos/minos_sterile_16e20_POT/minos.jl")
#include("experiments/icecube/deepcore_3y_highstats_sample_b/deepcore.jl")
include("experiments/icecube/deepcore_9y_verification_sample/deepcore.jl")
include("experiments/super_k/sk_atm_2023/super_k.jl")
include("experiments/icecube/upgrade_sim_2020/ic_upgrade.jl")
include("experiments/kamland/kamland_7years/kamland.jl")
include("experiments/km3net/orca6_433kton/orca.jl")
#include("experiments/coherent/coherent_2020/coherent_csi.jl")
include("experiments/coherent/coherent_2020/coherent_csi.jl")
include("experiments/coherent/coherent_2020/coherent_lAr.jl")

include("experiments/juno/juno.jl")
include("experiments/juno/tao.jl")
include("experiments/gerda/gerda.jl")
include("experiments/katrin/katrin.jl")
end
