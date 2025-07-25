module Newtrinos


abstract type Physics end
abstract type Experiment end

export Physics, Experiment
export NewtrinosResult, plot
export make_init_samples, make_prior_samples, whack_a_moles, whack_many_moles

include("physics/osc.jl")
using .osc
include("physics/earth_layers.jl")
include("physics/atm_flux.jl")
include("physics/xsec.jl")
include("physics/cevns_xsec.jl")
include("physics/sns_flux.jl")
include("analysis/analysis_tools.jl")
include("analysis/molewhacker.jl")
include("utils/plotting.jl")
include("utils/autodiff.jl")

include("experiments/daya_bay/daya_bay_3158days/dayabay.jl")
include("experiments/minos/minos_sterile_16e20_POT/minos.jl")
include("experiments/icecube/deepcore_3y_highstats_sample_b/deepcore.jl")
include("experiments/kamland/kamland_7years/kamland.jl")
include("experiments/km3net/orca6_433kton/orca.jl")
include("experiments/coherent/coherent_2020/coherent_csi.jl")

include("experiments/juno/juno.jl")
include("experiments/juno/tao.jl")
end
