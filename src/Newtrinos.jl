module Newtrinos


abstract type Physics end
abstract type Experiment end

export Physics, Experiment
export NewtrinosResult, plot

include("physics/osc.jl")
using .osc
include("physics/earth_layers.jl")
include("physics/atm_flux.jl")
include("physics/xsec.jl")
include("analysis/analysis_tools.jl")
include("utils/plotting.jl")
include("utils/autodiff.jl")

include("experiments/daya_bay/daya_bay_3158days/dayabay.jl")
include("experiments/minos/minos_sterile_16e20_POT/minos.jl")
include("experiments/icecube/deepcore_3y_highstats_sample_b/deepcore.jl")
include("experiments/kamland/kamland_7years/kamland.jl")
include("experiments/km3net/orca6_433kton/orca.jl")
include("experiments/nova/nova_try.jl")
include("experiments/juno/juno.jl")
include("experiments/katrin/katrin.jl")

end
