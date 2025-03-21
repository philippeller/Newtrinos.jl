module Newtrinos

export NewtrinosResult, plot

include("theory/osc.jl")
include("analysis/analysis_tools.jl")
include("utils/plotting.jl")
include("utils/autodiff.jl")

include("experiments/daya_bay/daya_bay_3158days/dayabay.jl")
include("experiments/minos/minos_sterile_16e20_POT/minos.jl")
include("experiments//kamland/kamland_7years/kamland.jl")

end
