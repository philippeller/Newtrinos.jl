
module katrin

import ..Newtrinos

using LinearAlgebra, Statistics, Distributions, StatsBase
using HDF5
using FileIO
using Base.Threads



@kwdef struct Katrin <: Newtrinos.Experiment
      physics::NamedTuple
      params::NamedTuple
      priors::NamedTuple
     # assets::NamedTuple
end

function configure(physics)
    physics = (;physics.osc)

    return Katrin(
        physics = physics,
        params = (;),
        priors = (;)
    )
end


 

end