module earth_layers

using CSV, DataFrames
using StatsBase
using StaticArrays, ArraysOfArrays, StructArrays
using DataStructures

using ..Newtrinos
export configure
export PREM

const datadir = @__DIR__ 

abstract type DensityModel end

@kwdef struct PREM <: DensityModel
    zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]
    p_fractions::Float64 = 0.5
    atm_heihgt::Float64 = 20.
end

@kwdef struct EarthLayers <: Newtrinos.Physics
    cfg::DensityModel
    params::NamedTuple
    priors::NamedTuple
    compute_layers::Function
    compute_paths::Function
end

function configure(cfg::DensityModel=PREM())
    EarthLayers(
        cfg=cfg,
        params = (;),
        priors = (;),
        compute_layers = get_compute_layers(cfg),
        compute_paths = compute_paths
        )
end


function get_compute_layers(cfg::PREM)
    function compute_layers()
        
        PREM = CSV.read(joinpath(datadir, "PREM_1s.csv"), DataFrame, header=["radius","depth","density","Vpv","Vph","Vsv","Vsh","eta","Q-mu","Q-kappa"])
        # density boundaries to define the constant density zones
        
        radii = Float64[]
        ave_densities = Float64[]
        
        push!(radii, 6371+cfg.atm_heihgt)
        push!(ave_densities, 0.)
        
        for i in 1:length(cfg.zones)-1
            mask = (PREM.density .< cfg.zones[i+1]) .& (PREM.density .>= cfg.zones[i])
            push!(radii, maximum(PREM.radius[mask]))
            push!(ave_densities, mean(PREM.density[mask]))
        end
        
        layers = StructArray{Newtrinos.Layer}((radii, ave_densities .* cfg.p_fractions, ave_densities .* (1 .- cfg.p_fractions)))
    end
end
function ray_circle_path_length(r, y, cz)    
    # Compute the discriminant
    disc = r^2 - y^2 + (y * cz)^2
    
    if disc < 0
        return 0.0  # No intersection
    end

    sqrt_disc = sqrt(disc)
    
    # Compute intersection points
    t1 = - y * cz - sqrt_disc
    t2 = - y * cz + sqrt_disc

    # Compute path length, ensuring we only count positive t-values
    L = max(0, t2 - max(0, t1))

    if L < 1
        return 0.
    end
    L
end

# ToDo: could probably skip layers smaller than few km and "absorb" those into the next outer layer

function compute_paths(cz::Number, layers, r_detector)
    radii = layers.radius
    intersections = ray_circle_path_length.(radii, r_detector, cz)
    for i in 1:length(intersections) - 1
        intersections[i] -= intersections[i+1]
    end
    mask = intersections .> 0.
    rs = radii[mask]
    intersections = intersections[mask]

    n_layers_outside = sum(radii .>= r_detector)

    n_layers = 2 * (length(intersections) - n_layers_outside) + n_layers_outside

    lengths_traversed = zeros(n_layers)
    layer_idx_traversed = zeros(Int, n_layers)

    for i in 1:length(intersections)
        if (i < n_layers_outside) | (i == length(intersections))
            lengths_traversed[i] = intersections[i]
            layer_idx_traversed[i] = i
        elseif i == n_layers_outside
            len_det = -cz * (rs[i] - r_detector)
            inter = intersections[i] - len_det
            lengths_traversed[i] = inter/2 + len_det
            layer_idx_traversed[i] = i
            lengths_traversed[end-i+n_layers_outside] = inter/2
            layer_idx_traversed[end-i+n_layers_outside] = i              
        else
            lengths_traversed[i] = intersections[i]/2
            layer_idx_traversed[i] = i
            lengths_traversed[end-i+n_layers_outside] = intersections[i]/2
            layer_idx_traversed[end-i+n_layers_outside] = i
        end
    end

    la = StructArray{Newtrinos.Path}((lengths_traversed, layer_idx_traversed))
    
end

function compute_paths(cz::AbstractArray, layers; r_detector = 6369)
    VectorOfVectors{Newtrinos.Path}(compute_paths.(cz, Ref(layers), r_detector));
end

end