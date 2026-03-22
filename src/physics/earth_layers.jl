module earth_layers

using CSV, DataFrames
using StatsBase
using StaticArrays, ArraysOfArrays, StructArrays
using DataStructures
using Distributions

using ..Newtrinos
export configure
export PREM

const datadir = @__DIR__

abstract type DensityModel end

@kwdef struct PREM <: DensityModel
    zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]
    p_fractions::Vector{Float64} = [0.496, 0.494, 0.468, 0.466]  # Ye per density zone
    atm_heihgt::Float64 = 20.
end

@kwdef struct VariableDensity <: DensityModel
    prem::PREM = PREM()
end

@kwdef struct EarthLayers <: Newtrinos.Physics
    cfg::DensityModel
    params::NamedTuple
    priors::NamedTuple
    compute_layers::Function
    compute_paths::Function
end

function configure(cfg::PREM=PREM())
    EarthLayers(
        cfg=cfg,
        params = (;),
        priors = (;),
        compute_layers = get_compute_layers(cfg),
        compute_paths = compute_paths
        )
end

function configure(cfg::VariableDensity)
    EarthLayers(
        cfg=cfg,
        params = (matter_density_scale = 1.0,),
        priors = (matter_density_scale = Normal(1.0, 0.068),),
        compute_layers = get_compute_layers(cfg.prem),
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

        ye = vcat([0.5], cfg.p_fractions)  # prepend atmosphere Ye (density=0, so value irrelevant)
        layers = StructArray{Newtrinos.Layer}((radii, ave_densities .* ye, ave_densities .* (1 .- ye)))
    end
end

function scale_densities(layers, scale)
    StructArray{Newtrinos.Layer}((layers.radius, layers.p_density .* scale, layers.n_density .* scale))
end

function ray_circle_path_length(r, y, cz)
    # Compute the discriminant
    disc = r^2 - y^2 + (y * cz)^2
    T = typeof(disc)

    if disc < 0
        return zero(T)  # No intersection
    end

    sqrt_disc = sqrt(disc)

    # Compute intersection points
    t1 = - y * cz - sqrt_disc
    t2 = - y * cz + sqrt_disc

    # Compute path length, ensuring we only count positive t-values
    L = max(zero(T), t2 - max(zero(T), t1))

    if L < 1
        return zero(T)
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
