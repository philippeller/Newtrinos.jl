module earth_layers

using CSV, DataFrames
using StatsBase
using StaticArrays, ArraysOfArrays, StructArrays
using DataStructures

using ..osc
export compute_layers

const datadir = @__DIR__ 

PREM = CSV.read(joinpath(datadir, "PREM_1s.csv"), DataFrame, header=["radius","depth","density","Vpv","Vph","Vsv","Vsh","eta","Q-mu","Q-kappa"])

# Radius at which the detector is sitting
r_detector = 6369
# height of the atmosphere to include
atm_heihgt = 20
# density boundaries to define the constant density zones
zones = [0, 4, 7.5, 12.5, 13.1]


radii = []
ave_densities = []

push!(radii, 6371+20.)
push!(ave_densities, 0.)

for i in 1:length(zones)-1
    mask = (PREM.density .< zones[i+1]) .& (PREM.density .>= zones[i])
    push!(radii, maximum(PREM.radius[mask]))
    push!(ave_densities, mean(PREM.density[mask]))
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


function compute_layers(cz::Number; p_fraction=0.5)
    intersections = ray_circle_path_length.(radii, r_detector, cz)
    for i in 1:length(intersections) - 1
        intersections[i] -= intersections[i+1]
    end
    mask = intersections .> 0.
    intersections = intersections[mask]
    rs = radii[mask]
    ds = ave_densities[mask]

    intersections, rs, ds

    n_layers_outside = sum(radii .>= r_detector)

    n_layers = 2 * (length(intersections) - n_layers_outside) + n_layers_outside

    lengths_traversed = zeros(n_layers)
    densities_traversed = zeros(n_layers)

    for i in 1:length(intersections)
        if (i < n_layers_outside) | (i == length(intersections))
            lengths_traversed[i] = intersections[i]
            densities_traversed[i] = ds[i]
        elseif i == n_layers_outside
            len_det = -cz * (rs[i] - r_detector)
            inter = intersections[i] - len_det
            lengths_traversed[i] = inter/2 + len_det
            densities_traversed[i] = ds[i]
            lengths_traversed[end-i+n_layers_outside] = inter/2
            densities_traversed[end-i+n_layers_outside] = ds[i]              
        else
            lengths_traversed[i] = intersections[i]/2
            densities_traversed[i] = ds[i]
            lengths_traversed[end-i+n_layers_outside] = intersections[i]/2
            densities_traversed[end-i+n_layers_outside] = ds[i]
        end
    end
    
    lengths_traversed , densities_traversed
    la = StructArray{Layer}((lengths_traversed, densities_traversed .* p_fraction, densities_traversed .* (1-p_fraction)))
    
end

function compute_layers(cz::AbstractArray; p_fraction=0.5)
    VectorOfVectors{Layer}(compute_layers.(cz));
end

end