module earth_layers

using CSV, DataFrames
using StatsBase
using StaticArrays, ArraysOfArrays, StructArrays
using DataStructures

using ..Newtrinos
export configure
export PREM

const datadir = @__DIR__

"""
    DensityModel

Abstract type for Earth density profile models.

Each subtype provides a recipe for dividing the Earth into concentric shells of constant
density. Currently the only implementation is [`PREM`](@ref).
"""
abstract type DensityModel end

"""
    PREM <: DensityModel

Preliminary Reference Earth Model (Dziewonski & Anderson, 1981).

Reads the tabulated PREM density profile from `PREM_1s.csv` and groups radial shells into
zones defined by density boundaries. The proton-to-nucleon fraction is assumed constant
across all layers.

# Fields
- `zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]`: density boundaries [g/cm³]
  defining the constant-density zones. Adjacent PREM rows whose density falls within the
  same bin are averaged into a single [`Layer`](@ref).
- `p_fractions::Float64 = 0.5`: proton number fraction ``Y_p = N_p / (N_p + N_n)``.
- `atm_heihgt::Float64 = 20.`: atmospheric shell thickness [km] added above the Earth's
  surface (density = 0).
"""
@kwdef struct PREM <: DensityModel
    zones::Array{Float64} = [0., 4., 7.5, 12.5, 13.1]
    p_fractions::Float64 = 0.5
    atm_heihgt::Float64 = 20.
end

"""
    EarthLayers <: Newtrinos.Physics

Configured Earth density model, returned by [`configure`](@ref).

Provides the layer structure and path-computation functions needed by the oscillation
module's matter-effect calculations (see [`Newtrinos.osc.SI`](@ref)).

# Fields
- `cfg::DensityModel`: the density model used to build this module.
- `params::NamedTuple`: oscillation parameters.
- `priors::NamedTuple`: prior distributions.
- `compute_layers::Function`: closure `compute_layers() -> StructVector{Layer}` returning
  the concentric density shells.
- `compute_paths::Function`: function
  `compute_paths(cz, layers; r_detector) -> VectorOfVectors{Path}` computing the layer
  traversal for each cosine-zenith value.
"""
@kwdef struct EarthLayers <: Newtrinos.Physics
    cfg::DensityModel
    params::NamedTuple
    priors::NamedTuple
    compute_layers::Function
    compute_paths::Function
end

"""
    configure(cfg::DensityModel=PREM()) -> EarthLayers

Create a fully configured Earth density physics module.

# Arguments
- `cfg::DensityModel`: density model (defaults to [`PREM`](@ref)).

# Returns
An [`EarthLayers`](@ref) instance with `compute_layers` and `compute_paths` closures and empty parameters and priors.

# Examples
```julia
using Newtrinos

earth = Newtrinos.earth_layers.configure()
layers = earth.compute_layers()
paths = earth.compute_paths([-1.0, -0.5, 0.0], layers)
```
"""
function configure(cfg::DensityModel=PREM())
    EarthLayers(
        cfg=cfg,
        params = (;),
        priors = (;),
        compute_layers = get_compute_layers(cfg),
        compute_paths = compute_paths
        )
end


"""
    get_compute_layers(cfg::PREM) -> Function

Construct a closure that builds the Earth layer structure from the PREM density profile.

The returned function `compute_layers()` reads `PREM_1s.csv`, bins rows by the density
boundaries in `cfg.zones`, and returns a `StructVector{Layer}` with one entry per zone
plus an atmospheric shell.

# Arguments
- `cfg::PREM`: PREM configuration with zone boundaries, proton fraction, and atmosphere height.

# Returns
A zero-argument closure `compute_layers() -> StructVector{Layer}`.
"""
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
"""
    ray_circle_path_length(r, y, cz) -> Real

Compute the chord length of a ray through a circle of radius `r`.

The ray originates at radial distance `y` from the Earth's centre with direction
cosine `cz` (cosine of the zenith angle). Returns zero if the ray does not intersect
the circle or if the chord is shorter than 1 km (numerical noise filter).

# Arguments
- `r`: radius of the spherical shell [km].
- `y`: radial position of the detector [km].
- `cz`: cosine of the zenith angle (−1 = vertically upgoing through the core).

# Returns
Path length through the shell [km], or zero if no intersection.
"""
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

"""
    compute_paths(cz::Number, layers, r_detector) -> StructArray{Path}
    compute_paths(cz::AbstractArray, layers; r_detector=6369) -> VectorOfVectors{Path}

Compute the sequence of [`Path`](@ref) segments a neutrino traverses through the Earth.

For a single cosine-zenith value, determines which [`Layer`](@ref) shells are crossed
using [`ray_circle_path_length`](@ref), then builds an ordered list of segments. Layers
below the detector are traversed twice (entry and exit), while layers above the detector
are traversed once.

The array method broadcasts over multiple cosine-zenith values and returns a
`VectorOfVectors{Path}`.

# Arguments
- `cz`: cosine of the zenith angle (scalar or array). −1 is vertically upgoing.
- `layers::StructVector{Layer}`: Earth density layers from `compute_layers()`.
- `r_detector::Real`: radial position of the detector [km] (default 6369, approximate
  IceCube depth).

# Returns
- Scalar method: `StructArray{Path}` with `length` and `layer_idx` columns.
- Array method: `VectorOfVectors{Path}`, one `Vector{Path}` per zenith angle.
"""
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