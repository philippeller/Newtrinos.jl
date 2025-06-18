module osc
using LinearAlgebra
using StaticArrays
using StatsBase
using ArraysOfArrays, StructArrays
using DataStructures
using Distributions
using Interpolations
using ..Newtrinos

export ftype
export Layer
export Path
export Decoherent, Damping, Basic
export All, Cut
export Vacuum, SI, NSI
export ThreeFlavour, ThreeFlavourXYCP, Sterile, ADD
export OscillationConfig
export configure

const ftype = Float64

# struct for matter layers
struct Layer{T}
    radius::T
    p_density::T
    n_density::T
end

# struct for matter paths
struct Path
    length::Float64
    layer_idx::Int
end 

# Physical constants
const N_A = 6.022e23 #[mol^-1]
const G_F = 8.961877245622253e-38 #[eV*cm^3]
const A = sqrt(2) * G_F * N_A
# conversion factor for km/GeV (1/(2*hbar*c))
const F_units = 2.5338653580781976
# um to eV
const umev = 5.067730716156395

# TYPE DEFINITIONS

abstract type PropagationModel end
struct Basic <: PropagationModel end
@kwdef struct Decoherent <: PropagationModel 
    σₑ::Float64=0.1
end
@kwdef struct Damping <: PropagationModel
    σₑ::Float64=0.1
end

abstract type StateSelector end
struct All <: StateSelector end
@kwdef struct Cut <: StateSelector
    cutoff::Float64 = Inf
end

abstract type InteractionModel end
struct Vacuum <: InteractionModel end
struct NSI <: InteractionModel end
struct SI <: InteractionModel end

abstract type FlavourModel end
@kwdef struct ThreeFlavour <: FlavourModel 
    ordering::Symbol = :NO
end
@kwdef struct ThreeFlavourXYCP <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
end
@kwdef struct Sterile <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
end
@kwdef struct ADD <: FlavourModel 
    three_flavour::ThreeFlavour = ThreeFlavour()
    N_KK::Int = 5
end

@kwdef struct Darkdim_Lambda <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
    N_KK::Int = 5
end

@kwdef struct Darkdim_Masses <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
    N_KK::Int = 5
end

@kwdef struct OscillationConfig{F<:FlavourModel, I<:InteractionModel, P<:PropagationModel, S<:StateSelector}
    flavour::F = ThreeFlavour()
    interaction::I = Vacuum()
    propagation::P = Basic()
    states::S = All()
end

@kwdef struct Osc <: Newtrinos.Physics
    cfg::OscillationConfig
    params::NamedTuple
    priors::NamedTuple
    matrices::Function
    osc_prob::Function
end

function configure(cfg::OscillationConfig=OscillationConfig())
    Osc(
        cfg=cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        matrices = get_matrices(cfg.flavour),
        osc_prob = get_osc_prob(cfg)
    )
end


# PARAMS & PRIORS

# for now only the flavour model has any params...to be changed
get_params(cfg::OscillationConfig) = get_params(cfg.flavour)
get_priors(cfg::OscillationConfig) = get_priors(cfg.flavour)

function get_params(cfg::ThreeFlavour)
    params = OrderedDict()
    params[:θ₁₂] = ftype(asin(sqrt(0.307)))
    params[:θ₁₃] = ftype(asin(sqrt(0.021)))
    params[:θ₂₃] = ftype(asin(sqrt(0.57)))
    params[:δCP] = ftype(1.)
    params[:Δm²₂₁] = ftype(7.53e-5)
    
    if cfg.ordering == :NO
        params[:Δm²₃₁] = ftype(2.4e-3 + params[:Δm²₂₁])
    elseif cfg.ordering == :IO
        params[:Δm²₃₁] = ftype(-2.4e-3)
    else
        throw("Unknown ordering `$(cfg.ordering)`. Must be either :NO or :IO.")
    end
    NamedTuple(params)
end

function get_priors(cfg::ThreeFlavour)
    priors = OrderedDict()
    priors[:θ₁₂] = Uniform(atan(sqrt(0.2)), atan(sqrt(1)))
    priors[:θ₁₃] = Uniform(ftype(0.1), ftype(0.2))
    priors[:θ₂₃] = Uniform(ftype(pi/4 *2/3), ftype(pi/4 *4/3))
    priors[:δCP] = Uniform(ftype(0), ftype(2*π))
    priors[:Δm²₂₁] = Uniform(ftype(6.5e-5), ftype(9e-5))
    if cfg.ordering == :NO
        priors[:Δm²₃₁] = Uniform(ftype(2e-3), ftype(3e-3))
    elseif cfg.ordering == :IO
        priors[:Δm²₃₁] = Uniform(ftype(-3e-3), ftype(-2e-3))
    else
        throw("Unknown ordering $ordering. Must be either :NO or :IO.")
    end
    NamedTuple(priors)
end

function get_params(cfg::ThreeFlavourXYCP)
    std = get_params(cfg.three_flavour)
    params = OrderedDict{Symbol, Any}(pairs(std))
    delete!(params, :δCP)
    params[:δCPshell] = [1., 0.]
    #params[:δCPy] = 0.
    NamedTuple(params)
end

function get_priors(cfg::ThreeFlavourXYCP)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    delete!(priors, :δCP)
    priors[:δCPshell] = MvNormal([1,1])
    #priors[:δCPy] = Normal(0., 1.)
    NamedTuple(priors)
end

function get_params(cfg::Sterile)
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    params[:Δm²₄₁] = 1
    params[:θ₁₄] = 0.1
    params[:θ₂₄] = 0.1
    params[:θ₃₄] = 0.1
    NamedTuple(params)
end

function get_priors(cfg::Sterile)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:Δm²₄₁] = Uniform(0.1, 10.)
    priors[:θ₁₄] = Uniform(0., 1.)
    priors[:θ₂₄] = Uniform(0., 1.)
    priors[:θ₃₄] = Uniform(0., 1.)
    NamedTuple(priors)
end
    
function get_params(cfg::ADD)
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    params[:m₀] = ftype(0.01)
    params[:ADD_radius] = ftype(1e-2)
    NamedTuple(params)
end

function get_priors(cfg::ADD)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
    priors[:ADD_radius] = LogUniform(ftype(1e-3),ftype(1))
    NamedTuple(priors)
end

function get_params(cfg::Darkdim_Lambda)
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    pop!(params, :Δm²₂₁)
    pop!(params, :Δm²₃₁)
    params[:Darkdim_radius] = 0.1
    params[:ca1] = ftype(1e-5)
    params[:ca2] = ftype(1e-5)
    params[:ca3] = ftype(1e-5)
    params[:λ₁] = ftype(1.)
    params[:λ₂] = ftype(1.)
    params[:λ₃] = ftype(1.)
    NamedTuple(params)
end

function get_priors(cfg::Darkdim_Lambda)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    pop!(priors, :Δm²₂₁)
    pop!(priors, :Δm²₃₁)
    priors[:Darkdim_radius] = LogUniform(ftype(1e-1),ftype(10))
    priors[:ca1] = Uniform(ftype(1e-5), ftype(10))
    priors[:ca2] = Uniform(ftype(1e-5), ftype(10))
    priors[:ca3] = Uniform(-ftype(10), -ftype(1e-5))
    priors[:λ₁] = Uniform(ftype(0), ftype(10))
    priors[:λ₂] = Uniform(ftype(0), ftype(10))
    priors[:λ₃] = Uniform(ftype(0), ftype(10))
    priors = NamedTuple(priors)
    NamedTuple(priors)
end

function get_params(cfg::Darkdim_Masses)
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    params[:m₀] = ftype(0.01)
    params[:Darkdim_radius] = 0.1
    params[:λ₁] = ftype(1.)
    params[:λ₂] = ftype(1.)
    params[:λ₃] = ftype(1.)
    NamedTuple(params)
end

function get_priors(cfg::Darkdim_Masses)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
    priors[:Darkdim_radius] = LogUniform(ftype(1e-1),ftype(10))
    priors[:λ₁] = Uniform(ftype(0), ftype(10))
    priors[:λ₂] = Uniform(ftype(0), ftype(10))
    priors[:λ₃] = Uniform(ftype(0), ftype(10))
    priors = NamedTuple(priors)
    NamedTuple(priors)
end

function get_PMNS(params)
    T = typeof(params.θ₂₃)
    U1 = SMatrix{3,3}(one(T), zero(T), zero(T), zero(T), cos(params.θ₂₃), -sin(params.θ₂₃), zero(T), sin(params.θ₂₃), cos(params.θ₂₃))
    T = typeof(params.θ₁₃)
    U2 = SMatrix{3,3}(cos(params.θ₁₃), zero(T), -sin(params.θ₁₃)*cis(params.δCP), zero(T), one(T), zero(T), sin(params.θ₁₃)*cis(-params.δCP), zero(T), cos(params.θ₁₃))
    T = typeof(params.θ₁₂)
    U3 = SMatrix{3,3}(cos(params.θ₁₂), -sin(params.θ₁₂), zero(T), sin(params.θ₁₂), cos(params.θ₁₂), zero(T), zero(T), zero(T), one(T))
    U = U1 * U2 * U3
end

function get_abs_masses(params)
    if params.Δm²₃₁ > 0
        m1 = params.m₀
        m2 = sqrt(params.Δm²₂₁ + params.m₀^2)
        m3 = sqrt(params.Δm²₃₁ + params.m₀^2)
    elseif params.Δm²₃₁ < 0
        m1 = sqrt(- params.Δm²₃₁ + params.m₀^2)
        m2 = sqrt(params.Δm²₂₁ - params.Δm²₃₁ + params.m₀^2)
        m3 = params.m₀
    else
        error("Error: Please enter only either 1 for normal or -1 for inverted hierarchy.")
    end
    return m1, m2, m3
end


# Oscillation Kernel Simple
function osc_kernel(U::AbstractMatrix{<:Number}, H::AbstractVector{<:Number}, e::Real, l::Real)
    phase_factors = -F_units * 1im * (l / e) .* H
    U * Diagonal(exp.(phase_factors)) * U'
end

# Oscillation Kernel with Low pass filter
function osc_kernel(U::AbstractMatrix{<:Number}, H::AbstractVector{<:Number}, e::Real, l::Real, σₑ::Real)
    phase_factors = -F_units * (l / e) .* H
    decay = exp.(-2 * abs.(phase_factors) * σₑ^2) #exp.(-abs.(σₑ / e * phase_factors)/2)
    U * Diagonal(exp.(1im * phase_factors) .* decay) * U', decay
end

function compute_matter_matrices(H_eff::AbstractMatrix{<:Number}, e, layer, anti, interaction::SI)
    H = copy(H_eff)
    if anti
        H[1,1] -= A * layer.p_density * 2 * e * 1e9
        for i in 1:3
            H[i,i] += A * layer.n_density * e * 1e9
        end
    else
        H[1,1] += A * layer.p_density * 2 * e * 1e9
        for i in 1:3
            H[i,i] -= A * layer.n_density * e * 1e9
        end
    end
    H = Hermitian(H)
    tmp = eigen(H)
    tmp.vectors, tmp.values
end   

function compute_matter_matrices(H_eff::SMatrix, e, layer, anti, interaction::SI)
    H_mat = zeros(typeof(e), size(H_eff))
    if anti
        H_mat[1,1] -= A * layer.p_density * 2 * e * 1e9
        for i in 1:3
            H_mat[i,i] += A * layer.n_density * e * 1e9
        end
    else
        H_mat[1,1] += A * layer.p_density * 2 * e * 1e9
        for i in 1:3
            H_mat[i,i] -= A * layer.n_density * e * 1e9
        end
    end
    H = Hermitian(H_eff + H_mat)
    tmp = eigen(H)
    #tmp = fast_eigen(H)
    tmp.vectors, tmp.values
end   

function osc_reduce(matter_matrices, path, e, propagation::Damping)
    res = map(section -> osc_kernel(matter_matrices[section.layer_idx]..., e, section.length, propagation.σₑ), path)
    decay = abs2.(reduce(.*, last.(res)))
    # taking an average mixing matrix along the path to compute the decoherent sum, which is a bold approximation
    P_ave  = mean([abs2.(matter_matrices[section.layer_idx][1]) for section in path], weights([section.length for section in path]))
    p = abs2.(reduce(*, first.(res))) .+ P_ave * Diagonal(1 .- decay) * P_ave'
end

function osc_reduce(matter_matrices, path, e, propagation::Basic)
    p = abs2.(mapreduce(section -> osc_kernel(matter_matrices[section.layer_idx]..., e, section.length), *, path))
end
    

function matter_osc_per_e(H_eff, e, layers, paths, anti, propagation::Union{Basic, Damping}, interaction)
    matter_matrices = compute_matter_matrices.(Ref(H_eff), e, layers, anti, Ref(interaction))
    p = stack(map(path -> osc_reduce(matter_matrices, path, e, propagation), paths))
end


function matter_osc_per_e(H_eff, e, layers, paths, anti, propagation::Decoherent, interaction)
    matter_matrices = compute_matter_matrices.(Ref(H_eff), e, layers, anti, Ref(interaction))
    ps = []
    for path in paths
        P = Matrix(abs.(zero(H_eff)))  # P[β, α]
        v = one(H_eff)
        
        for α in 1:size(v)[1]
            # Initial flavor state density matrix |να⟩⟨να|
            eα = v[α, :]
            ρ = eα * eα'
        
            # Propagate through each layer
            for section in path
            #for (L_km, H_flavor) in layer_Hs
                l = section.length
        
                # Diagonalize Hamiltonian
                U, h = matter_matrices[section.layer_idx]
        
                # Step 1: Transform to eigenbasis
                ρ_eig = U' * ρ * U
        
                # Step 2: Coherent evolution
                phases = exp.(-F_units * 1im * (l / e) .* h)
                U_phase = Diagonal(phases)
                ρ_eig = U_phase * ρ_eig * U_phase'
        
                # Step 3: Decoherence damping
                Δφ = abs.(h .- h') * (l / e) * F_units
                D = exp.(-2 .* Δφ .* propagation.σₑ^2)
                ρ_eig .= ρ_eig .* D
        
                # Step 4: Transform back to flavor basis
                ρ = U * ρ_eig * U'
            end
        
            # Fill in transition probabilities to each final flavor β
            for β in 1:size(v)[1]
                eβ = v[β, :]
                P[β, α] = real(eβ' * ρ * eβ)  # P(ν_α → ν_β)
            end
        end
        push!(ps, P)
    end
    p = stack(ps)
end

function select(U, h, cfg::All)
    return U, h, 0.
end

function select(U, h, cfg::Cut)
    mask = sqrt.(abs.(h)) .< cfg.cutoff;
    if any(.!mask)
        h = h[mask];
        U_rest = U[:, .!mask]
        U = U[:, mask];
    else
        U_rest = 0
    end

    return U, h, abs2.(U_rest) * abs2.(U_rest)'
end


function propagate(U, h, E, L, propagation::Basic)
    p = stack(broadcast((e, l) -> abs2.(osc_kernel(U, h, e, l)), E, L'))
end

function propagate(U, h, E, L, propagation::Damping)
    res = broadcast((e, l) -> osc_kernel(U, h, e, l, propagation.σₑ), E, L')
    p = stack(map(x -> abs2.(first(x)) + abs2.(U) * Diagonal(1 .- abs2.(last(x))) * abs2.(U)', res))
end

function propagate(U, h, E, L, propagation::Decoherent)

    function kernel(e,l)
        P = Matrix(abs.(zero(U*U')))  # P[β, α]
        v = one(U*U')
        
        for α in 1:size(v)[1]
            # Initial flavor state density matrix |να⟩⟨να|
            eα = v[α, :]
            ρ = eα * eα'
        
            # Step 1: Transform to eigenbasis
            ρ_eig = U' * ρ * U
    
            # Step 2: Coherent evolution
            phases = exp.(-F_units * 1im * (l / e) .* h)
            U_phase = Diagonal(phases)
            ρ_eig = U_phase * ρ_eig * U_phase'
    
            # Step 3: Decoherence damping
            Δφ = abs.(h .- h') * (l / e) * F_units
            D = exp.(-2 .* Δφ .* propagation.σₑ^2)
            ρ_eig .= ρ_eig .* D
    
            # Step 4: Transform back to flavor basis
            ρ = U * ρ_eig * U'

        
            # Fill in transition probabilities to each final flavor β
            for β in 1:size(v)[1]
                eβ = v[β, :]
                P[β, α] = real(eβ' * ρ * eβ)  # P(ν_α → ν_β)
            end
        end
        P
    end

    res = stack(broadcast((e, l) -> kernel(e, l), E, L'))
end

function propagate(U, h, E, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, propagation::PropagationModel, interaction::Vacuum, anti::Bool)
    L = [sum([segment.length for segment in path]) for path in paths]
    propagate(U, h, E, L, propagation)
end

function propagate(U, h, E, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, propagation::PropagationModel, interaction::Union{SI, NSI}, anti::Bool)
    H_eff = U * Diagonal(h) * adjoint(U)
    p = stack(map(e -> matter_osc_per_e(H_eff, e, layers, paths, anti, propagation, interaction), E))
    permutedims(p, (1, 2, 4, 3))
end

function get_osc_prob(cfg::OscillationConfig)

    function osc_prob(E::AbstractVector{<:Real}, L::AbstractVector{<:Real}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U
    
        U, h, rest = select(Uc, h, cfg.states)
        
        p = propagate(U, h, E, L, cfg.propagation)
            
        # results
        p = p .+ rest
        return permutedims(p, (3, 4, 1, 2))
    end

    function osc_prob(E::AbstractVector{<:Real}, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U
    
        U, h, rest = select(Uc, h, cfg.states)
    
        p = propagate(U, h, E, paths, layers, cfg.propagation, cfg.interaction, anti)
        
        # results
        p = p .+ rest
        return permutedims(p, (3, 4, 1, 2))
    end

    return osc_prob
end


function get_matrices(cfg::ThreeFlavour)
    function matrices(params::NamedTuple)
        U = get_PMNS(params)
        T = promote_type(typeof(params.Δm²₂₁), typeof(params.Δm²₃₁))
        h = @SVector [zero(T), params.Δm²₂₁, params.Δm²₃₁]
        #h = SVector{3, typeof(params.Δm²₃₁)}([0.,params.Δm²₂₁,params.Δm²₃₁])
        return U, h
    end
end

function get_matrices(cfg::ThreeFlavourXYCP)
    function matrices(params::NamedTuple)

        # norm = sqrt(params.δCPy^2 + params.δCPx^2)
        # if norm == 0.
        #     δCP = 0.
        #     #@show params.δCPy, params.δCPx
        # else
        #     δCP = atan(params.δCPy/norm, params.δCPx/norm)
        # end
        δCP = params.δCPshell[1]
        #δCP = angle(params.δCPx + 1im * params.δCPy)
        #@show δCP
        U = get_PMNS(merge(params, (;δCP,)))
        h = SVector{3, typeof(params.Δm²₃₁)}([0.,params.Δm²₂₁,params.Δm²₃₁])
        return U, h
    end
end

function get_matrices(cfg::Sterile)
    function matrices(params::NamedTuple)
        h = [0. ,params.Δm²₂₁, params.Δm²₃₁, params.Δm²₄₁]
     
        R14 = [cos(params.θ₁₄) 0 0 sin(params.θ₁₄); 0 1 0 0; 0 0 1 0; -sin(params.θ₁₄) 0 0 cos(params.θ₁₄)]
        R24 = [1 0 0 0; 0 cos(params.θ₂₄) 0 sin(params.θ₂₄); 0 0 1 0; 0 -sin(params.θ₂₄) 0 cos(params.θ₂₄)]
        R34 = [1 0 0 0; 0 1 0 0; 0 0 cos(params.θ₃₄) sin(params.θ₃₄); 0 0 -sin(params.θ₃₄) cos(params.θ₃₄)]
        
        U = get_PMNS(params)
        
        U_sterile = R34 * R24 * R14 * hcat(vcat(U, [0 0 0]), [0 0 0 1]')
        
        return U_sterile, h
    end
end

function get_matrices(cfg::ADD)
    function matrices(params::NamedTuple)
        
        PMNS = get_PMNS(params)
    
        m1, m2, m3 = get_abs_masses(params)
    
        # MD is the Dirac mass matrix that appears in the Lagrangian.
        MD = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
        aM1 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        aM2 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
    
        # init buffers
        for i in 1:3*(cfg.N_KK+1)
            for j in 1:3*(cfg.N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
    
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.ADD_radius * MD[i, j] * umev
            end
        end
        
        for n in 1:cfg.N_KK
            for i in 1:3
                for j in 1:3
                    aM1[3*n + i, j] = sqrt(2) * params.ADD_radius * MD[i, j] * umev
                end
            end
        end
    
        for i in 1:cfg.N_KK
            aM2[3*i + 1, 3*i + 1] = i
            aM2[3*i + 2, 3*i + 2] = i
            aM2[3*i + 3, 3*i + 3] = i
        end
    
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
    
        h, U = eigen(aaMM)
        h = h / (params.ADD_radius^2 * umev^2.)
        return U, h
    end
end



# module Darkdim
#     using Distributions
#     using DataStructures
#     using ..osc
#     using LinearAlgebra

#     function get_matrices(params)
#         N_KK = 5
        
#         # um to eV
#         umev = 5.067730716156395
#         PMNS = get_PMNS(params)
    
#         m1, m2, m3 = get_abs_masses(params)
    
#         m1_MD = m1 * sqrt((exp(2 * π * params.ca1) - 1) / (2 * π * params.ca1))
#         m2_MD = m2 * sqrt((exp(2 * π * params.ca2) - 1) / (2 * π * params.ca2))
#         m3_MD = m3 * sqrt((exp(2 * π * params.ca3) - 1) / (2 * π * params.ca3))
        
#         #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.
        
#         # Compute MDc00
#         MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
#         # Initialize aM1 matrix
#         aM1 = similar(PMNS, 3*(N_KK+1), 3*(N_KK+1))
#         aM2 = similar(PMNS, 3*(N_KK+1), 3*(N_KK+1))
#         # init buffers
#         for i in 1:3*(N_KK+1)
#             for j in 1:3*(N_KK+1)
#                 aM1[i,j] = 0.
#                 aM2[i,j] = 0.
#             end
#         end
        
#         # Fill in the aM1 matrix for the first term
#         for i in 1:3
#             for j in 1:3
#                 aM1[i, j] = params.Darkdim_radius * MDc00[i, j] * umev
#             end
#         end
    
#         # Update aM1 matrix for the second term
#         for n in 1:N_KK
#             MDcoff = PMNS * Diagonal([
#                 m1_MD * sqrt(n^2 / (n^2 + params.ca1^2)),
#                 m2_MD * sqrt(n^2 / (n^2 + params.ca2^2)),
#                 m3_MD * sqrt(n^2 / (n^2 + params.ca3^2))
#             ]) * adjoint(PMNS)
#             for i in 1:3
#                 for j in 1:3
#                     aM1[3 * n + i, j] = sqrt(2) * params.Darkdim_radius * MDcoff[i, j] * umev
#                 end
#             end
#         end
    
#         # Fill in the aM2 matrix
#         for n in 1:N_KK
#             aMD2 = PMNS * Diagonal([
#                 sqrt(n^2 + params.ca1^2),
#                 sqrt(n^2 + params.ca2^2),
#                 sqrt(n^2 + params.ca3^2)
#             ]) * adjoint(PMNS)
#             for i in 1:3
#                 for j in 1:3
#                     aM2[3 * n + i, 3 * n + j] = aMD2[i, j]
#                 end
#             end
#         end
    
#         aM = copy(aM1) + copy(aM2)
#         aaMM = Hermitian(conj(transpose(aM)) * aM)
    
#         h, U = eigen(aaMM)
#         h = h / (params.Darkdim_radius^2 * umev^2)
    
#         return U, h
#     end


#     osc_prob = make_osc_prob_function(get_matrices)

#     params = OrderedDict(pairs(standard.params))
#     params[:m₀] = ftype(0.01)
#     params[:ca1] = ftype(1e-4)
#     params[:ca2] = ftype(1e-4)
#     params[:ca3] = ftype(1e-4)
#     params[:Darkdim_radius] = ftype(1e-2)
#     params = NamedTuple(params)
   
#     priors = OrderedDict{Symbol, Distribution}(pairs(standard.priors))
#     priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
#     priors[:ca1] = LogUniform(ftype(1e-5), ftype(10))
#     priors[:ca2] = LogUniform(ftype(1e-5), ftype(10))
#     priors[:ca3] = LogUniform(ftype(1e-5), ftype(10))
#     priors[:Darkdim_radius] = LogUniform(ftype(1e-3),ftype(1))
#     priors = NamedTuple(priors)

# end

function get_matrices(cfg::Darkdim_Lambda)
    function matrices(params::NamedTuple)
        MP = 2.435e18 # GeV
        M5 = 1e6 # GeV
        vev = 174e9 # eV
        lambda_list = [params.λ₁, params.λ₂, params.λ₃]
        m1_MD, m2_MD, m3_MD = (vev * M5 / MP) .* lambda_list
  
  
        m1 = m1_MD * (sqrt(2 * π * params.ca1 / (exp(2 * π * params.ca1) - 1)))
        m2 = m2_MD * (sqrt(2 * π * params.ca2 / (exp(2 * π * params.ca2) - 1)))
        m3 = m3_MD * (sqrt(2 * π * params.ca3 / (exp(2 * π * params.ca3) - 1)))
      
        PMNS = get_PMNS(params)    
      
        #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.
      
        # Compute MDc00
        MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
  
        # Initialize aM1 matrix
        aM1 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        aM2 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        # init buffers
        for i in 1:3*(cfg.N_KK+1)
            for j in 1:3*(cfg.N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
      
        # Fill in the aM1 matrix for the first term
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.Darkdim_radius * MDc00[i, j] * umev
            end
        end
  
        # Update aM1 matrix for the second term
        for n in 1:cfg.N_KK
            MDcoff = PMNS * Diagonal([
                m1_MD * sqrt(n^2 / (n^2 + params.ca1^2)),
                m2_MD * sqrt(n^2 / (n^2 + params.ca2^2)),
                m3_MD * sqrt(n^2 / (n^2 + params.ca3^2))
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM1[3 * n + i, j] = sqrt(2) * params.Darkdim_radius * MDcoff[i, j] * umev
                end
            end
        end
  
        # Fill in the aM2 matrix
        for n in 1:cfg.N_KK
            aMD2 = PMNS * Diagonal([
                sqrt(n^2 + params.ca1^2),
                sqrt(n^2 + params.ca2^2),
                sqrt(n^2 + params.ca3^2)
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM2[3 * n + i, 3 * n + j] = aMD2[i, j]
                end
            end
        end
  
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
  
        h, U = eigen(aaMM)
        h = h / (params.Darkdim_radius^2 * umev^2) 
        return U, h
    end
end

function get_matrices(cfg::Darkdim_Masses)

    function get_mass(ca)
        x = 2 * π * ca
        b = x == 0. ? 1. : sqrt(x / (expm1(x)))
    end
    
    cas = LinRange(10, -10, 300)
    masses = get_mass.(cas)
    get_ca = LinearInterpolation(masses, cas; extrapolation_bc=Line())
    
    function matrices(params::NamedTuple)
        MP = 2.435e18 # GeV
        M5 = 1e6 # GeV
        vev = 174e9 # eV
        lambda_list = [params.λ₁, params.λ₂, params.λ₃]
        m1_MD, m2_MD, m3_MD = (vev * M5 / MP) .* lambda_list

        m1, m2, m3 = get_abs_masses(params)
        
        ca1 = get_ca(m1 / m1_MD)
        ca2 = get_ca(m2 / m2_MD)
        ca3 = get_ca(m3 / m3_MD)
        
        PMNS = get_PMNS(params)    
      
        #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.
      
        # Compute MDc00
        MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
  
        # Initialize aM1 matrix
        aM1 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        aM2 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        # init buffers
        for i in 1:3*(cfg.N_KK+1)
            for j in 1:3*(cfg.N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
      
        # Fill in the aM1 matrix for the first term
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.Darkdim_radius * MDc00[i, j] * umev
            end
        end
  
        # Update aM1 matrix for the second term
        for n in 1:cfg.N_KK
            MDcoff = PMNS * Diagonal([
                m1_MD * sqrt(n^2 / (n^2 + ca1^2)),
                m2_MD * sqrt(n^2 / (n^2 + ca2^2)),
                m3_MD * sqrt(n^2 / (n^2 + ca3^2))
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM1[3 * n + i, j] = sqrt(2) * params.Darkdim_radius * MDcoff[i, j] * umev
                end
            end
        end
  
        # Fill in the aM2 matrix
        for n in 1:cfg.N_KK
            aMD2 = PMNS * Diagonal([
                sqrt(n^2 + ca1^2),
                sqrt(n^2 + ca2^2),
                sqrt(n^2 + ca3^2)
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM2[3 * n + i, 3 * n + j] = aMD2[i, j]
                end
            end
        end
  
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
  
        h, U = eigen(aaMM)
        h = h / (params.Darkdim_radius^2 * umev^2) 
        return U, h
    end
end

end
