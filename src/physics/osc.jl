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
export Decoherent, Damping, Basic, Spray
export All, Cut
export Vacuum, SI, NSI
export ThreeFlavour, ThreeFlavourXYCP, Sterile, ADD
export OscillationConfig
export EigenMethod, DefaultEigen, decompose
export configure

const ftype = Float64

# struct for matter layers
struct Layer{T, U}
    radius::T
    p_density::U
    n_density::U
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
@kwdef struct Spray <: PropagationModel
    averaging::Symbol = :gaussian  # :gaussian or :uniform (sinc)
    σ_E::Float64 = 0.15           # fractional energy smearing (ΔE/E)
    σ_h::Float64 = 10.0           # production height uncertainty [km]
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

abstract type EigenMethod end
struct DefaultEigen <: EigenMethod end

decompose(H::Hermitian, ::DefaultEigen) = eigen(H)
export DefaultEigen

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

@kwdef struct Darkdim_cas <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
    N_KK::Int = 5
end

@kwdef struct OscillationConfig{F<:FlavourModel, I<:InteractionModel, P<:PropagationModel, S<:StateSelector, E<:EigenMethod}
    flavour::F = ThreeFlavour()
    interaction::I = Vacuum()
    propagation::P = Basic()
    states::S = All()
    eigen_method::E = Newtrinos.BargerEigen()
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
        matrices = get_matrices(cfg.flavour, cfg.eigen_method),
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
    priors[:θ₁₃] = Uniform(ftype(0.05), ftype(0.3))
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
    priors[:λ₁] = Uniform(ftype(0), ftype(1))
    priors[:λ₂] = Uniform(ftype(0), ftype(1))
    priors[:λ₃] = Uniform(ftype(0), ftype(1))
    priors = NamedTuple(priors)
    NamedTuple(priors)
end

function get_params(cfg::Darkdim_cas)
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    params[:m₀] = ftype(0.01)
    params[:Darkdim_radius] = 0.1
    params[:ca1] = ftype(1e-5)
    params[:ca2] = ftype(1e-5)
    params[:ca3] = ftype(1e-5)
    NamedTuple(params)
end

function get_priors(cfg::Darkdim_cas)
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
    priors[:Darkdim_radius] = LogUniform(ftype(1e-1),ftype(10))
    priors[:ca1] = Uniform(ftype(1e-5), ftype(10))
    priors[:ca2] = Uniform(ftype(1e-5), ftype(10))
    priors[:ca3] = Uniform(-ftype(10), -ftype(1e-5))
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

function compute_matter_matrices(H_eff::AbstractMatrix{<:Number}, e, layer, anti, interaction::SI, eigen_method::EigenMethod=DefaultEigen())
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
    tmp = decompose(H, eigen_method)
    tmp.vectors, tmp.values
end

function compute_matter_matrices(H_eff::SMatrix{3,3}, e, layer, anti, interaction::SI, eigen_method::EigenMethod=DefaultEigen())
    ve = A * e * 1e9
    if anti
        d1 = ve * (-2 * layer.p_density + layer.n_density)
        dn = ve * layer.n_density
    else
        d1 = ve * (2 * layer.p_density - layer.n_density)
        dn = ve * (-layer.n_density)
    end
    z = zero(d1)
    H_mat = @SMatrix [d1 z z; z dn z; z z dn]
    H = Hermitian(H_eff + H_mat)
    tmp = decompose(H, eigen_method)
    tmp.vectors, tmp.values
end

# --- Spray (ray-to-spray) averaging helpers ---

# Normalized sinc: sinc(x) = sin(πx)/(πx), but here we use unnormalized: sin(x)/x
function _sinc_unnorm(x)
    abs(x) < 1e-8 ? one(x) - x^2/6 : sin(x) / x
end

# (exp(ix) - 1) / (ix) with Taylor expansion for small x (AD-safe)
function _safe_C(x)
    if abs(x) < 1e-6
        # Taylor: 1 + ix/2 - x²/6 - ix³/24 + ...
        return complex(one(real(x))) + 1im * x / 2 - x^2 / 6 - 1im * x^3 / 24
    else
        return (exp(1im * x) - 1) / (1im * x)
    end
end

# dV/dE for standard matter interactions (SI) — same structure as V but without the E factor
function compute_dVdE(layer, anti, interaction::SI, ::Val{3})
    ve = A * 1e9
    if anti
        d1 = ve * (-2 * layer.p_density + layer.n_density)
        dn = ve * layer.n_density
    else
        d1 = ve * (2 * layer.p_density - layer.n_density)
        dn = ve * (-layer.n_density)
    end
    z = zero(d1)
    @SMatrix [d1 z z; z dn z; z z dn]
end

# Generic fallback for non-SMatrix sizes
function compute_dVdE(layer, anti, interaction::SI, ::Val{N}) where N
    ve = A * 1e9
    dVdE = zeros(typeof(ve), N, N)
    if anti
        dVdE[1,1] = ve * (-2 * layer.p_density + layer.n_density)
        for i in 2:N
            dVdE[i,i] = ve * layer.n_density
        end
    else
        dVdE[1,1] = ve * (2 * layer.p_density - layer.n_density)
        for i in 2:N
            dVdE[i,i] = ve * (-layer.n_density)
        end
    end
    dVdE
end

# Compute (S̄, K_E, K_Θ) for a single constant-density layer
# K_E encodes energy perturbation; K_Θ encodes path-length (zenith) perturbation
function compute_spray_layer(U_layer, h_layer, dVdE, e, l, dldcz)
    n = length(h_layer)

    # Effective frequencies: ω_n = F_units · h_n / E
    omega = F_units .* h_layer ./ e

    # Evolution matrix S̄
    S = U_layer * Diagonal(exp.(-1im .* omega .* l)) * U_layer'

    # --- K_E (energy perturbation) ---
    # H'_E in eigenbasis: dω/dE = F_units/E · dV/dE_eig - diag(ω)/E
    dVdE_eig = U_layer' * dVdE * U_layer
    H_prime = F_units / e .* dVdE_eig - Diagonal(omega ./ e)

    # C matrix (Eq. 2.5): C_ij = (exp(i·Δω·L)-1)/(i·Δω·L)
    K_E_eig = SMatrix{n,n}(
        ntuple(n*n) do idx
            i, j = (idx - 1) % n + 1, (idx - 1) ÷ n + 1
            x = (omega[i] - omega[j]) * l
            l * H_prime[i, j] * _safe_C(x)
        end
    )
    K_E = U_layer * K_E_eig * U_layer'

    # --- K_Θ (zenith/path-length perturbation) ---
    # Only L changes with cosθ, not the Hamiltonian. K_Θ is diagonal in eigenbasis:
    # K̃_Θ = diag(ω_i · dL/dcosθ)
    K_Theta = U_layer * Diagonal(omega .* dldcz) * U_layer'

    return S, K_E, K_Theta
end

# Multi-layer composition for Spray (Eq. 2.9): S_total = S_N · ... · S_1 (physical order)
# Both K_E and K_Θ follow the same composition rule: K_combined = S_acc†·K_new·S_acc + K_acc
function osc_reduce(matter_matrices, spray_data, path, e, propagation::Spray, dldcz_path)
    sec = first(path)
    U1, h1 = matter_matrices[sec.layer_idx]
    S_acc, KE_acc, KT_acc = compute_spray_layer(U1, h1, spray_data[sec.layer_idx], e, sec.length, dldcz_path[1])

    for i in 2:length(path)
        sec = path[i]
        U_n, h_n = matter_matrices[sec.layer_idx]
        S_n, KE_n, KT_n = compute_spray_layer(U_n, h_n, spray_data[sec.layer_idx], e, sec.length, dldcz_path[i])
        KE_acc = S_acc' * KE_n * S_acc + KE_acc
        KT_acc = S_acc' * KT_n * S_acc + KT_acc
        S_acc = S_n * S_acc
    end

    return S_acc, KE_acc, KT_acc
end

# Compute damping factor for given x and averaging type
_spray_damping(x, averaging) = averaging === :gaussian ? exp(-x^2 / 2) : _sinc_unnorm(x / 2)

# Diagonalize K_E (and optionally K_Θ) and compute bin-averaged oscillation probabilities.
# When Delta_CZ > 0: joint E+Θ averaging via density-matrix formalism (handles non-commuting K).
function spray_average(S, K_E, K_Theta, Delta_E, Delta_CZ, averaging::Symbol, eigen_method::EigenMethod=DefaultEigen())
    n = size(S, 1)

    # Diagonalize K_E
    K_E_herm = Hermitian((K_E + K_E') / 2)
    decomp_E = decompose(K_E_herm, eigen_method)
    V_E = SMatrix{n,n}(decomp_E.vectors)
    λ_E = SVector{n}(real.(decomp_E.values))
    SV = S * V_E

    # Energy damping matrix
    G_E = SMatrix{n,n}(
        ntuple(n*n) do idx
            i, j = (idx - 1) % n + 1, (idx - 1) ÷ n + 1
            _spray_damping((λ_E[i] - λ_E[j]) * Delta_E, averaging)
        end
    )

    if iszero(Delta_CZ)
        # E-only averaging (fast path)
        P = SMatrix{n,n}(
            ntuple(n*n) do idx
                β, α = (idx - 1) % n + 1, (idx - 1) ÷ n + 1
                s = zero(eltype(G_E))
                for j in 1:n, i in 1:n
                    s += real(SV[β,i] * conj(V_E[α,i]) * V_E[α,j] * conj(SV[β,j]) * G_E[i,j])
                end
                s
            end
        )
        return P
    end

    # Joint E+Θ averaging: transform K_Θ into K_E eigenbasis, diagonalize there
    K_Theta_VE = V_E' * Hermitian((K_Theta + K_Theta') / 2) * V_E
    decomp_Θ = decompose(Hermitian((K_Theta_VE + K_Theta_VE') / 2), eigen_method)
    W = SMatrix{n,n}(decomp_Θ.vectors)
    λ_Θ = SVector{n}(real.(decomp_Θ.values))

    # Zenith damping matrix (in W basis within V_E basis)
    G_Θ = SMatrix{n,n}(
        ntuple(n*n) do idx
            i, j = (idx - 1) % n + 1, (idx - 1) ÷ n + 1
            _spray_damping((λ_Θ[i] - λ_Θ[j]) * Delta_CZ, averaging)
        end
    )

    # Density-matrix formalism: for each input flavour α,
    # apply E damping in V_E basis, then Θ damping in W basis
    CT = complex(eltype(G_E))
    P = MMatrix{n,n,eltype(G_E)}(undef)
    for α in 1:n
        # Compute A = W† ρ_E W  (3×3 matrix operations)
        A = MMatrix{n,n,CT}(undef)
        for s in 1:n, r in 1:n
            a = zero(CT)
            for q in 1:n, p in 1:n
                a += conj(W[p, r]) * W[q, s] * G_E[p, q] * conj(V_E[α, p]) * V_E[α, q]
            end
            A[r, s] = a
        end

        # ρ_EΘ = W (G_Θ ⊙ A) W†  (back to V_E basis)
        ρ = MMatrix{n,n,CT}(undef)
        for q in 1:n, p in 1:n
            v = zero(CT)
            for s in 1:n, r in 1:n
                v += W[p, r] * conj(W[q, s]) * G_Θ[r, s] * A[r, s]
            end
            ρ[p, q] = v
        end

        # P[β,α] = [(SV) ρ_EΘ (SV)†]_ββ
        for β in 1:n
            s = zero(eltype(G_E))
            for q in 1:n, p in 1:n
                s += real(SV[β, p] * ρ[p, q] * conj(SV[β, q]))
            end
            P[β, α] = s
        end
    end
    return SMatrix(P)
end

function matter_osc_per_e(H_eff, e, layers, paths, anti, propagation::Spray, interaction::SI,
                           eigen_method::EigenMethod=DefaultEigen();
                           Delta_E=zero(e), Delta_h=zero(e), dldh_all=nothing)
    matter_matrices = compute_matter_matrices.(Ref(H_eff), e, layers, anti, Ref(interaction), Ref(eigen_method))
    n_flav = size(H_eff, 1)
    spray_data = map(layer -> compute_dVdE(layer, anti, interaction, Val(n_flav)), layers)

    n_paths = length(paths)
    p = stack(map(1:n_paths) do idx
        path = paths[idx]
        dldh_path = dldh_all !== nothing ? dldh_all[idx] : zeros(length(path))
        S, K_E, K_Theta = osc_reduce(matter_matrices, spray_data, path, e, propagation, dldh_path)
        spray_average(S, K_E, K_Theta, Delta_E, Delta_h, propagation.averaging, eigen_method)
    end)
end

function osc_reduce(matter_matrices, path, e, propagation::Damping)
    res = map(section -> osc_kernel(matter_matrices[section.layer_idx]..., e, section.length, propagation.σₑ), path)
    decay = abs2.(reduce(.*, last.(res)))
    # taking an average mixing matrix along the path to compute the decoherent sum, which is a bold approximation
    w = weights([section.length for section in path])
    P_ave  = mean([abs2.(matter_matrices[section.layer_idx][1]) for section in path], w)
    # Physical order: S_N · ... · S_1 (later layers multiply from the left)
    S_matrices = first.(res)
    S_total = S_matrices[1]
    for i in 2:length(S_matrices)
        S_total = S_matrices[i] * S_total
    end
    p = abs2.(S_total) .+ P_ave * Diagonal(1 .- decay) * P_ave'
end

function osc_reduce(matter_matrices, path, e, propagation::Basic)
    # Physical order: S_total = S_N · ... · S_1 (later layers multiply from the left)
    # Path is entry→exit, so each new section's S multiplies from the left
    sec = first(path)
    S = osc_kernel(matter_matrices[sec.layer_idx]..., e, sec.length)
    for sec in Iterators.drop(path, 1)
        S = osc_kernel(matter_matrices[sec.layer_idx]..., e, sec.length) * S
    end
    abs2.(S)
end
    

function matter_osc_per_e(H_eff, e, layers, paths, anti, propagation::Union{Basic, Damping}, interaction, eigen_method::EigenMethod=DefaultEigen())
    matter_matrices = compute_matter_matrices.(Ref(H_eff), e, layers, anti, Ref(interaction), Ref(eigen_method))
    p = stack(map(path -> osc_reduce(matter_matrices, path, e, propagation), paths))
end


function matter_osc_per_e(H_eff, e, layers, paths, anti, propagation::Decoherent, interaction, eigen_method::EigenMethod=DefaultEigen())
    matter_matrices = compute_matter_matrices.(Ref(H_eff), e, layers, anti, Ref(interaction), Ref(eigen_method))
    n = size(H_eff, 1)
    RT = real(eltype(H_eff))
    CT = eltype(H_eff)
    ps = Matrix{RT}[]
    for path in paths
        P = zeros(RT, n, n)  # P[β, α]

        for α in 1:n
            # Initial flavor state density matrix |να⟩⟨να| = sparse, only (α,α) = 1
            # Start with identity-like ρ: zero everywhere except ρ[α,α] = 1
            ρ = zeros(CT, n, n)
            ρ[α, α] = one(CT)

            # Propagate through each layer
            for section in path
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
                ρ_eig = ρ_eig .* D

                # Step 4: Transform back to flavor basis
                ρ = U * ρ_eig * U'
            end

            # P[β, α] = real(ρ[β, β]) since eβ is a standard basis vector
            for β in 1:n
                P[β, α] = real(ρ[β, β])
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
    mask = sqrt.(abs.(h)) .< cfg.cutoff
    notmask = .!mask
    if any(notmask)
        h = h[mask]
        U_rest = U[:, notmask]
        U = U[:, mask]
    else
        U_rest = U[:, Int[]]
    end

    return U, h, abs2.(U_rest) * abs2.(U_rest)'
end


function propagate(U, h, E, L, propagation::Basic)
    n = size(U, 1)
    RT = real(promote_type(eltype(U), eltype(h), eltype(E), eltype(L)))
    # Write directly in (n_flav, n_flav, n_E, n_L) layout — avoids permutedims
    p = Array{RT}(undef, n, n, length(E), length(L))
    for (j, l) in enumerate(L), (i, e) in enumerate(E)
        result = abs2.(osc_kernel(U, h, e, l))
        for b in 1:n, a in 1:n
            p[a, b, i, j] = result[a, b]
        end
    end
    p
end

function propagate(U, h, E, L, propagation::Damping)
    n = size(U, 1)
    RT = real(promote_type(eltype(U), eltype(h), eltype(E), eltype(L)))
    U2 = abs2.(U)
    p = Array{RT}(undef, n, n, length(E), length(L))
    for (j, l) in enumerate(L), (i, e) in enumerate(E)
        amp, decay = osc_kernel(U, h, e, l, propagation.σₑ)
        result = abs2.(amp) + U2 * Diagonal(1 .- abs2.(decay)) * U2'
        for b in 1:n, a in 1:n
            p[a, b, i, j] = result[a, b]
        end
    end
    p
end

function propagate(U, h, E, L, propagation::Decoherent)
    n = size(U, 1)
    RT = real(promote_type(eltype(U), eltype(h), eltype(E), eltype(L)))

    p = Array{RT}(undef, n, n, length(E), length(L))

    for (j, l) in enumerate(L), (i, e) in enumerate(E)
        # Precompute phase and damping (same for all α)
        phases = exp.(-F_units * 1im * (l / e) .* h)
        U_phase = Diagonal(phases)
        Δφ = abs.(h .- h') * (l / e) * F_units
        D = exp.(-2 .* Δφ .* propagation.σₑ^2)

        for α in 1:n
            # ρ_eig = U' * |α⟩⟨α| * U, so ρ_eig[i,j] = conj(U[α,i]) * U[α,j]
            ρ_eig = U[α:α, :]' * U[α:α, :]

            # Coherent evolution
            ρ_eig = U_phase * ρ_eig * U_phase'

            # Decoherence damping
            ρ_eig = ρ_eig .* D

            # Transform back to flavor basis
            ρ = U * ρ_eig * U'

            # P[β, α] = real(ρ[β, β]) since eβ is a standard basis vector
            for β in 1:n
                p[β, α, i, j] = real(ρ[β, β])
            end
        end
    end
    p
end

function propagate(U, h, E, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, propagation::PropagationModel, interaction::Vacuum, anti::Bool)
    L = [sum(segment.length for segment in path) for path in paths]
    propagate(U, h, E, L, propagation)
end

function propagate(U, h, E, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, propagation::PropagationModel, interaction::Union{SI, NSI}, anti::Bool, eigen_method::EigenMethod=DefaultEigen())
    # U is already conj(U_PMNS) for antineutrinos, so this gives:
    #   neutrino:     U_PMNS  × diag(h) × U_PMNS†
    #   antineutrino: U_PMNS* × diag(h) × U_PMNS^T
    H_eff = U * Diagonal(h) * adjoint(U)
    p = stack(map(e -> matter_osc_per_e(H_eff, e, layers, paths, anti, propagation, interaction, eigen_method), E))
    permutedims(p, (1, 2, 4, 3))
end

function propagate(U, h, E, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, propagation::Spray, interaction::Union{SI, NSI}, anti::Bool, eigen_method::EigenMethod=DefaultEigen())
    H_eff = U * Diagonal(h) * adjoint(U)
    # Production height: only first (atmosphere) section varies.
    # dL/dh = 1/cos(α) where α is the angle between the path and the radial
    # direction at the production point. From the cosine rule (triangle with
    # sides R_atm, R_det, L_atm):
    #   cos(α) = (R_atm² + L_atm² - R_det²) / (2·R_atm·L_atm)
    R_atm = layers.radius[1]  # atmosphere outer radius
    R_det = layers.radius[2]  # next layer below atmosphere
    dldh = map(paths) do p
        L_atm = p[1].length
        if L_atm > 1e-3
            cos_alpha = (R_atm^2 + L_atm^2 - R_det^2) / (2 * R_atm * L_atm)
            dldh_val = 1.0 / max(cos_alpha, 1e-3)
        else
            dldh_val = 1.0
        end
        vcat([dldh_val], zeros(length(p) - 1))
    end
    p = stack(map((e, de) -> matter_osc_per_e(H_eff, e, layers, paths, anti, propagation, interaction, eigen_method; Delta_E=de, Delta_h=propagation.σ_h, dldh_all=dldh), E, propagation.σ_E .* E))
    permutedims(p, (1, 2, 4, 3))
end

# Resolve Delta_E argument: nothing→zeros, scalar→broadcast, vector→pass through
function _resolve_delta_E(Delta_E::Nothing, E)
    zeros(eltype(E), length(E))
end
function _resolve_delta_E(Delta_E::Real, E)
    fill(Delta_E, length(E))
end
function _resolve_delta_E(Delta_E::AbstractVector, E)
    Delta_E
end

# Fuse rest addition + permutedims + flavour transpose into one pass.
# p_raw layout: [out, in, n_E, n_L] (from propagate, where out=detected, in=source)
# result layout: [n_E, n_L, in, out] so that P[i, j, α, β] = P(να → νβ)
function _add_rest_and_permute(p_raw, rest)
    n1, n2, n3, n4 = size(p_raw)
    result = similar(p_raw, n3, n4, n1, n2)
    @inbounds for b in 1:n2, a in 1:n1, j in 1:n4, i in 1:n3
        result[i, j, b, a] = p_raw[a, b, i, j] + (rest isa AbstractArray ? rest[a, b] : rest)
    end
    result
end

function get_osc_prob(cfg::OscillationConfig)

    # Returns P[i, j, α, β] = P(να → νβ), i.e.:
    #   3rd index = input (source) flavour
    #   4th index = output (detected) flavour
    #   Flavour indices: 1=νe, 2=νμ, 3=ντ
    #
    # Example: P[:, :, 2, 1] = P(νμ → νe) — probability of detecting νe given initial νμ
    # Probability conservation: sum(P[i, j, α, :]) ≈ 1 for any input flavour α.

    function osc_prob(E::AbstractVector{<:Real}, L::AbstractVector{<:Real}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour, cfg.eigen_method)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U

        U, h, rest = select(Uc, h, cfg.states)

        # propagate returns p_raw[out, in, n_E, n_L]
        p_raw = propagate(U, h, E, L, cfg.propagation)

        # fuse rest addition + permutedims into P[n_E, n_L, in, out]
        return _add_rest_and_permute(p_raw, rest)
    end

    function osc_prob(E::AbstractVector{<:Real}, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour, cfg.eigen_method)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U

        U, h, rest = select(Uc, h, cfg.states)

        # propagate returns p_raw[out, in, n_E, n_cz]
        p_raw = propagate(U, h, E, paths, layers, cfg.propagation, cfg.interaction, anti, cfg.eigen_method)

        # fuse rest addition + permutedims into P[n_E, n_cz, in, out]
        return _add_rest_and_permute(p_raw, rest)
    end

    return osc_prob
end


function get_matrices(cfg::ThreeFlavour, eigen_method::EigenMethod=DefaultEigen())
    function matrices(params::NamedTuple)
        U = get_PMNS(params)
        T = promote_type(typeof(params.Δm²₂₁), typeof(params.Δm²₃₁))
        h = @SVector [zero(T), params.Δm²₂₁, params.Δm²₃₁]
        #h = SVector{3, typeof(params.Δm²₃₁)}([0.,params.Δm²₂₁,params.Δm²₃₁])
        return U, h
    end
end

function get_matrices(cfg::ThreeFlavourXYCP, eigen_method::EigenMethod=DefaultEigen())
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

function get_matrices(cfg::Sterile, eigen_method::EigenMethod=DefaultEigen())
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

function get_matrices(cfg::ADD, eigen_method::EigenMethod=DefaultEigen())
    function matrices(params::NamedTuple)
        
        PMNS = get_PMNS(params)
    
        m1, m2, m3 = get_abs_masses(params)
    
        # MD is the Dirac mass matrix that appears in the Lagrangian.
        MD = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
        aM1 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        aM2 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))

        fill!(aM1, zero(eltype(aM1)))
        fill!(aM2, zero(eltype(aM2)))

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

        aM = aM1 + aM2
        aaMM = Hermitian(conj(transpose(aM)) * aM)
    
        h, U = decompose(aaMM, eigen_method)
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
    
#         h, U = decompose(aaMM, eigen_method)
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

function get_matrices(cfg::Darkdim_Lambda, eigen_method::EigenMethod=DefaultEigen())
    function matrices(params::NamedTuple)
        MP = 2.435e18 # GeV
        #M5 = 1e6 # GeV
        M5 = 1.055e9 * (1/(2π * params.Darkdim_radius))^(1/3) # GeV
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
        fill!(aM1, zero(eltype(aM1)))
        fill!(aM2, zero(eltype(aM2)))

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

        aM = aM1 + aM2
        aaMM = Hermitian(conj(transpose(aM)) * aM)

        h, U = decompose(aaMM, eigen_method)
        h = h / (params.Darkdim_radius^2 * umev^2)
        return U, h
    end
end

function get_matrices(cfg::Darkdim_Masses, eigen_method::EigenMethod=DefaultEigen())

    function get_mass(ca)
        x = 2 * π * ca
        b = x == 0. ? 1. : sqrt(x / (expm1(x)))
    end

    cas = LinRange(10, -10, 300)
    masses = get_mass.(cas)
    get_ca = LinearInterpolation(masses, cas; extrapolation_bc=Line())

    function matrices(params::NamedTuple)
        MP = 2.435e18 # GeV
        M5 = 1.055e9 * (1/(2π * params.Darkdim_radius))^(1/3) # GeV
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
        fill!(aM1, zero(eltype(aM1)))
        fill!(aM2, zero(eltype(aM2)))

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

        aM = aM1 + aM2
        aaMM = Hermitian(conj(transpose(aM)) * aM)

        h, U = decompose(aaMM, eigen_method)
        h = h / (params.Darkdim_radius^2 * umev^2)
        return U, h
    end
end

function get_matrices(cfg::Darkdim_cas, eigen_method::EigenMethod=DefaultEigen())

    function get_lambda(ca, m)
        MP = 2.435e18 # GeV
        M5 = 1e6 # GeV
        vev = 174e9 # eV
        MD = (vev * M5 / MP)
        x = 2 * π * ca
        b = iszero(x) ? one(x) : sqrt(x / (expm1(x)))
        m / (MD * b)
    end

    function matrices(params::NamedTuple)
        MP = 2.435e18 # GeV
        M5 = 1e6 # GeV
        vev = 174e9 # eV

        m1, m2, m3 = get_abs_masses(params)

        ca1 = params.ca1
        ca2 = params.ca2
        ca3 = params.ca3

        λ₁ = get_lambda(ca1, m1)
        λ₂ = get_lambda(ca2, m2)
        λ₃ = get_lambda(ca3, m3)

        lambda_list = [λ₁, λ₂, λ₃]

        m1_MD, m2_MD, m3_MD = (vev * M5 / MP) .* lambda_list

        PMNS = get_PMNS(params)

        #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.

        # Compute MDc00
        MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)

        # Initialize aM1 matrix
        aM1 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        aM2 = similar(PMNS, 3*(cfg.N_KK+1), 3*(cfg.N_KK+1))
        fill!(aM1, zero(eltype(aM1)))
        fill!(aM2, zero(eltype(aM2)))

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

        aM = aM1 + aM2
        aaMM = Hermitian(conj(transpose(aM)) * aM)

        h, U = decompose(aaMM, eigen_method)
        h = h / (params.Darkdim_radius^2 * umev^2)
        return U, h
    end
end
end
