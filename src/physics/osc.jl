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
export EigenMethod, DefaultEigen, decompose
export configure

"""
    ftype

Type alias for `Float64`, used as the default floating-point precision throughout the
oscillation module.
"""
const ftype = Float64

"""
    Layer{T}

A spherical shell of matter with uniform proton and neutron number densities.

Used to represent concentric layers of the Earth in the PREM density model
(see [`Newtrinos.earth_layers.EarthLayers`](@ref) in `earth_layers.jl`). A neutrino trajectory through the
Earth is described by a sequence of [`Path`](@ref) segments, each referencing one layer.

# Fields
- `radius::T`: outer radius of the shell [km].
- `p_density::T`: proton number density [mol/cm``^3``].
- `n_density::T`: neutron number density [mol/cm``^3``].
"""
struct Layer{T}
    radius::T
    p_density::T
    n_density::T
end

"""
    Path

A single segment of a neutrino propagation path through one matter [`Layer`](@ref).

A full baseline is represented as a `Vector{Path}`, one entry per layer traversed.
For vacuum oscillations the total baseline is simply `sum(segment.length for segment in path)`.

# Fields
- `length::Float64`: segment length [km].
- `layer_idx::Int`: index into the corresponding `StructVector{Layer}`.
"""
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

"""
    PropagationModel

Abstract type governing how neutrino quantum states evolve along the baseline.

Subtypes select different physical approximations for the oscillation amplitude:
- [`Basic`](@ref): standard coherent quantum-mechanical propagation.
- [`Decoherent`](@ref): density-matrix evolution with off-diagonal damping.
- [`Damping`](@ref): amplitude-level low-pass filter with incoherent recovery.
"""
abstract type PropagationModel end

"""
    Basic <: PropagationModel

Standard coherent quantum-mechanical propagation (no decoherence effects).

The oscillation amplitude is computed as

```math
A_{\\alpha\\beta} = \\bigl(U \\, \\mathrm{diag}\\bigl(e^{-i\\,\\Delta m^2_j\\, L \\,/\\, 4E}\\bigr) \\, U^\\dagger\\bigr)_{\\alpha\\beta}
```

and the transition probability is ``P_{\\alpha\\beta} = |A_{\\alpha\\beta}|^2``.
"""
struct Basic <: PropagationModel end

"""
    Decoherent <: PropagationModel

Propagation with energy-dependent decoherence via density-matrix evolution.

Off-diagonal elements of the density matrix in the mass eigenbasis are damped by

```math
D_{ij} = \\exp\\!\\bigl(-2\\,|\\Delta\\phi_{ij}|\\,\\sigma_e^2\\bigr), \\qquad
\\Delta\\phi_{ij} = \\frac{\\Delta m^2_{ij}\\, L}{4E}
```

after coherent phase evolution at each layer crossing.

# Fields
- `σₑ::Float64 = 0.1`: decoherence strength parameter (dimensionless).
"""
@kwdef struct Decoherent <: PropagationModel
    σₑ::Float64=0.1
end

"""
    Damping <: PropagationModel

Propagation with a low-pass filter that damps fast-oscillating amplitudes.

Each mass-eigenstate phase factor is multiplied by a decay
``d_j = \\exp(-2\\,|\\phi_j|\\,\\sigma_e^2)`` and the probability is corrected by an
incoherent recovery term ``|U|^2 \\, \\mathrm{diag}(1-d_j^2) \\, |U|^{2\\,\\prime}``.

# Fields
- `σₑ::Float64 = 0.1`: damping strength parameter (dimensionless).
"""
@kwdef struct Damping <: PropagationModel
    σₑ::Float64=0.1
end

"""
    StateSelector

Abstract type controlling which mass eigenstates contribute to the coherent oscillation
calculation. Subtypes:
- [`All`](@ref): include every eigenstate (default).
- [`Cut`](@ref): exclude eigenstates above a mass-squared cutoff.
"""
abstract type StateSelector end

"""
    All <: StateSelector

Include all mass eigenstates in the oscillation calculation. This is the default.
"""
struct All <: StateSelector end

"""
    Cut <: StateSelector

Exclude mass eigenstates whose ``\\sqrt{|m^2_j|}`` exceeds a cutoff threshold.

States above the cutoff are averaged incoherently, which is useful for
[`ADD`](@ref) or dark-dimension models where heavy Kaluza-Klein states oscillate
too rapidly to be resolved experimentally.

# Fields
- `cutoff::Float64 = Inf`: mass eigenvalue cutoff [eV]. Default `Inf` keeps all states.
"""
@kwdef struct Cut <: StateSelector
    cutoff::Float64 = Inf
end

"""
    InteractionModel

Abstract type selecting the neutrino-matter interaction model used when propagating
through Earth layers. Subtypes:
- [`Vacuum`](@ref): no matter effects.
- [`SI`](@ref): Standard Interactions (MSW effect).
- [`NSI`](@ref): Non-Standard Interactions.
"""
abstract type InteractionModel end

"""
    Vacuum <: InteractionModel

Vacuum oscillations — no matter effects. Layer densities in [`Layer`](@ref) are ignored;
the baseline is the sum of [`Path`](@ref) segment lengths.
"""
struct Vacuum <: InteractionModel end

"""
    NSI <: InteractionModel

Non-Standard Interactions in matter. Extends the matter Hamiltonian beyond the Standard
Model Wolfenstein potential.
"""
struct NSI <: InteractionModel end

"""
    SI <: InteractionModel

Standard Interactions — coherent forward scattering in matter (the MSW effect).

Adds the Wolfenstein charged-current potential
``V_{CC} = \\sqrt{2}\\, G_F\\, N_e`` to the ``(\\nu_e, \\nu_e)`` element and a
neutral-current potential to all diagonal elements of the effective Hamiltonian.
The sign is flipped for antineutrinos.
"""
struct SI <: InteractionModel end

"""
    EigenMethod

Abstract type selecting the eigendecomposition algorithm for the effective matter
Hamiltonian. Subtypes:
- [`DefaultEigen`](@ref): Julia's built-in `LinearAlgebra.eigen`.
- [`BargerEigen`](@ref) (defined in `barger_eigen.jl`): fast analytic 3×3 decomposition.
"""
abstract type EigenMethod end

"""
    DefaultEigen <: EigenMethod

Use Julia's built-in `LinearAlgebra.eigen` for Hermitian matrix diagonalization.
"""
struct DefaultEigen <: EigenMethod end

"""
    decompose(H::Hermitian, method::EigenMethod) -> Eigen

Eigendecompose the Hermitian Hamiltonian `H` using the selected [`EigenMethod`](@ref).

This is the extension point for custom eigensolvers. To add a new method, define a
subtype of [`EigenMethod`](@ref) and a corresponding `decompose` dispatch.

# Arguments
- `H::Hermitian`: effective Hamiltonian matrix (mass-squared basis or matter-modified).
- `method::EigenMethod`: algorithm selector.

# Returns
An `Eigen` factorization with fields `vectors` and `values`.
"""
decompose(H::Hermitian, ::DefaultEigen) = eigen(H)
export DefaultEigen

"""
    FlavourModel

Abstract type selecting the neutrino mass/flavour model.

Each subtype defines its own set of oscillation parameters (mixing angles, mass splittings,
and any BSM quantities) via `get_params` and `get_priors` dispatches, as well as a
`get_matrices` method that returns the mixing matrix ``U`` and mass-squared eigenvalues.

Standard and BSM subtypes:
- [`ThreeFlavour`](@ref): standard three-neutrino oscillations.
- [`ThreeFlavourXYCP`](@ref): three-flavour with shell parametrization of ``\\delta_{CP}``.
- [`Sterile`](@ref): 3+1 sterile neutrino model.
- [`ADD`](@ref): Arkani-Hamed–Dimopoulos–Dvali large extra dimensions.
- `Darkdim_Lambda`, `Darkdim_Masses`, `Darkdim_cas`: dark-dimension model variants.
"""
abstract type FlavourModel end

"""
    ThreeFlavour <: FlavourModel

Standard three-flavour neutrino oscillation model.

The PMNS mixing matrix is constructed as
``U_{PMNS} = R_{23}(\\theta_{23}) \\cdot R_{13}(\\theta_{13},\\, \\delta_{CP}) \\cdot R_{12}(\\theta_{12})``
and the oscillation parameters are:

| Parameter   | Description                          | Default           |
|:----------- |:------------------------------------ |:----------------- |
| `θ₁₂`      | Solar mixing angle                   | ``\\arcsin(\\sqrt{0.307})`` |
| `θ₁₃`      | Reactor mixing angle                 | ``\\arcsin(\\sqrt{0.021})`` |
| `θ₂₃`      | Atmospheric mixing angle             | ``\\arcsin(\\sqrt{0.57})``  |
| `δCP`       | Dirac CP-violation phase             | 1.0 rad           |
| `Δm²₂₁`    | Solar mass-squared splitting [eV²]   | ``7.53 \\times 10^{-5}`` |
| `Δm²₃₁`    | Atmospheric mass-squared splitting [eV²] | depends on ordering |

# Fields
- `ordering::Symbol = :NO`: neutrino mass ordering. `:NO` for normal ordering
  (``\\Delta m^2_{31} > 0``), `:IO` for inverted ordering (``\\Delta m^2_{31} < 0``).
"""
@kwdef struct ThreeFlavour <: FlavourModel
    ordering::Symbol = :NO
end

"""
    ThreeFlavourXYCP <: FlavourModel

Three-flavour model with a shell (Cartesian) parametrization of ``\\delta_{CP}``.

Replaces the scalar `δCP` parameter with a 2-vector `δCPshell` to avoid discontinuities
at the circular boundary ``[0, 2\\pi)`` during gradient-based fitting.

# Fields
- `three_flavour::ThreeFlavour = ThreeFlavour()`: underlying three-flavour configuration.

See also [`ThreeFlavour`](@ref).
"""
@kwdef struct ThreeFlavourXYCP <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
end

"""
    Sterile <: FlavourModel

3+1 sterile neutrino model extending the standard three flavours.

Adds a fourth mass eigenstate with mass-squared splitting `Δm²₄₁` and three new mixing
angles `θ₁₄`, `θ₂₄`, `θ₃₄`. The PMNS matrix is extended to 4×4 via

```math
U_{4\\times4} = R_{34}\\, R_{24}\\, R_{14} \\cdot
\\begin{pmatrix} U_{PMNS} & 0 \\\\ 0 & 1 \\end{pmatrix}
```

# Fields
- `three_flavour::ThreeFlavour = ThreeFlavour()`: underlying three-flavour configuration.

See also [`ThreeFlavour`](@ref).
"""
@kwdef struct Sterile <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
end

"""
    ADD <: FlavourModel

Arkani-Hamed–Dimopoulos–Dvali (ADD) large extra dimensions model.

Introduces a tower of ``N_{KK}`` Kaluza-Klein excitations for each active neutrino,
yielding a ``3(N_{KK}+1) \\times 3(N_{KK}+1)`` mass matrix. The Dirac mass matrix
is constructed from the PMNS matrix and absolute neutrino masses, with off-diagonal
couplings controlled by the extra-dimension compactification radius.

Additional parameters beyond [`ThreeFlavour`](@ref):

| Parameter    | Description                                  | Default |
|:------------ |:-------------------------------------------- |:------- |
| `m₀`         | Lightest neutrino mass [eV]                 | 0.01    |
| `ADD_radius`  | Extra-dimension compactification radius [μm] | 0.01    |

# Fields
- `three_flavour::ThreeFlavour = ThreeFlavour()`: underlying three-flavour configuration.
- `N_KK::Int = 5`: number of Kaluza-Klein modes to include.

See also [`Cut`](@ref) for excluding heavy KK states from coherent oscillation.
"""
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

@kwdef struct NND <: FlavourModel      
    three_flavour::ThreeFlavour = ThreeFlavour()
end

@kwdef struct NNM <: FlavourModel
    three_flavour::ThreeFlavour = ThreeFlavour()
end

"""
    OscillationConfig{F, I, P, S, E}

Master configuration for the neutrino oscillation probability engine.

Each type parameter selects a different physical model via multiple dispatch.
Construct with keyword arguments and pass to [`configure`](@ref) to obtain a
ready-to-use [`Osc`](@ref) physics module.

# Fields
- `flavour::F = ThreeFlavour()`: neutrino mass/flavour model ([`FlavourModel`](@ref)).
- `interaction::I = Vacuum()`: matter interaction model ([`InteractionModel`](@ref)).
- `propagation::P = Basic()`: propagation mechanism ([`PropagationModel`](@ref)).
- `states::S = All()`: eigenstate selection strategy ([`StateSelector`](@ref)).
- `eigen_method::E = DefaultEigen()`: eigendecomposition algorithm ([`EigenMethod`](@ref)).

# Examples
```julia
# Standard 3-flavour vacuum oscillations (all defaults)
cfg = OscillationConfig()

# Inverted ordering with MSW matter effects
cfg = OscillationConfig(flavour=ThreeFlavour(ordering=:IO), interaction=SI())

# ADD model with decoherence and Barger eigensolver
cfg = OscillationConfig(
    flavour=ADD(N_KK=3),
    interaction=SI(),
    propagation=Decoherent(σₑ=0.05),
    eigen_method=BargerEigen()
)
```
"""
@kwdef struct OscillationConfig{F<:FlavourModel, I<:InteractionModel, P<:PropagationModel, S<:StateSelector, E<:EigenMethod}
    flavour::F = ThreeFlavour()
    interaction::I = Vacuum()
    propagation::P = Basic()
    states::S = All()
    eigen_method::E = DefaultEigen()
end

"""
    Osc <: Newtrinos.Physics

Configured oscillation physics module, returned by [`configure`](@ref).

This struct satisfies the `Newtrinos.Physics` interface and can be passed into any
experiment's `configure(physics=...)` method.

# Fields
- `cfg::OscillationConfig`: the full configuration used to build this module.
- `params::NamedTuple`: default oscillation parameter values (e.g. `θ₁₂`, `Δm²₃₁`, …).
- `priors::NamedTuple`: prior distributions for each parameter.
- `matrices::Function`: closure `matrices(params) -> (U, h)` returning mixing matrix and
  mass-squared eigenvalues.
- `osc_prob::Function`: closure computing oscillation probabilities. See
  [`get_osc_prob`](@ref) for the two call signatures.
"""
@kwdef struct Osc <: Newtrinos.Physics
    cfg::OscillationConfig
    params::NamedTuple
    priors::NamedTuple
    matrices::Function
    osc_prob::Function
end

"""
    configure(cfg::OscillationConfig=OscillationConfig()) -> Osc

Create a fully configured oscillation physics module from the given configuration.

Assembles default parameters, priors, mixing-matrix closure, and oscillation-probability
closure into an [`Osc`](@ref) struct ready for use in experiment forward models.

# Arguments
- `cfg::OscillationConfig`: oscillation configuration (defaults to standard 3-flavour vacuum).

# Returns
An [`Osc`](@ref) instance.

# Examples
```julia
using Newtrinos

# Default 3-flavour vacuum
physics = Newtrinos.osc.configure()

# With matter effects and inverted ordering
physics = Newtrinos.osc.configure(
    OscillationConfig(flavour=ThreeFlavour(ordering=:IO), interaction=SI())
)
```
"""
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

"""
    get_params(cfg::OscillationConfig) -> NamedTuple
    get_params(cfg::FlavourModel) -> NamedTuple

Return the default oscillation parameter values for the given configuration or flavour model.

Delegates to the flavour-model-specific method. Each [`FlavourModel`](@ref) subtype defines
its own set of parameters (mixing angles, mass splittings, and any BSM quantities).

# Arguments
- `cfg`: an [`OscillationConfig`](@ref) or a [`FlavourModel`](@ref) subtype instance.

# Returns
A `NamedTuple` of `Symbol => Float64` (or `Vector` for shell parameters) with the nominal
parameter values.
"""
get_params(cfg::OscillationConfig) = get_params(cfg.flavour)

"""
    get_priors(cfg::OscillationConfig) -> NamedTuple
    get_priors(cfg::FlavourModel) -> NamedTuple

Return prior distributions for each oscillation parameter.

Delegates to the flavour-model-specific method. Priors are `Distributions.jl` objects
(typically `Uniform` or `LogUniform`) used for sampling and inference.

# Arguments
- `cfg`: an [`OscillationConfig`](@ref) or a [`FlavourModel`](@ref) subtype instance.

# Returns
A `NamedTuple` of `Symbol => Distribution` mapping each parameter to its prior.
"""
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



function get_params(cfg::NND)  #N-Naturalness Dirac
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
    params[:m₀] = ftype(0.01)
    params[:N] = ftype(50)
    params[:r] = ftype(1e-8)
    

    NamedTuple(params)
end

function get_priors(cfg::NND)    #N-Naturalness Dirac
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:m₀] = LogUniform(ftype(1e-6),ftype(1e-1))
    priors[:N] = DiscreteUniform(ftype(2),ftype(5000))
    priors[:r] = LogUniform(ftype(1e-6),ftype(1))
  
    NamedTuple(priors)
end

   
function get_params(cfg::NNM)  #N-Naturalness Majorana
    std = get_params(cfg.three_flavour)
    params = OrderedDict(pairs(std))
   
    params[:m₀] = ftype(0.01)
    params[:N] = ftype(50)
    params[:r] = ftype(1e-8)
  
    
    NamedTuple(params)
end

function get_priors(cfg::NNM)    #N-Naturalness Majorana
    std = get_priors(cfg.three_flavour)
    priors = OrderedDict(pairs(std))
    priors = OrderedDict{Symbol, Distribution}(pairs(std))
    priors[:m₀] = LogUniform(ftype(1e-6),ftype(1e-1))
    priors[:N] = DiscreteUniform(ftype(2),ftype(5000))
    priors[:r] = LogUniform(ftype(1e-6),ftype(1))
   

    NamedTuple(priors)
end

"""
    get_PMNS(params) -> SMatrix{3,3}

Construct the ``3 \\times 3`` PMNS leptonic mixing matrix from oscillation parameters.

The matrix is built as

```math
U_{PMNS} = R_{23}(\\theta_{23}) \\;\\cdot\\; R_{13}(\\theta_{13},\\, \\delta_{CP}) \\;\\cdot\\; R_{12}(\\theta_{12})
```

where each rotation matrix uses `zero(T)` / `one(T)` to preserve `ForwardDiff.Dual`
number types for automatic differentiation.

# Arguments
- `params::NamedTuple`: must contain `θ₂₃`, `θ₁₃`, `θ₁₂`, and `δCP`.

# Returns
`SMatrix{3,3,Complex{T}}` — the PMNS mixing matrix, where `T` is inferred from the
parameter types.
"""
function get_PMNS(params)
    T = typeof(params.θ₂₃)
    U1 = SMatrix{3,3}(one(T), zero(T), zero(T), zero(T), cos(params.θ₂₃), -sin(params.θ₂₃), zero(T), sin(params.θ₂₃), cos(params.θ₂₃))
    T = typeof(params.θ₁₃)
    U2 = SMatrix{3,3}(cos(params.θ₁₃), zero(T), -sin(params.θ₁₃)*cis(params.δCP), zero(T), one(T), zero(T), sin(params.θ₁₃)*cis(-params.δCP), zero(T), cos(params.θ₁₃))
    T = typeof(params.θ₁₂)
    U3 = SMatrix{3,3}(cos(params.θ₁₂), -sin(params.θ₁₂), zero(T), sin(params.θ₁₂), cos(params.θ₁₂), zero(T), zero(T), zero(T), one(T))
    U = U1 * U2 * U3
end

"""
    get_abs_masses(params) -> (m1, m2, m3)

Compute the absolute neutrino masses from the lightest mass and mass-squared splittings.

For normal ordering (``\\Delta m^2_{31} > 0``): ``m_1 = m_0``,
``m_2 = \\sqrt{\\Delta m^2_{21} + m_0^2}``, ``m_3 = \\sqrt{\\Delta m^2_{31} + m_0^2}``.

For inverted ordering (``\\Delta m^2_{31} < 0``): ``m_3 = m_0``,
``m_1 = \\sqrt{-\\Delta m^2_{31} + m_0^2}``,
``m_2 = \\sqrt{\\Delta m^2_{21} - \\Delta m^2_{31} + m_0^2}``.

# Arguments
- `params::NamedTuple`: must contain `m₀`, `Δm²₂₁`, and `Δm²₃₁`.

# Returns
A tuple `(m1, m2, m3)` of absolute neutrino masses [eV].
"""
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


"""
    osc_kernel(U, H, e, l) -> AbstractMatrix

Compute the coherent oscillation amplitude matrix for a single energy and baseline.

Evaluates ``A = U \\, \\mathrm{diag}\\!\\bigl(e^{-i\\, \\phi_j}\\bigr) \\, U^\\dagger``
where ``\\phi_j = \\frac{\\Delta m^2_j \\, L}{4E}`` (in natural units via `F_units`).

# Arguments
- `U::AbstractMatrix{<:Number}`: mixing matrix (unitary, ``n \\times n``).
- `H::AbstractVector{<:Number}`: mass-squared eigenvalues [eV²].
- `e::Real`: neutrino energy [GeV].
- `l::Real`: baseline length [km].

# Returns
An ``n \\times n`` complex amplitude matrix ``A_{\\beta\\alpha}``.
"""
function osc_kernel(U::AbstractMatrix{<:Number}, H::AbstractVector{<:Number}, e::Real, l::Real)
    phase_factors = -F_units * 1im * (l / e) .* H
    U * Diagonal(exp.(phase_factors)) * U'
end

"""
    osc_kernel(U, H, e, l, σₑ) -> (A, decay)

Compute the oscillation amplitude matrix with low-pass damping.

Same as the coherent [`osc_kernel`](@ref) but each phase factor is additionally
multiplied by a decay ``d_j = \\exp(-2\\,|\\phi_j|\\,\\sigma_e^2)``.

# Arguments
- `U::AbstractMatrix{<:Number}`: mixing matrix.
- `H::AbstractVector{<:Number}`: mass-squared eigenvalues [eV²].
- `e::Real`: neutrino energy [GeV].
- `l::Real`: baseline length [km].
- `σₑ::Real`: damping strength parameter.

# Returns
A tuple `(A, decay)` where `A` is the damped amplitude matrix and `decay` is the
per-eigenstate decay vector.
"""
function osc_kernel(U::AbstractMatrix{<:Number}, H::AbstractVector{<:Number}, e::Real, l::Real, σₑ::Real)
    phase_factors = -F_units * (l / e) .* H
    decay = exp.(-2 * abs.(phase_factors) * σₑ^2) #exp.(-abs.(σₑ / e * phase_factors)/2)
    U * Diagonal(exp.(1im * phase_factors) .* decay) * U', decay
end

"""
    compute_matter_matrices(H_eff, e, layer, anti, interaction::SI, eigen_method=DefaultEigen()) -> (U, h)

Add the MSW matter potential to the effective Hamiltonian and diagonalize.

Modifies `H_eff` by adding the Wolfenstein charged-current and neutral-current potentials
for the given [`Layer`](@ref) density, then eigendecomposes the result via
[`decompose`](@ref). Two methods are provided: one for generic `AbstractMatrix` and an
optimized one for `SMatrix{3,3}`.

# Arguments
- `H_eff`: effective Hamiltonian in the flavour basis (``U\\,\\mathrm{diag}(h)\\,U^\\dagger``).
- `e::Real`: neutrino energy [GeV].
- `layer::Layer`: matter layer with proton and neutron densities.
- `anti::Bool`: `true` for antineutrinos (flips potential sign).
- `interaction::SI`: matter interaction selector.
- `eigen_method::EigenMethod`: eigendecomposition algorithm.

# Returns
A tuple `(U_m, h_m)` of matter-modified eigenvectors and eigenvalues.
"""
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

"""
    osc_reduce(matter_matrices, path, e, propagation) -> Matrix

Combine per-layer oscillation amplitudes along a multi-layer path into a single
transition probability matrix for one energy.

Dispatches on [`PropagationModel`](@ref):
- [`Basic`](@ref): multiplies amplitude matrices across layers, then takes ``|\\cdot|^2``.
- [`Damping`](@ref): multiplies damped amplitudes and adds an incoherent recovery term
  using a path-length-weighted average of ``|U|^2``.

# Arguments
- `matter_matrices`: vector of `(U_m, h_m)` tuples, one per layer (from
  [`compute_matter_matrices`](@ref)).
- `path::Vector{Path}`: sequence of path segments for this baseline.
- `e::Real`: neutrino energy [GeV].
- `propagation::PropagationModel`: propagation model instance.

# Returns
An ``n \\times n`` real matrix of transition probabilities ``P_{\\beta\\alpha}``.
"""
function osc_reduce(matter_matrices, path, e, propagation::Damping)
    res = map(section -> osc_kernel(matter_matrices[section.layer_idx]..., e, section.length, propagation.σₑ), path)
    decay = abs2.(reduce(.*, last.(res)))
    # taking an average mixing matrix along the path to compute the decoherent sum, which is a bold approximation
    w = weights([section.length for section in path])
    P_ave  = mean([abs2.(matter_matrices[section.layer_idx][1]) for section in path], w)
    p = abs2.(reduce(*, first.(res))) .+ P_ave * Diagonal(1 .- decay) * P_ave'
end

function osc_reduce(matter_matrices, path, e, propagation::Basic)
    p = abs2.(mapreduce(section -> osc_kernel(matter_matrices[section.layer_idx]..., e, section.length), *, path))
end
    

"""
    matter_osc_per_e(H_eff, e, layers, paths, anti, propagation, interaction, eigen_method=DefaultEigen()) -> Array

Compute matter-affected oscillation probabilities at a single energy across all paths.

For each [`Layer`](@ref), computes the matter-modified eigensystem via
[`compute_matter_matrices`](@ref), then reduces along each path via
[`osc_reduce`](@ref) (for [`Basic`](@ref)/[`Damping`](@ref)) or via explicit
density-matrix evolution (for [`Decoherent`](@ref)).

# Arguments
- `H_eff`: effective Hamiltonian in the flavour basis.
- `e::Real`: neutrino energy [GeV].
- `layers::StructVector{Layer}`: Earth density layers.
- `paths`: per-baseline layer traversals (`VectorOfVectors{Path}`).
- `anti::Bool`: `true` for antineutrinos.
- `propagation::PropagationModel`: propagation model.
- `interaction::InteractionModel`: matter interaction model.
- `eigen_method::EigenMethod`: eigendecomposition algorithm.

# Returns
An `Array` of shape `(n_flav, n_flav, n_paths)` with ``P_{\\beta\\alpha}`` for each path.
"""
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

"""
    select(U, h, cfg::StateSelector) -> (U_sel, h_sel, rest)

Partition mass eigenstates into coherent and incoherent subsets based on the
[`StateSelector`](@ref) strategy.

- [`All`](@ref): returns all states unchanged with `rest = 0.0`.
- [`Cut`](@ref): states with ``\\sqrt{|m^2_j|}`` below the cutoff are kept for coherent
  oscillation; states above it are averaged incoherently and their contribution is returned
  as the `rest` matrix.

# Arguments
- `U`: mixing matrix (``n_{flav} \\times n_{states}``).
- `h`: mass-squared eigenvalues vector.
- `cfg::StateSelector`: selection strategy.

# Returns
A tuple `(U_sel, h_sel, rest)` where:
- `U_sel`: mixing matrix columns for coherent states.
- `h_sel`: eigenvalues for coherent states.
- `rest`: incoherent probability correction matrix (scalar `0.0` or ``n \\times n`` matrix).
"""
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


"""
    propagate(U, h, E, L, propagation) -> Array{T,4}
    propagate(U, h, E, paths, layers, propagation, interaction, anti, ...) -> Array{T,4}

Compute oscillation probabilities over grids of energy and baseline/path.

**Vacuum methods** (dispatched by baseline vector `L`):
Loops over all `(E, L)` pairs, evaluating [`osc_kernel`](@ref) and squaring amplitudes.
Dispatches on [`PropagationModel`](@ref) for [`Basic`](@ref), [`Damping`](@ref), and
[`Decoherent`](@ref) propagation.

**Matter methods** (dispatched by `paths::VectorOfVectors{Path}`):
- [`Vacuum`](@ref) interaction: collapses paths to total lengths and delegates to the
  vacuum method.
- [`SI`](@ref)/[`NSI`](@ref): reconstructs the flavour-basis Hamiltonian and calls
  [`matter_osc_per_e`](@ref) for each energy.

# Arguments
- `U`: mixing matrix.
- `h`: mass-squared eigenvalues [eV²].
- `E`: neutrino energies [GeV].
- `L`: baselines [km] (vacuum) or `paths`/`layers` (matter).
- `propagation::PropagationModel`: propagation model.
- `interaction::InteractionModel`: matter interaction model (matter methods only).
- `anti::Bool`: `true` for antineutrinos (matter methods only).

# Returns
`Array{T,4}` of shape `(n_flav, n_flav, n_E, n_L)` with ``P_{\\beta\\alpha}``
(note: the caller [`get_osc_prob`](@ref) transposes this to `(n_E, n_L, n_flav, n_flav)`).
"""
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
    if anti
        H_eff = conj.(U) * Diagonal(h) * transpose(U)
    else
        H_eff = U * Diagonal(h) * adjoint(U)
    end
    p = stack(map(e -> matter_osc_per_e(H_eff, e, layers, paths, anti, propagation, interaction, eigen_method), E))
    permutedims(p, (1, 2, 4, 3))
end

# Fuse rest addition + permutedims(p, (3,4,1,2)) into one pass
function _add_rest_and_permute(p_raw, rest)
    n1, n2, n3, n4 = size(p_raw)
    result = similar(p_raw, n3, n4, n1, n2)
    @inbounds for b in 1:n2, a in 1:n1, j in 1:n4, i in 1:n3
        result[i, j, a, b] = p_raw[a, b, i, j] + (rest isa AbstractArray ? rest[a, b] : rest)
    end
    result
end

"""
    get_osc_prob(cfg::OscillationConfig) -> Function

Construct the oscillation probability closure for the given configuration.

The returned function `osc_prob` has two methods:

**Vacuum-style** (baselines as distances):
```julia
osc_prob(E::AbstractVector, L::AbstractVector, params::NamedTuple; anti=false)
```

**Matter-style** (baselines as Earth layer paths):
```julia
osc_prob(E::AbstractVector, paths::VectorOfVectors{Path}, layers::StructVector{Layer},
         params::NamedTuple; anti=false)
```

# Arguments
- `E`: neutrino energies [GeV].
- `L`: baselines [km] (vacuum method).
- `paths`: per-baseline layer traversals (matter method), from [`Newtrinos.earth_layers.EarthLayers`](@ref).
- `layers`: Earth density layers (matter method).
- `params::NamedTuple`: oscillation parameters (mixing angles, mass splittings, etc.).
- `anti::Bool = false`: set `true` for antineutrinos (conjugates the PMNS matrix).

# Returns
`Array{T,4}` of shape `(n_E, n_L, n_flav, n_flav)` where entry
`result[i, j, β, α]` gives the transition probability
``P(\\nu_\\alpha \\to \\nu_\\beta)`` at energy `E[i]` and baseline/path index `j`.
"""
function get_osc_prob(cfg::OscillationConfig)

    function osc_prob(E::AbstractVector{<:Real}, L::AbstractVector{<:Real}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour, cfg.eigen_method)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U

        U, h, rest = select(Uc, h, cfg.states)

        # propagate returns (n_flav, n_flav, n_E, n_L)
        p_raw = propagate(U, h, E, L, cfg.propagation)

        # fuse rest addition + permutedims into (n_E, n_L, n_flav, n_flav)
        return _add_rest_and_permute(p_raw, rest)
    end

    function osc_prob(E::AbstractVector{<:Real}, paths::VectorOfVectors{Path}, layers::StructVector{Layer}, params::NamedTuple; anti=false)
        U, h_raw = get_matrices(cfg.flavour, cfg.eigen_method)(params)
        h = h_raw .- minimum(h_raw)
        Uc = anti ? conj.(U) : U

        U, h, rest = select(Uc, h, cfg.states)

        # propagate returns (n_flav, n_flav, n_E, n_cz)
        p_raw = propagate(U, h, E, paths, layers, cfg.propagation, cfg.interaction, anti, cfg.eigen_method)

        # fuse rest addition + permutedims into (n_E, n_cz, n_flav, n_flav)
        return _add_rest_and_permute(p_raw, rest)
    end

    return osc_prob
end


"""
    get_matrices(cfg::FlavourModel, eigen_method::EigenMethod=DefaultEigen()) -> Function

Construct a closure that computes the mixing matrix and mass-squared eigenvalues from
oscillation parameters.

The returned function has signature `matrices(params::NamedTuple) -> (U, h)` where:
- `U` is the mixing matrix (unitary).
- `h` is a vector of mass-squared eigenvalues [eV²].

For [`ThreeFlavour`](@ref), returns `SMatrix{3,3}` and `SVector{3}` (zero-allocation).
For [`Sterile`](@ref), returns 4×4 dense matrices.
For [`ADD`](@ref) and dark-dimension models, returns ``3(N_{KK}+1) \\times 3(N_{KK}+1)``
dense matrices obtained by diagonalizing the full KK mass matrix via
[`decompose`](@ref).

The closure is ForwardDiff-compatible: all intermediate computations preserve dual-number
types through `zero(T)` / `one(T)` patterns and type promotion.

# Arguments
- `cfg::FlavourModel`: flavour model instance (dispatches to the appropriate method).
- `eigen_method::EigenMethod`: eigendecomposition algorithm (only used by models that
  diagonalize a mass matrix, e.g. [`ADD`](@ref)).

See also [`get_PMNS`](@ref), [`configure`](@ref).
"""
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



function get_matrices(cfg::NND)

   function get_Nnaturalness(params::NamedTuple)
        
        N_int = round(Int, ForwardDiff.value(params[:N])) 
        N_dual = params[:N]   
        r=params[:r]  
        
        η=1+ 1/N_dual #params[:η]
        
        T = promote_type(
            typeof(params[:N]), 
            typeof(params[:m₀]),
            typeof(params[:r]), 
            typeof(params[:Δm²₂₁]), 
            typeof(params[:Δm²₃₁]),
            typeof(params[:δCP]),
            typeof(params[:θ₁₂]),
            typeof(params[:θ₁₃]),
            typeof(params[:θ₂₃])
        ) 

        m1, m2, m3 = get_abs_masses(params)

        
        m1_T = T(m1)
        m2_T = T(m2) 
        m3_T = T(m3)
        
        factor=(η-1) * 2^(1/(N_dual-1)) 
        factorN=factor #0.5*(η-1)*(1+(N_dual/(η-1)))

        if r>=0.0
            scale_1= (m1_T ^2)/(r*factor)
            scale_2= (m2_T ^2)/(r*factor)
            scale_3= (m3_T ^2)/(r*factor)
        end

      

        matrix_e=zeros(T, N_int,N_int)
        matrix_m=zeros(T, N_int,N_int)
        matrix_t=zeros(T, N_int,N_int)

      
    
        for i in 1:N_int
            
            sqrt_i = sqrt(T(2*(i-1)) + T(params[:r]))
           
            
            for j in 1:N_int
                
                sqrt_j =sqrt(T(2*(j-1)) + T(params[:r]))

                
                if i == j

                    matrix_e[i, j] =sqrt_i * sqrt_j *η
                    matrix_m[i, j] =sqrt_i * sqrt_j *η
                    matrix_t[i, j] =sqrt_i * sqrt_j *η
                else
                    matrix_e[i, j] =sqrt_i * sqrt_j 
                    matrix_m[i, j] =sqrt_i * sqrt_j 
                    matrix_t[i, j] =sqrt_i * sqrt_j 
                end

            

            end


        end    
    
        # PMNS matrix 

        U = get_PMNS(params)

        eigenvalues_e, V_e = eigen(Hermitian(matrix_e))
        eigenvalues_m, V_m = eigen(Hermitian(matrix_m))
        eigenvalues_t, V_t = eigen(Hermitian(matrix_t))
        
     
        eigenvalues= Vector{T}(undef, 3*N_int)
        
        for i in 1:N_int
            eigenvalues[3*i-2] =(eigenvalues_e[i])*scale_1
            eigenvalues[3*i-1] =(eigenvalues_m[i])*scale_2
            eigenvalues[3*i] = (eigenvalues_t[i])*scale_3
        end

        
        eigenvalues[end-2] = eigenvalues[end-2]*(factor)/(factorN)
        eigenvalues[end-1] =eigenvalues[end-1]*(factor)/(factorN)
        eigenvalues[end] =  eigenvalues[end]*(factor)/(factorN)

        Vmatrix = zeros(T, 3*N_int, 3*N_int)

        col = 1
        for i in 1:N_int 

            Vmatrix[1:3:3*N_int, col] = V_e[:, i]
            col += 1
            
          
            Vmatrix[2:3:3*N_int, col] = V_m[:, i]
            col += 1
            
           
            Vmatrix[3:3:3*N_int, col] = V_t[:, i]
            col += 1
        end
      

        bigU = kron(Matrix{T}(I, N_int, N_int), U)

        FinalUmatrix = bigU * Vmatrix 

        delta_mass = Vector{T}(undef, 3*N_int)

        if r==0.0

            delta_mass[1] = zero(T)
            delta_mass[2] = T(params.Δm²₂₁)
            delta_mass[3] = T(params.Δm²₃₁)
            
           
        else

            delta_mass[1] =(eigenvalues_e[1])*scale_1-  (eigenvalues_e[1])*scale_1
            delta_mass[2] =(eigenvalues_m[1])*scale_2- (eigenvalues_e[1])*scale_1
            delta_mass[3] =(eigenvalues_t[1])*scale_3- (eigenvalues_e[1])*scale_1
        
            for i in 2:N_int
                delta_mass[3*i-2] =(eigenvalues_e[i])*scale_1-  (eigenvalues_e[1])*scale_1
                delta_mass[3*i-1] =(eigenvalues_m[i])*scale_2- (eigenvalues_e[1])*scale_1
                delta_mass[3*i] = (eigenvalues_t[i])*scale_3- (eigenvalues_e[1])*scale_1

                if i==N_int
                    delta_mass[3*i-2] =((eigenvalues_e[i])*scale_1*(factor)/(factorN))- ((eigenvalues_e[1])*scale_1)
                    delta_mass[3*i-1] =((eigenvalues_m[i])*scale_2*(factor)/(factorN))- ((eigenvalues_e[1])*scale_1)
                    delta_mass[3*i] = ((eigenvalues_t[i])*scale_3*(factor)/(factorN))- ((eigenvalues_e[1])*scale_1)
                end    
            end

            

        end
        
        h = delta_mass
        
         
        return FinalUmatrix, h , eigenvalues, V_e, V_m, V_t
    end

end


function get_matrices(cfg::NNM)
   function get_Nnaturalness(params::NamedTuple)

       
        N_int = round(Int, ForwardDiff.value(params[:N])) 
        N_dual = params[:N]   
    
        r=params[:r]   

        η=1 + 1/N_dual 
        
        T = promote_type(
            typeof(params[:N]), 
            typeof(params[:m₀]),
            typeof(params[:r]), 
            typeof(params[:Δm²₂₁]), 
            typeof(params[:Δm²₃₁]),
            typeof(params[:δCP]),
            typeof(params[:θ₁₂]),
            typeof(params[:θ₁₃]),
            typeof(params[:θ₂₃])
        ) 

        m1, m2, m3 = get_abs_masses(params)

        
        m1_T = T(m1)
        m2_T = T(m2) 
        m3_T = T(m3)
        
        
        factor=(η-1) * 2^(1/(N_dual-1)) #first method
        factorN= factor#(1+(η-1)/N_dual)*(η-1)

        if r>=0.0
            scale_1= (m1_T)/(r*factor)
            scale_2= (m2_T)/(r*factor)
            scale_3= (m3_T)/(r*factor)
        end

       
        matrix_e=zeros(T, N_int,N_int)
        matrix_m=zeros(T, N_int,N_int)
        matrix_t=zeros(T, N_int,N_int)

      
    
        for i in 1:N_int
            
            sqrt_i = sqrt(T(2*(i-1)) + T(params[:r]))
           
            
            for j in 1:N_int
                
                sqrt_j =sqrt(T(2*(j-1)) + T(params[:r]))

                
                if i == j

                    matrix_e[i, j] =sqrt_i * sqrt_j *η
                    matrix_m[i, j] =sqrt_i * sqrt_j *η
                    matrix_t[i, j] =sqrt_i * sqrt_j *η
                else
                    matrix_e[i, j] =sqrt_i * sqrt_j 
                    matrix_m[i, j] =sqrt_i * sqrt_j 
                    matrix_t[i, j] =sqrt_i * sqrt_j 
                end


            end


        end

        
    
        # PMNS matrix 

        U = get_PMNS(params)

       eigenvalues_e, V_e = eigen(Hermitian(matrix_e))
       eigenvalues_m, V_m = eigen(Hermitian(matrix_m))
       eigenvalues_t, V_t = eigen(Hermitian(matrix_t))
      
     
       eigenvalues= Vector{T}(undef, 3*N_int)
     
        for i in 1:N_int
            eigenvalues[3*i-2] =((eigenvalues_e[i])*scale_1)^2
            eigenvalues[3*i-1] =((eigenvalues_m[i])*scale_2)^2
            eigenvalues[3*i] = ((eigenvalues_t[i])*scale_3)^2
        end

        eigenvalues[end-2] = eigenvalues[end-2]*(factor^2)/(factorN^2)
        eigenvalues[end-1] = eigenvalues[end-1]*(factor^2)/(factorN^2)
        eigenvalues[end] =  eigenvalues[end]*(factor^2)/(factorN^2)
       
        Vmatrix = zeros(T, 3*N_int, 3*N_int)

        col = 1
        for i in 1:N_int 
            Vmatrix[1:3:3*N_int, col] = V_e[:, i]
            col += 1
            
            Vmatrix[2:3:3*N_int, col] = V_m[:, i]
            col += 1
            
           
            Vmatrix[3:3:3*N_int, col] = V_t[:, i]
            col += 1
        end
        
    

        bigU = kron(Matrix{T}(I, N_int, N_int), U)

        FinalUmatrix = bigU * Vmatrix 

      

        delta_mass = Vector{T}(undef, 3*N_int)
       
        if r==0.0

            delta_mass[1] = zero(T)
            delta_mass[2] = T(params.Δm²₂₁)
            delta_mass[3] = T(params.Δm²₃₁)
            
        
        else

            delta_mass[1] =((eigenvalues_e[1])*scale_1)^2-((eigenvalues_e[1])*scale_1)^2
            delta_mass[2] =((eigenvalues_m[1])*scale_2)^2- ((eigenvalues_e[1])*scale_1)^2
            delta_mass[3] =((eigenvalues_t[1])*scale_3)^2- ((eigenvalues_e[1])*scale_1)^2 
        
            for i in 2:N_int
                delta_mass[3*i-2] =((eigenvalues_e[i])*scale_1)^2- ((eigenvalues_e[1])*scale_1)^2 
                delta_mass[3*i-1] =((eigenvalues_m[i])*scale_2)^2- ((eigenvalues_e[1])*scale_1)^2 
                delta_mass[3*i] = ((eigenvalues_t[i])*scale_3)^2- ((eigenvalues_e[1])*scale_1)^2 

                if i==N_int
                    delta_mass[3*i-2] =((eigenvalues_e[i])*scale_1*(factor)/(factorN))^2- ((eigenvalues_e[1])*scale_1)^2 
                    delta_mass[3*i-1] =((eigenvalues_m[i])*scale_2*(factor)/(factorN))^2- ((eigenvalues_e[1])*scale_1)^2 
                    delta_mass[3*i] = ((eigenvalues_t[i])*scale_3*(factor)/(factorN))^2- ((eigenvalues_e[1])*scale_1)^2 
                end    
            end

        end
       
        h = delta_mass
        
        return FinalUmatrix, h , eigenvalues, V_e, V_m, V_t
    end

end


end
