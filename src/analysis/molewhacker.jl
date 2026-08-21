using Distributions
using DataStructures
using DensityInterface
using DataFrames
using Accessors
using Optimization, ADTypes
using MeasureBase
using LinearAlgebra
using PositiveFactorizations
using PDMats
using InverseFunctions
using Logging
using StatsBase
using ArraysOfArrays
using BAT
using MGVI
using ForwardDiff
using ProgressMeter

"""
    importance_sampling(pstr, approx_dist, nsamples) -> DensitySampleVector

Draw importance-weighted samples from `pstr` using `approx_dist` as the proposal.

Samples `nsamples` points from `approx_dist`, evaluates ``\\log p`` (pstr) and ``\\log q`` (approx_dist)
at each point, and computes normalized importance weights. The evaluation of ``\\log p`` is
parallelized over threads. The weights are calculated as:

```math
\\log \\tilde{w}_i = \\log p - \\log q
w_i = \\exp\\!\\bigl( \\log \\tilde{w}_i - \\max(\\log \\tilde{w}_i) \\bigr)
```

with the ``\\max(\\log \\tilde{w}_i)`` substracted in the exponent such that 
``\\max_i w_i = 1`` for numerical stability. 

# Arguments
- `pstr`: BAT posterior (typically already transformed via `PriorToNormal`).
- `approx_dist`: proposal distribution supporting `bat_sample` and `logdensityof`.
- `nsamples::Int`: number of samples to draw from `approx_dist`.

# Returns
A `DensitySampleVector` with samples from `approx_dist`, their respective ``\\log p``, and their importance weights.
"""
function importance_sampling(pstr, approx_dist, nsamples)
    smpls_q, _ = bat_sample(approx_dist, IIDSampling(nsamples = nsamples))
    x_q = smpls_q.v
    logd_q = smpls_q.logd;
    logd_p = similar(logd_q)
    @showprogress Threads.@threads for i in eachindex(x_q)
        logd_p[i] = logdensityof(pstr, x_q[i])
    end
    logw_raw = logd_p .- logd_q;
    w = exp.(logw_raw .- maximum(logw_raw));
    smpls_p = DensitySampleVector(x_q, logd_p, weight=w)
end

"""
    local_MGVI_approx(pstr, θ_sel) -> MvNormal

Fit a local Gaussian approximation to `pstr` at the selected point `θ_sel`.

Uses the MGVI (Metric Gaussian Variational Inference) approach: the covariance is

```math
\\Sigma = \\bigl(J^\\top\\, F(\\theta)\\, J + I\\bigr)^{-1}
```

where ``F(\\theta)`` is the Fisher information matrix of the likelihood model
evaluated at ``\\theta``, and ``J = \\partial_\\theta(\\texttt{flat\\_params} \\circ m)``
is the Jacobian of the parameter reparameterization. The result is regularized with
a Cholesky decomposition via `PositiveFactorizations`.

# Arguments
- `pstr`: BAT posterior with a `likelihood` field exposing the forward model `k`.
- `θ_sel`: parameter vector (in the transformed space) at which to centre the Gaussian.

# Returns
A `MvNormal` distribution centred at `θ_sel` with covariance ``\\Sigma``.
"""
function local_MGVI_approx(pstr, θ_sel)
    m_tr=pstr.likelihood.k
    FI_inner = MGVI.fisher_information(m_tr(θ_sel))
    J = ForwardDiff.jacobian(MGVI.flat_params ∘ m_tr, θ_sel);
    Σ_raw = inv(Matrix(J' * FI_inner * J + I))
    Σ = PDMat(cholesky(Positive, Σ_raw))
    approx_dist = MvNormal(θ_sel, Σ)
    return approx_dist
end

"""
    make_prior_samples(posterior, nsamples::Int=10_000) -> NamedTuple

Initialize importance samples by using a standard normal as the proposal distribution.

Transforms `posterior` to the standard-normal space via BAT's `PriorToNormal`
reparameterization, then generates `nsamples` samples by calling [`importance_sampling`](@ref) 
with the transformed posterior and the standard normal ``\\mathcal{N}(0, I)`` as the proposal distribution. 

# Arguments
- `posterior`: a BAT posterior measure.
- `nsamples::Int`: number of importance samples to draw (default: `10_000`).

# Returns
A `NamedTuple` with fields:
- `approx_dist`: the standard normal proposal used.
- `samples_p`: `DensitySampleVector` in the transformed (normal) space.
- `samples_user`: samples back-transformed to the original parameter space.
"""
function make_prior_samples(posterior, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    pr_dist = MvNormal(zeros(pstr.prior.dist._dim), ones(pstr.prior.dist._dim))

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, pr_dist, nsamples)

    (approx_dist=pr_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)
end

"""
    make_init_samples(posterior, nseeds::Int=10, nsamples::Int=10_000) -> NamedTuple

Initialize importance samples by fitting a mixture of local Gaussian approximations
at posterior modes.

Finds `nseeds` approximate modes via L-BFGS optimization (using Sobol-sequence
starting points), fits a [`local_MGVI_approx`](@ref) at each mode, and combines
them into a `MixtureModel`. The mixture model weights are calculated as 

```math
\\begin{aligned}
\\log \\tilde{w}_i &= \\log p (\\mu_i) - \\log q (\\mu_i) \\\\
\\tilde{w}_i &= \\exp\\bigl(\\log \\tilde{w}_i - \\max(\\log \\tilde{w})\\bigr) \\\\
w_i &= \\frac{\\tilde{w}_i}{\\sum_j \\tilde{w}_j}
\\end{aligned}
```

where ``\\mu_i`` is the mode of the ``i``-th component. Importance samples are
then drawn from the mixture via [`importance_sampling`](@ref).

# Arguments
- `posterior`: a BAT posterior measure.
- `nseeds::Int`: number of mode-finding restarts (default: `10`).
- `nsamples::Int`: number of importance samples to draw (default: `10_000`).

# Returns
A `NamedTuple` with fields:
- `approx_dist::MixtureModel`: the weighted Gaussian mixture proposal.
- `samples_p`: `DensitySampleVector` in the transformed space.
- `samples_user`: samples back-transformed to the original parameter space.
"""
function make_init_samples(posterior, nseeds::Int=10, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = bat_sample(pstr.prior, SobolSampler(nsamples=nseeds)).result.v
    components = Array{MvNormal}(undef, nseeds)

    @info "Finding modes"

    Threads.@threads for i in 1:nseeds
        adsel = AutoForwardDiff()
        set_batcontext(ad = adsel)
        r = bat_findmode(pstr, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([seeds[i]])))
        components[i] = local_MGVI_approx(pstr, r.result)
    end

    approx_dist = MixtureModel(components)

    mode_logd_p_approx = [logdensityof(pstr, mode(ad)) for ad in approx_dist.components]
    mode_logd_q_approx = [logdensityof(approx_dist, mode(ad)) for ad in approx_dist.components]

    raw_mixture_logw = mode_logd_p_approx .- mode_logd_q_approx
    raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
    mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

    approx_dist = MixtureModel(approx_dist.components, mixture_w)

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, approx_dist, nsamples)

    (approx_dist=approx_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)
end

"""
    make_init_samples(posterior, seed_points::DataFrame, nsamples::Int=10_000) -> NamedTuple

Initialize importance samples from user-supplied seed points given as a DataFrame.

Like the integer-seed form but uses rows of `seed_points` as starting points for
mode-finding instead of Sobol samples. Each row is projected into the transformed
parameter space and L-BFGS is run from there.

!!! warning
    This method contains hardcoded field names (`Darkdim_radius`, `ca1`, `ca2`,
    `ca3`) specific to Darkdim models. It must be generalized before use with
    other parameter sets.

# Arguments
- `posterior`: a BAT posterior measure.
- `seed_points::DataFrame`: each row provides the starting parameter values.
  Must contain columns `Darkdim_radius`, `ca1`, `ca2`, `ca3`.
- `nsamples::Int`: number of importance samples to draw (default: `10_000`).

# Returns
A `NamedTuple` with fields:
- `approx_dist::MixtureModel`: the weighted Gaussian mixture proposal.
- `samples_p`: `DensitySampleVector` in the transformed space.
- `samples_user`: samples back-transformed to the original parameter space.
"""
function make_init_samples(posterior, seed_points::DataFrame, nsamples::Int=10_000)
    # NOTE: This method contains hardcoded field names (Darkdim_radius, ca1, ca2, ca3)
    # specific to Darkdim models. Consider generalizing if used with other models.
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = []
    for row in eachrow(seed_points)
        smpl = bat_sample(posterior, SobolSampler(nsamples=1)).result
        @reset smpl.v[1].Darkdim_radius = row.Darkdim_radius
        @reset smpl.v[1].ca1 = row.ca1
        @reset smpl.v[1].ca2 = row.ca2
        @reset smpl.v[1].ca3 = row.ca3
        push!(seeds, BAT.transform_samples(f_trafo, smpl)[1].v)
    end

    components = Array{MvNormal}(undef, length(seeds))

    @info "Finding modes"

    Threads.@threads for i in 1:length(seeds)
        adsel = AutoForwardDiff()
        set_batcontext(ad = adsel)
        r = bat_findmode(pstr, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([seeds[i]]), kwargs = (reltol=1e-4, maxiters=100)))
        components[i] = local_MGVI_approx(pstr, r.result)
    end

    approx_dist = MixtureModel(components)

    mode_logd_p_approx = [logdensityof(pstr, mode(ad)) for ad in approx_dist.components]
    mode_logd_q_approx = [logdensityof(approx_dist, mode(ad)) for ad in approx_dist.components]

    raw_mixture_logw = mode_logd_p_approx .- mode_logd_q_approx
    raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
    mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

    approx_dist = MixtureModel(approx_dist.components, mixture_w)

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, approx_dist, nsamples)

    (approx_dist=approx_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)
end

"""
    whack_a_mole(posterior, init_samples, n_whack=100) -> NamedTuple

Refine an importance sampling approximation by iteratively adding Gaussian
components at high-weight sample points.

At each of `n_whack` iterations:
1. The highest-weight sample in the current mixture is identified.
2. A [`local_MGVI_approx`](@ref) is fitted at that point.
3. The new component is added to the mixture with weight proportional to its
   posterior density at its mode relative to the mixture density.
4. New samples are drawn from the new component and importance-reweighted.

This "whack-a-mole" strategy adaptively targets undersampled regions of the
posterior. Use [`whack_many_moles`](@ref) for a parallelized variant with
convergence criteria.

# Arguments
- `posterior`: a BAT posterior measure.
- `init_samples`: initialization NamedTuple as returned by [`make_init_samples`](@ref)
  or [`make_prior_samples`](@ref).
- `n_whack::Int`: number of refinement iterations (default: `100`).

# Returns
A `NamedTuple` with fields:
- `approx_dist::MixtureModel`: the refined Gaussian mixture proposal.
- `samples_p`: accumulated `DensitySampleVector` in the transformed space.
- `samples_user`: samples back-transformed to the original parameter space.
"""
function whack_a_mole(posterior, init_samples, n_whack=100)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    smpls_p = init_samples.samples_p
    approx_dist = init_samples.approx_dist

    if init_samples.approx_dist isa MixtureModel
        approx_mix = approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist)) for approx_dist in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
    end

    samples_mix = smpls_p

    for n in 1:n_whack
        ess = bat_eff_sample_size(samples_mix, KishESS()).result
        @info "Effective sample size = $ess"
        eff = ess / length(samples_mix)
        @info "Efficiency = $eff"

        θ_iter_idx = findmax(samples_mix.weight)[2]
        θ_iter = samples_mix.v[θ_iter_idx]

        approx_dist = local_MGVI_approx(pstr, θ_iter)
        mode_logd_p_approx = logdensityof(pstr, mode(approx_dist))
        append!(mode_logd_p_mix, mode_logd_p_approx)

        approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]))

        mode_logd_q_mix = [logdensityof(approx_mix, mode(ad)) for ad in approx_mix.components]

        raw_mixture_logw = mode_logd_p_mix .- mode_logd_q_mix
        raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
        mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

        approx_mix = MixtureModel(approx_mix.components, mixture_w)

        new_nsamples = floor(Int, last(mixture_w) * length(samples_mix))
        @info "Generating $new_nsamples new samples"
        if new_nsamples > 0
            smpls_p = importance_sampling(pstr, approx_dist, new_nsamples)
            samples_mix = vcat(samples_mix, smpls_p)
        end

        logd_p = samples_mix.logd
        logd_q = logdensityof.(Ref(approx_mix), samples_mix.v)
        logw_raw = logd_p .- logd_q;
        w = exp.(logw_raw .- maximum(logw_raw));
        samples_mix.weight .= w;
    end

    (approx_dist=approx_mix, samples_p=samples_mix, samples_user=bat_transform(inverse(f_trafo), samples_mix).result)
end

"""
    whack_many_moles(posterior, init_samples;
                     target_efficiency=Inf, target_samplesize=Inf,
                     maxiter=100, n_parallel=Threads.nthreads(),
                     cache_dir=nothing) -> NamedTuple

Parallelized adaptive importance sampling with convergence criteria.

Like [`whack_a_mole`](@ref) but processes `n_parallel` high-weight sample
points simultaneously per each iteration using `Threads.@threads`. Iteration stops
when any of the following conditions is met:

- Effective sample size (ESS) exceeds `target_samplesize`.
- Efficiency (ESS / total samples) exceeds `target_efficiency`.
- Number of iterations reaches `maxiter`.

Intermediate results can be saved to disk after each iteration for fault tolerance.

# Arguments
- `posterior`: a BAT posterior measure.
- `init_samples`: initialization NamedTuple as returned by [`make_init_samples`](@ref)
  or [`make_prior_samples`](@ref).
- `target_efficiency::Real`: stop when ESS / n_samples exceeds this value
  (default: `Inf`, i.e. no efficiency target).
- `target_samplesize::Real`: stop when ESS exceeds this value
  (default: `Inf`, i.e. no ESS target).
- `maxiter::Int`: maximum number of refinement iterations (default: `100`).
- `n_parallel::Int`: number of new components to fit per iteration
  (default: `Threads.nthreads()`).
- `cache_dir::Union{String,Nothing}`: if provided, saves a JLD2 checkpoint after
  each iteration to this directory. (Default: `nothing`).

# Returns
A `NamedTuple` with fields:
- `approx_dist::MixtureModel`: the refined Gaussian mixture proposal.
- `samples_p`: accumulated `DensitySampleVector` in the transformed space.
- `samples_user`: samples back-transformed to the original parameter space.
"""
function whack_many_moles(posterior, init_samples; target_efficiency=Inf, target_samplesize=Inf, maxiter=100, n_parallel=Threads.nthreads(), cache_dir=nothing)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    smpls_p = init_samples.samples_p
    approx_dist = init_samples.approx_dist

    if approx_dist isa MixtureModel
        approx_mix = init_samples.approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(d)) for d in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
    end

    samples_mix = smpls_p
    iter = 0

    if !isnothing(cache_dir)
        if !isdir(cache_dir)
            mkdir(cache_dir)
        end
    end

    while true
        ess = bat_eff_sample_size(samples_mix, KishESS()).result
        @info "Effective sample size = $ess"
        eff = ess / length(samples_mix)
        @info "Efficiency = $eff"

        if (eff > target_efficiency) | (iter > maxiter) | (ess > target_samplesize)
            break
        end

        idxs = partialsortperm(samples_mix.weight, 1:n_parallel, rev=true)

        approx_dists = Array{MvNormal}(undef, n_parallel)
        mode_logd_p_approx = Array{Float64}(undef, n_parallel)

        Threads.@threads for i in 1:n_parallel
            θ_iter = samples_mix.v[idxs[i]]
            approx_dists[i] = local_MGVI_approx(pstr, θ_iter)
            mode_logd_p_approx[i] = logdensityof(pstr, mode(approx_dists[i]))
        end

        append!(mode_logd_p_mix, mode_logd_p_approx)
        approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, approx_dists))

        mode_logd_q_mix = [logdensityof(approx_mix, mode(ad)) for ad in approx_mix.components]

        raw_mixture_logw = mode_logd_p_mix .- mode_logd_q_mix
        raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
        mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

        approx_mix = MixtureModel(approx_mix.components, mixture_w)

        new_nsamples = [floor(Int, w * length(samples_mix)) for w in last(mixture_w, n_parallel)]

        for (i, n) in enumerate(new_nsamples)
            if n > 0
                smpls_p = importance_sampling(pstr, approx_dists[i], n)
                samples_mix = vcat(samples_mix, smpls_p)
            end
        end

        logd_p = samples_mix.logd
        logd_q = logdensityof.(Ref(approx_mix), samples_mix.v)
        logw_raw = logd_p .- logd_q;
        w = exp.(logw_raw .- maximum(logw_raw));
        samples_mix.weight .= w;

        iter += 1

        if !isnothing(cache_dir)
            FileIO.save(joinpath(cache_dir, "molewhacker_iter_$(iter).jld2"), Dict("approx_dist"=>approx_mix, "samples_p"=>samples_mix, "samples_user"=>bat_transform(inverse(f_trafo), samples_mix).result))
        end
    end

    (approx_dist=approx_mix, samples_p=samples_mix, samples_user=bat_transform(inverse(f_trafo), samples_mix).result)
end
