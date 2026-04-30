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

function local_MGVI_approx(pstr, θ_sel)
    m_tr=pstr.likelihood.k
    FI_inner = MGVI.fisher_information(m_tr(θ_sel))
    J = ForwardDiff.jacobian(MGVI.flat_params ∘ m_tr, θ_sel);
    Σ_raw = inv(Matrix(J' * FI_inner * J + I))
    Σ = PDMat(cholesky(Positive, Σ_raw))
    approx_dist = MvNormal(θ_sel, Σ)
    return approx_dist
end

function make_prior_samples(posterior, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    pr_dist = MvNormal(zeros(pstr.prior.dist._dim), ones(pstr.prior.dist._dim))

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, pr_dist, nsamples)

    (approx_dist=pr_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)
end

function make_init_samples(posterior, nseeds::Int=10, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = bat_sample(pstr.prior, SobolSampler(nsamples=nseeds)).result.v
    components = Array{MvNormal}(undef, nseeds)

    @info "Finding modes"

    Threads.@threads for i in 1:nseeds
        set_batcontext(ad = select_ad(length(seeds[i])))
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

# NOTE: This method contains hardcoded field names (Darkdim_radius, ca1, ca2, ca3)
# specific to Darkdim models. Consider generalizing if used with other models.
function make_init_samples(posterior, seed_points::DataFrame, nsamples::Int=10_000)
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
        set_batcontext(ad = select_ad(length(seeds[i])))
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

# ── IFT-guided profile scan ────────────────────────────────────────────────────

# Extract the fixed scalar value from a scan-point prior entry.
# Handles both ConstValueDist (from distprod with scalar) and plain numbers.
_scan_val(d::ValueShapes.ConstValueDist) = d.value
_scan_val(d::Real) = d

"""
    _ift_predict_z(pstr_prev, pstr_next, Σ_prev, z_ν)

IFT predictor in prior-normalized z-space.

At the previous optimum `z_ν`, the gradient of `pstr_prev` is zero.
The gradient of `pstr_next` at `z_ν` approximates H_zθ·dθ (cross-derivative × scan step).
One Newton step with MGVI covariance Σ_prev = H_zz⁻¹ gives the IFT prediction.

Cost: one gradient evaluation of `pstr_next` at `z_ν`.
"""
function _ift_predict_z(pstr_prev, pstr_next, Σ_prev::AbstractMatrix, z_ν::AbstractVector)
    g = ForwardDiff.gradient(z -> logdensityof(pstr_next, z), z_ν)
    z_ν .+ Σ_prev * g
end

"""
    ift_profile(likelihood, priors, vars_to_scan, params; cache_dir, start_from, polish)

IFT-guided profile posterior scan. At each scan step:
1. Compute MGVI covariance Σ at the previous optimum (actual H_zz⁻¹, not prior approximation)
2. Evaluate gradient of the new scan point's posterior at the previous z — gives H_zθ·dθ
3. Predict: z_new ≈ z_old + Σ·∇log_post_new(z_old)  (IFT Newton step)
4. Polish with bat_findmode in z-space (optional, enabled by default)

Works in prior-normalized z-space throughout (same as molewhacker); transforms back to
user space only at the end.
"""
function ift_profile(likelihood, priors, vars_to_scan, params;
                     cache_dir=nothing, start_from=nothing, polish=true)
    t_start = time()
    values, scanpoints, _ = generate_scanpoints(vars_to_scan, priors)
    grid_shape  = size(scanpoints)
    flat_sp     = vec(scanpoints)
    n_pts       = prod(grid_shape)

    center_linear = if !isnothing(start_from)
        axis_keys = collect(keys(vars_to_scan))
        idxs = [argmin(abs.(values[findfirst(==(k), axis_keys)] .- Float64(start_from[k])))
                for k in axis_keys]
        LinearIndices(grid_shape)[CartesianIndex(Tuple(idxs))]
    else
        div(n_pts, 2) + 1
    end
    order, parent = _sequential_order(grid_shape, center_linear)

    @info "IFT profile: $n_pts scan points" * (polish ? " + LBFGS polish in z-space" : " (prediction only, no polish)")

    z_results   = Vector{Vector{Float64}}(undef, n_pts)
    pstr_cache  = Vector{Any}(undef, n_pts)
    f_trafo_ref = Ref{Any}(nothing)
    opt_results = Vector{Any}(undef, n_pts)

    prog = ProgressMeter.Progress(n_pts)

    for idx in order
        sp      = flat_sp[idx]
        prior_k = distprod(; sp...)
        pstr_k, f_trafo_k = bat_transform(PriorToNormal(), PosteriorMeasure(likelihood, prior_k))
        pstr_cache[idx] = pstr_k
        isnothing(f_trafo_ref[]) && (f_trafo_ref[] = f_trafo_k)

        if parent[idx] < 0
            # Seed: project starting params to z-space at this scan point
            p_seed = deepcopy(params)
            if !isnothing(start_from)
                for (k, v) in pairs(start_from)
                    haskey(p_seed, k) && (@reset p_seed[k] = v)
                end
            end
            for k in keys(vars_to_scan)
                @reset p_seed[k] = _scan_val(sp[k])
            end
            z_init = collect(Float64, f_trafo_k(p_seed))
        else
            z_prev    = z_results[parent[idx]]
            pstr_prev = pstr_cache[parent[idx]]

            mgvi_approx = local_MGVI_approx(pstr_prev, z_prev)
            Σ           = Matrix(mgvi_approx.Σ)

            z_init = _ift_predict_z(pstr_prev, pstr_k, Σ, z_prev)
        end

        if polish
            set_batcontext(ad = select_ad(length(z_init)))
            r = bat_findmode(pstr_k, OptimizationAlg(
                optalg = Optimization.LBFGS(),
                init   = ExplicitInit([z_init]),
                kwargs = (reltol=1e-7, maxiters=1000)
            ))
            z_results[idx] = collect(Float64, r.result)
        else
            z_results[idx] = collect(Float64, z_init)
        end

        # Recover user-space params for result storage
        f_trafo    = f_trafo_ref[]
        p_nuisance = inverse(f_trafo)(z_results[idx])
        scan_keys  = Tuple(keys(vars_to_scan))
        scan_vals  = NamedTuple{scan_keys}(_scan_val(sp[k]) for k in scan_keys)
        p_result   = merge(p_nuisance, scan_vals)

        llh      = logdensityof(likelihood, p_result)
        log_post = logdensityof(PosteriorMeasure(likelihood, prior_k), p_result)
        opt_results[idx] = (llh, log_post, p_result)

        ProgressMeter.next!(prog)
    end

    res  = assemble_profile_results(opt_results, grid_shape)
    meta = Dict("task"=>"ift_profile", "priors"=>priors, "vars_to_scan"=>vars_to_scan,
                "params"=>params, "exec_time"=>time()-t_start, "sequential"=>true, "polish"=>polish)
    add_meta!(meta)
    axes = NamedTuple{tuple(keys(vars_to_scan)...)}(values)
    NewtrinosResult(axes=axes, values=res, meta=meta)
end
