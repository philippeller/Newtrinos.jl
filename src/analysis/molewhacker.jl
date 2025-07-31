using Distributions
using DataStructures
using DensityInterface
using DataFrames
#using Accessors
using Optimization, ADTypes
using MeasureBase
using LinearAlgebra
#using PositiveFactorizations
#using PDMats
using InverseFunctions
using Logging
using StatsBase
using ArraysOfArrays
using BAT
#using MGVI
using ForwardDiff
#using ProgressMeter

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
    #approx_dist = MvTDist(1, θ_sel, Σ)
    return approx_dist
end

# function make_init_samples(posterior, nsamples=10_000)
#     pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
#     @info "Finding mode"
#     result = bat_findmode(pstr, OptimizationAlg(optalg=Optimization.LBFGS()))
    
#     θ_sel = result.result

#     #smpls, _ = bat_transform(inverse(f_trafo), smpls_p)

#     approx_dist = local_MGVI_approx(pstr, θ_sel)
#     @info "Generating initial samples"
#     smpls_p = importance_sampling(pstr, approx_dist, nsamples)

#     (approx_dist=approx_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)

# end

function make_prior_samples(posterior, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    pr_dist = MvNormal(zeros(pstr.prior.dist._dim), ones(pstr.prior.dist._dim))

    @info "Generating initial samples"
    smpls_p = importance_sampling(pstr, pr_dist, nsamples)

    (approx_dist=pr_dist, samples_p=smpls_p, samples_user=bat_transform(inverse(f_trafo), smpls_p).result)

end

function make_init_samples(posterior, nseeds::Int=10, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = bat_sample(pstr.prior, SobolSampler(nsamples=nseeds)).result.v#[2:end]
    #@show seeds
    components = Array{MvNormal}(undef, nseeds)

    @info "Finding modes"
    
    Threads.@threads for i in 1:nseeds
        #@show pstr
        #@show seeds[i]
        adsel = AutoForwardDiff()
        set_batcontext(ad = adsel)
        r = bat_findmode(pstr, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([seeds[i]])))
        #@show r.result
        components[i] = local_MGVI_approx(pstr, r.result)
    end

    #@show components
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

function make_init_samples(posterior, seed_points::DataFrame, nsamples::Int=10_000)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    seeds = []

    for row in eachrow(seed_points)
        smpl = bat_sample(posterior, SobolSampler(nsamples=1)).result
        @reset smpl.v[1].Darkdim_radius = row.Darkdim_radius
        @reset smpl.v[1].ca1 = row.ca1
        @reset smpl.v[1].ca2 = row.ca2
        @reset smpl.v[1].ca3 = row.ca3
        #@reset smpl.v[1].λ₁ = row.λ₁
        #@reset smpl.v[1].λ₂ = row.λ₂
        #@reset smpl.v[1].λ₃ = row.λ₃
        push!(seeds, BAT.transform_samples(f_trafo, smpl)[1].v)
    end


    components = Array{MvNormal}(undef, length(seeds))

    @info "Finding modes"
    
    Threads.@threads for i in 1:length(seeds)
        #@show pstr
        #@show seeds[i]
        adsel = AutoForwardDiff()
        set_batcontext(ad = adsel)
        r = bat_findmode(pstr, OptimizationAlg(optalg=Optimization.LBFGS(), init = ExplicitInit([seeds[i]]), kwargs = (reltol=1e-4, maxiters=100)))
        @show r.result
        components[i] = local_MGVI_approx(pstr, r.result)
    end

    #@show components
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
    
    #μ = mean(smpls_p.v, ProbabilityWeights(smpls_p.weight))
    #Σ = cov(Matrix(flatview(smpls_p.v)'), ProbabilityWeights(smpls_p.weight))
    #approx_dist = MvNormal(μ, Σ)
    approx_dist = init_samples.approx_dist

    if init_samples.approx_dist isa MixtureModel
        approx_mix = approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist)) for approx_dist in approx_mix.components]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist)) for approx_dist in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist))]
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
        
        #approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]), mixture_w)
        
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
    
    #μ = mean(smpls_p.v, ProbabilityWeights(smpls_p.weight))
    #Σ = cov(Matrix(flatview(smpls_p.v)'), ProbabilityWeights(smpls_p.weight))
    #approx_dist = MvNormal(μ, Σ)
    approx_dist = init_samples.approx_dist

    if approx_dist isa MixtureModel
        approx_mix = init_samples.approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(d)) for d in approx_mix.components]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist)) for approx_dist in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist))]
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
        mode_logd_p_approx = Array{Any}(undef, n_parallel)

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


        
        # still need to parallelize
        new_nsamples = [floor(Int, w * length(samples_mix)) for w in last(mixture_w, n_parallel)]

        sampls_ps = Array{Any}(undef, n_parallel)

        for (i, n) in enumerate(new_nsamples)
            if n > 0
                smpls_p = importance_sampling(pstr, approx_dists[i], n)
                samples_mix = vcat(samples_mix, smpls_p)
            end
        end
        
        #approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]), mixture_w)
        
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

