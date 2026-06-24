using Test
using Newtrinos
using BAT
using Distributions
using LinearAlgebra
using DensityInterface
using MeasureBase
using FileIO
import JLD2
using DataFrames

@testset "Molewhacker" begin

    # Shared test posterior — Normal(0,1) priors make PriorToNormal the identity transform,
    # which enables analytical predictions for local_MGVI_approx covariances.
    fwd_model(a, b) = MvNormal([a, b], I(2))
    likelihood = likelihoodof(splat(fwd_model), [1.0, 0.5])
    prior = distprod(a=Normal(0, 1), b=Normal(0, 1))
    posterior = PosteriorMeasure(likelihood, prior)
    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)

    @testset "weight formula: exact by-hand calculation for N(0,1) vs N(0,2)" begin
        
        # --- By-hand verification of the weight formula ---
        # For p = N(0,1) and q = N(0,2) (σ=2 -> σ^2=4):
        # logw(x) = logp(x) - logq(x) =  log(2) -3x²/8
        # Maximum at x=0: logw(0) = log(2).
        # Normalized weights: w(x) = exp(-3x²/8)
        # w(0) = exp(0) = 1.0, w(±1) = exp(-3/8), w(±2) = exp(-3/2)
        
        p_dist = Normal(0.0, 1.0)
        q_dist = Normal(0.0, 2.0)
        test_x = [0.0, 1.0, -1.0, 2.0, -2.0]

        logd_p = logpdf.(p_dist, test_x)
        logd_q = logpdf.(q_dist, test_x)
        logw_raw = logd_p .- logd_q
        w = exp.(logw_raw .- maximum(logw_raw))

        @test w[1] == 1.0  # x=0:  max weight
        @test w[2] == w[3] == exp(-0.375) # x=+-1  (symmetric)
        @test w[4] == w[5] == exp(-1.5)  #x=+-2  (symmetric)
    end


    @testset "importance_sampling" begin

        approx_dist = MvNormal([0.0, 0.0], 2.0 * I(2))

        result = Newtrinos.importance_sampling(pstr, approx_dist, 50)
        
        #test return type and sample size
        @test result isa BAT.DensitySampleVector
        @test length(result) == 50

        #test weight bounds
        @test all(0.0 .<= result.weight .<= 1.0)
        
        #test finitenes of log values
        @test all(isfinite, result.logd)

        #verify correct computation of importance weights
        result = Newtrinos.importance_sampling(pstr, approx_dist, 200)
        logd_q_indep = logdensityof.(Ref(approx_dist), result.v)
        logw_raw = result.logd .- logd_q_indep
        w_expected = exp.(logw_raw .- maximum(logw_raw))
        @test result.weight ≈ w_expected atol=1e-10
        @test argmax(result.weight) == argmax(logw_raw)

        # samples near posterior moder (around 0.5, 0.25) get higher weight 
        # than samples far from mode 
        norms = [norm(v) for v in result.v]
        near_mask = norms .< 0.5
        far_mask  = norms .> 2.5
        if sum(near_mask) >= 5 && sum(far_mask) >= 5
            @test sum(result.weight[near_mask]) / sum(near_mask) >
                    sum(result.weight[far_mask]) / sum(far_mask)
        end
    end

    @testset "local_MGVI_approx" begin
    
        approx = Newtrinos.local_MGVI_approx(pstr, [0.5, -0.5])
        
        @test approx isa MvNormal
        @test length(mean(approx)) == 2
        @test size(Matrix(approx.Σ)) == (2, 2)
        @test issymmetric(Matrix(approx.Σ))
        @test all(eigvals(Matrix(approx.Σ)) .> 0)

        #mean should be equal to evaluation point
        @test mean(approx) ≈ [0.5, -0.5] atol=1e-10
        
        # for multivariate gaussian: Fischer matrix FI = Σ⁻¹ i.e. FI = I₂
        # also priors are already normally distributed -> Jacobian is also I₂
        # => covariance = (Jᵀ * FI * J + I)⁻¹ = 0.5 * I₂
        @test Matrix(approx.Σ) ≈ 0.5 * I(2) atol=1e-6

        # For a linear fwd model the MGVI covariance is a constant (independent of θ_sel)
        #[A forward model $f(a, b)$ is linear if its mapping from the parameter space to the 
        #data space can be written as a matrix multiplication:\mu(a, b) = M * (a, b) ]
        approx1 = Newtrinos.local_MGVI_approx(pstr, [0.0, 0.0])
        approx2 = Newtrinos.local_MGVI_approx(pstr, [1.5, -1.0])
        @test Matrix(approx1.Σ) ≈ Matrix(approx2.Σ) atol=1e-8
    end

    @testset "make_prior_samples" begin
        result = make_prior_samples(posterior, 100)

        #test fields and types
        @test result isa NamedTuple
        @test haskey(result, :approx_dist)
        @test haskey(result, :samples_p)
        @test haskey(result, :samples_user)
        @test result.approx_dist isa MvNormal
        @test result.samples_p isa BAT.DensitySampleVector
        @test result.samples_user isa BAT.DensitySampleVector

        #test approx_dist is a standard Normal distribution (N(0, I))
        @test mean(result.approx_dist) == [0.0, 0.0]
        @test Matrix(result.approx_dist.Σ) == I(2)

        #test sample size and normalization
        @test length(result.samples_p) == length(result.samples_user) == 100
        @test all(result.samples_p.weight .<= 1.0)
        @test all(isfinite, result.samples_p.logd)
        @test !(result.samples_p.v isa NamedTuple)  #samples_p not in parameter space with a,b 

        #test if samples_user is in user parameter space
        @test haskey(result.samples_user.v[1], :a)
        @test haskey(result.samples_user.v[1], :b)
    end

    @testset "make_init_samples" begin
        # nseeds=2, nsamples=100 to keep the LBFGS optimization fast
        result = make_init_samples(posterior, 2, 100)

        #test fields and types
        @test result isa NamedTuple
        @test haskey(result, :approx_dist)
        @test haskey(result, :samples_p)
        @test haskey(result, :samples_user)
        @test result.approx_dist isa MixtureModel
        @test result.samples_p isa BAT.DensitySampleVector
        @test result.samples_user isa BAT.DensitySampleVector

        #test approx_dist, samples_p, and samples_user properties
        @test length(result.approx_dist.components) == 2 # nseeds=2
        for comp in result.approx_dist.components
            @test comp isa MvNormal
            @test length(mean(comp)) == 2
            @test mean(comp) ≈ [0.5, 0.25] atol=1e-6 #mean at optimization result
            @test Matrix(comp.Σ) ≈ 0.5 * I(2) atol=1e-6
        end

        @test length(result.samples_p) == 100
        @test all(0.0 .<= result.samples_p.weight .<= 1.0)
        @test !(result.samples_p.v[1] isa NamedTuple)

        @test result.samples_user.v[1] isa NamedTuple
        @test length(result.samples_user) == length(result.samples_p)
        @test haskey(result.samples_user.v[1], :a)
        @test haskey(result.samples_user.v[1], :b)

    end


    @testset "make_init_samples: DataFrame seeded variant" begin
        # This method (molewhacker.jl lines 87-126) is hardcoded to use the field names
        # Darkdim_radius, ca1, ca2, ca3 via @reset. Build a posterior with those exact names.
        fwd_dd(Darkdim_radius, ca1, ca2, ca3) = MvNormal([Darkdim_radius, ca1, ca2, ca3], I(4))
        lik_dd = likelihoodof(splat(fwd_dd), [1.0, 0.5, 0.2, -0.3])
        prior_dd = distprod(Darkdim_radius=Normal(0,1), ca1=Normal(0,1), ca2=Normal(0,1), ca3=Normal(0,1))
        posterior_dd = PosteriorMeasure(lik_dd, prior_dd)
        seed_df = DataFrame(
            Darkdim_radius=[1.0, 0.5],
            ca1=[0.5, -0.5], ca2=[0.2, 0.1], ca3=[-0.3, 0.0]
        )
        result = make_init_samples(posterior_dd, seed_df, 50)

        @test result isa NamedTuple
        @test haskey(result, :approx_dist) && haskey(result, :samples_p) && haskey(result, :samples_user)
        @test result.approx_dist isa MixtureModel
        @test length(result.approx_dist.components) == nrow(seed_df)  # one component per seed row
        for comp in result.approx_dist.components
            @test comp isa MvNormal
            @test length(mean(comp)) == 4
        end
        @test length(result.samples_p) == 50
        @test all(0.0 .<= result.samples_p.weight .<= 1.0)
        @test result.samples_user.v[1] isa NamedTuple
        @test haskey(result.samples_user.v[1], :Darkdim_radius)
        @test haskey(result.samples_user.v[1], :ca1)
        @test haskey(result.samples_user.v[1], :ca2)
        @test haskey(result.samples_user.v[1], :ca3)
    end


    @testset "whack_a_mole" begin
        prior_samples = make_prior_samples(posterior, 100) #returns 100 samples from a one component mixtureModel 
        init_samples = make_init_samples(posterior, 2, 50) # 50 samples with #nseeds = 2

        @testset "fields,types" begin
            result = Newtrinos.whack_a_mole(posterior, prior_samples, 3)   
            @test result isa NamedTuple
            @test haskey(result, :approx_dist)
            @test haskey(result, :samples_p)
            @test haskey(result, :samples_user)
            @test result.approx_dist isa MixtureModel
            for comp in result.approx_dist.components
                @test comp isa MvNormal
            end
        end
        
        @testset "samples_user is in user parameter space" begin
            result = Newtrinos.whack_a_mole(posterior, prior_samples, 1)
            @test result.samples_user.v[1] isa NamedTuple
            @test haskey(result.samples_user.v[1], :a)
            @test haskey(result.samples_user.v[1], :b)
        end

        @testset "test components and weights" begin  
            #with init_samples
            result_init = Newtrinos.whack_a_mole(posterior, init_samples, 3) #nwhack = 3
            @test length(result_init.approx_dist.components) == 5 #nseeds + nwhack

            #with prior samples
            result_0whacks = Newtrinos.whack_a_mole(posterior, prior_samples, 0) #adds 0 components
            result_2whacks = Newtrinos.whack_a_mole(posterior, prior_samples, 2) #adds 2 components
            @test length(result_0whacks.approx_dist.components) == 1
            @test length(result_2whacks.approx_dist.components) == 3

            #test weights are valid (i.e. positive and normalized) + check weights recomputation in final step
            for result in [result_init, result_0whacks, result_2whacks]
                #posittive and normalized
                @test sum(probs(result.approx_dist)) ≈ 1.0 atol=1e-6 #probs sum to 1
                @test all(0.0 .<= probs(result.approx_dist) .<= 1.0) #probs are in interval [0,1]
                @test maximum(result.samples_p.weight) ≈ 1.0 #max weight is 1
                @test all(result.samples_p.weight .> 0.0)  #positive weights

                #recomputation w[i] = exp(logd_p[i] - logd_q_mix[i] - max(logd_p - logd_q_mix))
                logd_p = result.samples_p.logd
                logd_q = logdensityof.(Ref(result.approx_dist), result.samples_p.v)
                logw_raw = logd_p .- logd_q
                w_expected = exp.(logw_raw .- maximum(logw_raw))
                @test result.samples_p.weight ≈ w_expected atol=1e-10
            end
        end
    end

    #the core algorithm of whack_many_moles is the same as in whack_a_mole 
    #the differences are: 
    # 1) parallelism: grab more than one of highest weight samples and build new gaussians simultaneously
    # 2) convergence criteria: stops when eff > target_efficiency, iter > maxiter, or ess > target_samplesize
    # 3) checkpointing by saving files to a cache directory
    # => focus tests on those points 
    @testset "whack_many_moles" begin
        prior_samples = make_prior_samples(posterior, 100) #returns 100 samples from a one component mixtureModel 
        init_samples = make_init_samples(posterior, 2, 50) # 50 samples with #nseeds = 2

        @testset "fields,types" begin
            result = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter = 0, n_parallel=1)
            @test result isa NamedTuple
            @test haskey(result, :approx_dist)
            @test haskey(result, :samples_p)
            @test haskey(result, :samples_user)
            @test result.approx_dist isa MixtureModel
            for comp in result.approx_dist.components
                @test comp isa MvNormal
            end
        end

        @testset "parallelism and convergence criteria" begin
            #maxiter = N , starts at 0 => stops after N+1 iterations 
            # => #final_components = 1  + (N+1)*k with k = n_parallel
            #test that only 1+(n+1)*k components are generated => can simultaneously test parallelism for various k!=1
            result_1it = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter=1, n_parallel=1)
            result_7it = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter=7, n_parallel=1)
            result_1it_9pa = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter=1, n_parallel=9)
            result_2it_2pa = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter=2, n_parallel=2)
            result_7it_3pa = Newtrinos.whack_many_moles(posterior, prior_samples, maxiter=7, n_parallel=3)
            
            @test length(result_1it.approx_dist.components) == 3 #1+(1+1)*1 = 3
            @test length(result_7it.approx_dist.components) == 9 #1+(7+1)*1 = 9
            @test length(result_1it_9pa.approx_dist.components) == 19 #1+(1+1)*9 = 19
            @test length(result_2it_2pa.approx_dist.components) == 7 #1+(2+1)*2 = 7
            @test length(result_7it_3pa.approx_dist.components) == 25 #1+(7+1)*3 = 25

            #target_efficiency: eff = ess/n > 0 always 
            # => target_eff = 0.0 triggers stop immediately -> 0 iterations => 1 component
            result_0targeteff = Newtrinos.whack_many_moles(posterior, prior_samples; target_efficiency=0.0, maxiter=100, n_parallel=1)
            @test length(result_0targeteff.approx_dist.components) == 1
            #test different target efficiencies
            for target_eff in [0.5, 0.6, 0.7, 0.8, 0.9]
                result_target_eff = Newtrinos.whack_many_moles(posterior, prior_samples; target_efficiency=target_eff, maxiter=100, n_parallel=1)
                samples_mix = result_target_eff.samples_p
                #eff calculation used inside whack_many_moles:
                ess = bat_eff_sample_size(samples_mix, KishESS()).result 
                eff = ess / length(samples_mix)
                @test eff > target_eff
            end

            #target_samplesize
            # target_samplesize = 0 triggers stop immediately -> 0 iterations => 1 component
            result_0targetss = Newtrinos.whack_many_moles(posterior, prior_samples; target_samplesize=0, maxiter=1000, n_parallel=1) # high maxiter s.t. not terminated by iteration count
            @test length(result_0targetss.approx_dist.components) == 1
            #test different target sample sizes
            for sample_size in [10, 100, 200, 1000, 5000]
                result_targetss = Newtrinos.whack_many_moles(posterior, prior_samples; target_samplesize=sample_size, maxiter=1000, n_parallel=1) #high maxiter s.t. not terminated by iteration count
                samples_mix = result_targetss.samples_p
                #ess calculation used inside whack_many_moles:
                ess = bat_eff_sample_size(samples_mix, KishESS()).result
                @test ess > sample_size
            end
        end

        @testset "checkpointing" begin
            #test that a file is created in cache_dir for each iteration 
            mktempdir() do cache_dir
                whack_many_moles(posterior, prior_samples; maxiter=1, n_parallel=1, cache_dir=cache_dir)
                saved = readdir(cache_dir)
                @test length(saved) == 2
                @test "molewhacker_iter_1.jld2" in saved
                @test "molewhacker_iter_2.jld2" in saved
            end
        end
    end

    @testset "non-Normal priors fwd model" begin
        # With Uniform priors, PriorToNormal applies the probit transform u = Φ⁻¹((a+3)/6),
        # unlike the Normal(0,1) case where it is the identity.
        fwd_u(a, b) = MvNormal([a, b], I(2))
        lik_u  = likelihoodof(splat(fwd_u), [1.0, 0.5])
        prior_u = distprod(a=Uniform(-3.0, 3.0), b=Uniform(-3.0, 3.0))
        posterior_u = PosteriorMeasure(lik_u, prior_u)
        pstr_u, _ = bat_transform(PriorToNormal(), posterior_u)

        @testset "local_MGVI_approx: covariance matches analytical prediction for Uniform prior" begin
            # At u=[0,0] the inverse transform gives a=0, b=0.
            # Jacobian da/du|_{u=0} = 6·φ(0) = 6/√(2π)  (both parameters symmetric)
            # J = diag(6/√(2π), 6/√(2π)),  FI = I(2)
            # Σ = inv(J'·J + I) = (2π/(36+2π))·I
            Σ_diag = 2π / (36 + 2π)  # ≈ 0.1486
            approx_u = Newtrinos.local_MGVI_approx(pstr_u, [0.0, 0.0])
            @test approx_u isa MvNormal
            @test mean(approx_u) ≈ [0.0, 0.0] atol=1e-10
            @test Matrix(approx_u.Σ) ≈ Σ_diag * I(2) atol=1e-6
            # Distinct from the Normal-prior covariance 0.5·I
            @test !isapprox(Matrix(approx_u.Σ), 0.5 * I(2), atol=0.01)
        end

        @testset "make_prior_samples works and produces samples within prior support" begin
            result = make_prior_samples(posterior_u, 100)
            @test result.approx_dist isa MvNormal
            @test result.samples_user.v[1] isa NamedTuple
            @test haskey(result.samples_user.v[1], :a)
            @test haskey(result.samples_user.v[1], :b)
            for s in result.samples_user.v
                @test -3.0 <= s.a <= 3.0
                @test -3.0 <= s.b <= 3.0
            end
        end

        @testset "whack_many_moles works and produces samples within prior support" begin
            init = make_prior_samples(posterior_u, 100)
            result = whack_many_moles(posterior_u, init; maxiter=1, n_parallel=1)
            @test result.approx_dist isa MixtureModel
            @test length(result.approx_dist.components) == 3
            for s in result.samples_user.v
                @test -3.0 <= s.a <= 3.0
                @test -3.0 <= s.b <= 3.0
            end
        end
    end

end
