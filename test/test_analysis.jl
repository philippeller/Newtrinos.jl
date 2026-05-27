using DataStructures
using Test
using Newtrinos
using BAT
using Distributions
using LinearAlgebra
using DensityInterface
using MeasureBase
using ValueShapes

struct MockPhysics <: Newtrinos.Physics
    params::NamedTuple
    priors::NamedTuple
end

struct MockExperiment <: Newtrinos.Experiment
    physics::MockPhysics
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

struct ArgumentErrorDensity end
DensityInterface.DensityKind(::ArgumentErrorDensity) = IsDensity()
DensityInterface.logdensityof(::ArgumentErrorDensity, x) = throw(ArgumentError("forced"))

@testset "Analysis Tools" begin

    @testset "NewtrinosResult" begin
        axes = (x=[1.0, 2.0, 3.0],)
        values = (llh=[-10.0, -5.0, -8.0], log_posterior=[-10.0, -5.0, -8.0])
        result = Newtrinos.NewtrinosResult(axes=axes, values=values)

        @test result.axes == axes
        @test result.values == values
        @test result.meta isa Dict
    end

    @testset "sort_nt" begin
        nt = (b=3, a=1, c=2)
        sorted = Newtrinos.sort_nt(nt)
        @test keys(sorted) == (:a, :b, :c)
        @test sorted.a == 1
        @test sorted.b == 3
        @test sorted.c == 2

        # Already sorted
        nt2 = (a=1, b=2)
        @test Newtrinos.sort_nt(nt2) == nt2

        # Single element
        nt1 = (x=42,)
        @test Newtrinos.sort_nt(nt1) == nt1
    end

    @testset "safe_merge" begin
        a = (x=1, y=2)
        b = (z=3,)
        merged = Newtrinos.safe_merge(a, b)
        @test haskey(merged, :x)
        @test haskey(merged, :y)
        @test haskey(merged, :z)
        @test merged.x == 1
        @test merged.z == 3

        # Duplicate keys with same values should work
        c = (x=1,)
        merged2 = Newtrinos.safe_merge(a, c)
        @test merged2.x == 1

        # Duplicate keys with different values should error
        d = (x=99,)
        @test_throws ErrorException Newtrinos.safe_merge(a, d)

        # Result should be sorted by key
        e = (z=1, a=2)
        f = (m=3,)
        merged3 = Newtrinos.safe_merge(e, f)
        @test keys(merged3) == (:a, :m, :z)
    end

    @testset "Wrapper and Base.getproperty" begin
        mock_physics = MockPhysics(
            (mu = 5.0,),
            (mu = Uniform(0.0, 10.0),)
        )
        mock_exp = MockExperiment(
            mock_physics,
            (scale = 1.0,),
            (scale = Uniform(0.5, 2.0),),
            (observed = [5.0],),
            p -> MvNormal([p.mu * p.scale], I(1)),
            (p, data=nothing) -> nothing
        )

        aliases = Dict(:mu => :mean_value)
        wrapper = Newtrinos.Wrapper(mock_exp, aliases)

        params = Newtrinos.get_params(wrapper)
        @test haskey(params, :mean_value)
        @test !haskey(params, :mu)
        @test params.mean_value == 5.0
        @test params.scale == 1.0

        priors = Newtrinos.get_priors(wrapper)
        @test haskey(priors, :mean_value)
        @test priors.mean_value isa Uniform

        # test correct call of Base.getproperty branches
        @test Newtrinos.Base.getproperty(wrapper, :forward_model) == wrapper.forward_model
        @test Newtrinos.Base.getproperty(wrapper, :plot) == wrapper.plot
        @test Newtrinos.Base.getproperty(wrapper, :aliases) == wrapper.aliases

        # getproperty branch forward_model: translates aliased -> original names
        dist = wrapper.forward_model((mean_value=3.0, scale=1.0))
        @test mean(dist) ≈ [3.0]

        # getproperty branch plot: applies the same alias translation
        wrapper.plot((mean_value=3.0, scale=1.0))   # must not throw

        # any other field delegates to the inner experiment returns (getfield(wrapper.x, name)),with x::Newtrinos.Experiment
        @test wrapper.assets == mock_exp.assets
        @test wrapper.physics === mock_exp.physics
    end

    @testset "get_params and get_priors" begin
        # Physics object
        osc = Newtrinos.osc.configure()
        
        # Experiment object
        mock_physics = MockPhysics(
            (mu = 5.0,),
            (mu = Uniform(0.0, 10.0),)
        )
        mock_exp = MockExperiment(
            mock_physics,
            (scale = 1.0,),
            (scale = Uniform(0.5, 2.0),),
            (observed = [5.0],),
            p -> MvNormal([p.mu * p.scale], I(1)),
            (p, data=nothing) -> nothing
        )

        # wrapper object
        aliases = Dict(:mu => :mean_value)  
        wrapper = Newtrinos.Wrapper(mock_exp, aliases)

        # named Tuple of modules
        modules = (osc=osc, mock_exp=mock_exp)

        # test get_params 
        params_physics = Newtrinos.get_params(mock_physics)
        params_experiment = Newtrinos.get_params(mock_exp)
        params_nt = Newtrinos.get_params(modules)
        params_wrapper = Newtrinos.get_params(wrapper)

        @test haskey(params_physics, :mu)
        @test haskey(params_experiment, :scale)
        @test haskey(params_nt, :scale)
        @test haskey(params_nt, :θ₁₂)
        @test haskey(params_wrapper, :mean_value)
        @test haskey(params_wrapper, :scale)

        @test params_physics.mu == params_experiment.mu == params_nt.mu == params_wrapper.mean_value == 5.0
        @test params_experiment.scale == params_nt.scale == params_wrapper.scale == 1.0

        # test get_priors
        priors_physics = Newtrinos.get_priors(mock_physics)
        priors_experiment = Newtrinos.get_priors(mock_exp)  
        priors_nt = Newtrinos.get_priors(modules)
        priors_wrapper = Newtrinos.get_priors(wrapper)

        @test haskey(priors_physics, :mu)
        @test haskey(priors_experiment, :scale)
        @test haskey(priors_nt, :scale)
        @test haskey(priors_nt, :θ₁₂)
        @test haskey(priors_wrapper, :mean_value)
        @test haskey(priors_wrapper, :scale)

        @test priors_physics.mu == priors_experiment.mu == priors_nt.mu == priors_wrapper.mean_value
        @test priors_experiment.scale == priors_nt.scale == priors_wrapper.scale

        #test that params are within the support of the priors
        @test Distributions.insupport(priors_physics.mu, params_physics.mu)
        @test Distributions.insupport(priors_experiment.scale, params_experiment.scale)

        for k in keys(params_physics)
            @test Distributions.insupport(priors_physics[k], params_physics[k])
        end
    end    
    
    @testset "conditional priors" begin
        priors = (a=Uniform(0.0, 1.0), b=Uniform(0.0, 1.0), c=Uniform(0.0, 1.0))
        params = (a=0.4, b=0.5, c=0.7)

        # conditional_vars = Array variant: fix :a to its value in params
        conditioned = Newtrinos.condition(priors, [:a], params)
        @test conditioned.a == 0.4
        @test conditioned.b isa Distribution
        @test conditioned.c isa Distribution

        # conditional_vars = Dict variant: :a to explicit value, :b => nothing uses params
        conditioned2 = Newtrinos.condition(priors, Dict(:a => 1, :b => nothing), params)
        @test conditioned2.a == 1.0
        @test conditioned2.b == 0.5
        @test conditioned2.c isa Distribution

        # test changing the prior distribution
        conditioned3 = Newtrinos.condition(priors, Dict(:a => Uniform(0.0, 0.5)), params)
        @test conditioned3.a == Uniform(0.0, 0.5)
    end

    @testset "get_observed, get_fwd_model, and generate_likelihood" begin
        mock_exp1 = MockExperiment(
            MockPhysics((mu=1.0,), (mu=Uniform(0.0, 5.0),)),
            (scale=1.0,), (scale=Uniform(0.5, 2.0),),
            (observed=[1.0, 2.0],), # observed data as an asset
            p -> MvNormal([p.mu, p.mu * p.scale], I(2)), # forward model
            (p, data=nothing) -> nothing
        )
        mock_exp2 = MockExperiment(
            MockPhysics((mu=3.0,), (mu=Uniform(0.0, 5.0),)),
            (scale=2.0,), (scale=Uniform(0.5, 2.0),),
            (observed=[3.0, 4.0, 5.0],), # observed data as an asset
            p -> MvNormal([p.mu, p.mu * p.scale, p.mu + p.scale], I(3)), # forward model
            (p, data=nothing) -> nothing
        )
        experiments = (exp1=mock_exp1, exp2=mock_exp2)

        #test get_observed
        observed = Newtrinos.get_observed(experiments)
        @test keys(observed) == keys(experiments)
        @test observed.exp1 == [1.0, 2.0]
        @test observed.exp2 == [3.0, 4.0, 5.0]

        # test get_fwd_model
        params = (mu=2.0, scale=1.5)
        combined_fwd = Newtrinos.get_fwd_model(experiments)
        dist = combined_fwd(params)
        @test dist isa ValueShapes.NamedTupleDist
        
        # sampling yields a NamedTuple with the right keys and dimensions
        s = rand(dist)
        @test s isa NamedTuple
        @test haskey(s, :exp1)
        @test haskey(s, :exp2)
        @test length(s.exp1) == 2   # dim matches exp1's MvNormal
        @test length(s.exp2) == 3   # dim matches exp2's MvNormal

        # component means match what each forward model produces individually
        @test mean(dist).exp1 ≈ mean(mock_exp1.forward_model(params))
        @test mean(dist).exp2 ≈ mean(mock_exp2.forward_model(params))

        #test generate_likellihood: combined likelihood should equal product of individual likelihoods
        combined_likelihood = Newtrinos.generate_likelihood(experiments)(params)
        likelihood1 = Newtrinos.generate_likelihood((exp1=mock_exp1,))(params)
        likelihood2 = Newtrinos.generate_likelihood((exp2=mock_exp2,))(params)
        @test combined_likelihood == likelihood1 * likelihood2
    end

    @testset "correlated_priors_vars" begin
        priors = (x = Normal(0, 1), y = Normal(0, 1), z = Exponential(1.0))
        
        # Define a correlated 2D normal distribution for x and y with mean [0,0], and covariance with 0.5 correlation
        target_dist = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])

        corr_prior, other_prior = Newtrinos.correlated_priors_vars(priors, [:x, :y], target_dist)
        
        # 'z' should be the only thing left in other_prior
        s = rand(other_prior) #other_prior is a NamedTupleDist, not a plain NamedTuple => haskey has no method for it.
        # The fix is to check keys through a sample — rand(NamedTupleDist) always returns a NamedTuple which does support haskey
        @test haskey(s, :z)
        @test !haskey(s, :x)
        @test !haskey(s, :y)
        @test s.z isa Real

        # 'x' and 'y' should be in the shape of corr_prior
        @test :x in keys(corr_prior().shape)
        @test :y in keys(corr_prior().shape)
    
        # Test statistical property: Check logdensity at a correlated point
        # At [1, 1], a correlated dist (0.5) should have higher density than an uncorrelated one (0.0)
        pt_correlated = (x = 1.0, y = 1.0)
        @test logdensityof(corr_prior(), pt_correlated) > logdensityof(MvNormal([0.0, 0.0], I(2)), [1.0, 1.0])
    end

    @testset "generate_toy_data" begin
        mock_exp1 = MockExperiment(
            MockPhysics((mu=3.0, sigma=4.3), (mu=Uniform(0.0, 5.0), sigma=Uniform(4.0, 5.0))),
            (;), (;),
            (observed=[3.0, 4.3],),
            p -> MvNormal([p.mu, p.sigma], I(2)),
            (p, data=nothing) -> nothing
        )
        params = (mu=3.0, sigma=4.0)

        mock_exp2 = MockExperiment(
            MockPhysics((mu=6.0,), (mu=Uniform(4.0, 8.0),)),
            (;), (;),
            (observed=[6.0],),
            p -> MvNormal([p.mu], 1.0),
            (p, data=nothing) -> nothing
        )

        experiments = (exp1=mock_exp1, exp2=mock_exp2)

        #test generate_toy_data
        toy1 = Newtrinos.generate_toy_data(mock_exp1, params)
        toy2 = Newtrinos.generate_toy_data(mock_exp2, params)
        toy_combined = Newtrinos.generate_toy_data(experiments, params)

        @test length(toy1) == length(toy_combined.exp1) == 2 # since 2dim forward model
        @test length(toy2) == length(toy_combined.exp2) == 1 # since 1dim forward model
        @test toy1 isa AbstractVector
        @test toy_combined isa NamedTuple
        @test length(toy_combined) == 2 # 2 experiments
    end

    @testset "generate_asimov_data" begin
        # Poisson model: asimov data should be rounded to integers
        mock_pois = MockExperiment(
            MockPhysics((mu = 2.0,), (mu = Uniform(0.0, 10.0),)),
            (scale = 1.0,),
            (scale = Uniform(0.5, 2.0),),
            (observed = zeros(Int, 2),),
            p -> distprod(Poisson.([p.mu * 5, p.mu * 10])), # λ = mu * [5, 10] -> [10, 20] for mu=2
            (p, data=nothing) -> nothing
        )
        asimov_pois = Newtrinos.generate_asimov_data(mock_pois, (mu = 2.0, scale = 1.0))
        @test all(x -> x isa Integer, asimov_pois)
        @test asimov_pois == [10, 20]

        # Gaussian model: asimov data should be floating point
        mock_gauss = MockExperiment(
            MockPhysics((mu = 3.5,), (mu = Uniform(0.0, 10.0),)),
            (scale = 1.0,),
            (scale = Uniform(0.5, 2.0),),
            (observed = [0.0, 0.0],),
            p -> MvNormal([p.mu, p.mu * p.scale], I(2)), # μ = [mu, mu * scale] -> [3.5, 7.0] for mu=3.5, scale=2.0
            (p, data=nothing) -> nothing
        )
        asimov_gauss = Newtrinos.generate_asimov_data(mock_gauss, (mu = 3.5, scale = 2.0))
        @test asimov_gauss isa AbstractArray{<:AbstractFloat}
        @test asimov_gauss ≈ [3.5, 7.0]

        combined_experiments = (pois=mock_pois, gauss=mock_gauss)
        asimov_combined = Newtrinos.generate_asimov_data(combined_experiments, (mu=2.0, scale=1.0))
        @test asimov_combined isa NamedTuple
        @test length(asimov_combined) == 2 # 2 experiments
        @test asimov_combined.pois == asimov_pois
        @test asimov_combined.gauss == Newtrinos.generate_asimov_data(mock_gauss, (mu = 2.0, scale = 1.0))  # other params in combined asimov
    end

    @testset "find_mle" begin
        fwd(a, b) = MvNormal([a, b], I(2)) # forward model with 2 uncorrelated parameters
        likelihood = likelihoodof(splat(fwd), [0.5, 0.5]) # 2nd arg = measured params that enter the likelihood 
        prior = distprod(a=Normal(0,1), b=Normal(0,1)) 
        params = (a=0.5, b=-0.5) # initial params for optimization
        llh, log_posterior, result = Newtrinos.find_mle(likelihood, prior, params)

        @test isfinite(llh)
        @test isfinite(log_posterior)
        @test abs(result.a) ≈ 0.25 atol=1e-6
        @test abs(result.b) ≈ 0.25 atol=1e-6
        @test llh ≈ log(likelihood((result.a, result.b))) atol=1e-6
        # algorithm doesnt include marginalized evidence -> log(posterior) = log(likelihood) + log(prior) (+ const_evidence) 
        @test log_posterior ≈ logdensityof(likelihood, (result.a, result.b)) + logdensityof(prior, (a=result.a, b=result.b)) atol=1e-6

        #fix one value with ConstValueDist and optimize the other to match the observed
        prior2 = distprod(a = ConstValueDist(0.5), b = Normal(0, 1))
        llh2, log_posterior2, result2 = Newtrinos.find_mle(likelihood, prior2, params)
        @test abs(result2.a) == 0.5  # held fixed by ConstValueDist
        @test abs(result2.b) ≈ 0.25 atol=1e-6  # optimized to observed 
    end

    @testset "find_mle catches ArgumentError" begin
        prior  = distprod(a=Uniform(0.0, 1.0), b=Uniform(0.0, 1.0))
        params = (a=0.5, b=0.5)

        llh, log_posterior, result = Newtrinos.find_mle(ArgumentErrorDensity(), prior, params)

        @test isnan(llh)
        @test isnan(log_posterior)
        @test all(isnan, values(result))
    end

    @testset "find_mle_cached" begin
        fwd(a, b) = MvNormal([a, b], I(2))
        observed = [2.0, 3.0]
        likelihood = likelihoodof(splat(fwd), observed)
        prior = distprod(a = Uniform(-5.0, 5.0), b = Uniform(-5.0, 5.0))
        params = (a = 0.0, b = 0.0)

        mktempdir() do cache_dir
            # First call: should compute and write a cache file
            llh1, lp1, r1 = Newtrinos.find_mle_cached(likelihood, prior, params, cache_dir)
            @test length(readdir(cache_dir)) == 1

            # Second call with identical inputs: should read from cache (no new file)
            llh2, lp2, r2 = Newtrinos.find_mle_cached(likelihood, prior, params, cache_dir)
            @test length(readdir(cache_dir)) == 1
            @test llh1 ≈ llh2
            @test r1.a ≈ r2.a
            @test r1.b ≈ r2.b

            #third call with different params: should compute a new result and write a new cache file
            params_new = (a = 1.0, b = 1.0)
            llh3, lp3, r3 = Newtrinos.find_mle_cached(likelihood, prior, params_new, cache_dir)
            @test length(readdir(cache_dir)) == 2
        end
    end

    @testset "_generate_grid" begin
        priors = (a=Uniform(0.0, 2.0), b=Uniform(-1.0, 1.0))
        vars_to_scan = OrderedDict(:a => 3, :b => 4)

        vars, values, mesh = Newtrinos._generate_grid(vars_to_scan, priors)

        @test vars == [:a, :b]
        @test length(values) == 2
        @test length(values[1]) == 3 # as assigned in vars_to_scan for :a
        @test length(values[2]) == 4 # as assigned in vars_to_scan for :b

        # endpoints are at quantile(0) and quantile(1) of each prior
        @test values[1][begin] ≈ quantile(Uniform(0.0, 2.0), 0.0)
        @test values[1][end]   ≈ quantile(Uniform(0.0, 2.0), 1.0)
        @test values[2][begin] ≈ quantile(Uniform(-1.0, 1.0), 0.0)
        @test values[2][end]   ≈ quantile(Uniform(-1.0, 1.0), 1.0)

        # mesh is the full Cartesian product
        @test size(mesh) == (3, 4)
        @test length(mesh) == 12

        # each mesh element is a tuple of one value per scanned variable
        @test mesh[1, 1] isa Tuple
        @test length(mesh[1, 1]) == 2
    end

    @testset "generate_scanpoints" begin
        priors = (x=Uniform(0.0, 2.0), y=Uniform(-3.0, 3.0))

        # 1D scan
        values, scanpoints = Newtrinos.generate_scanpoints(OrderedDict(:x => 5), priors)

        @test length(values) == 1
        @test length(values[1]) == 5
        @test size(scanpoints) == (5,)
        @test all(sp -> sp isa Distribution, scanpoints)
        # prior of y is not touched, since we scan only over x
        [@test scanpoints[i].y == Uniform(-3.0, 3.0) for i in 1:length(scanpoints)] 
        # scangrid values match what _generate_grid would produce
        @test values == Newtrinos._generate_grid(OrderedDict(:x => 5), priors)[2]

        # 2D scan: scanpoints array has the right shape
        _, scanpoints_2d = Newtrinos.generate_scanpoints(OrderedDict(:x => 3, :y => 4), priors)
        @test size(scanpoints_2d) == (3, 4)
        @test scanpoints_2d[1,1].x == ConstValueDist{Univariate, Float64}(0.0)
        @test scanpoints_2d[3,1].x == ConstValueDist{Univariate, Float64}(2.0)
        @test scanpoints_2d[1,1].y == ConstValueDist{Univariate, Float64}(-3.0)
        @test scanpoints_2d[1,4].y == ConstValueDist{Univariate, Float64}(3.0)
    end

    @testset "assemble_profile_results" begin
        opt_results = [
            (-10.0, -12.0, (a=1.0, b=2.0)), # llh, log_posterior, params output of find_mle
            (-5.0,  -7.0,  (a=3.0, b=4.0)),
            (-8.0,  -10.0, (a=5.0, b=6.0)),
            (-9.0,  -11.0, (a=7.0, b=8.0))
        ]

        res = Newtrinos.assemble_profile_results(opt_results, (4,))

        @test res isa NamedTuple
        @test size(res.llh) == (4,)
        @test res.llh           == [-10.0, -5.0, -8.0, -9.0]
        @test res.log_posterior == [-12.0, -7.0, -10.0, -11.0]
        @test res.a             == [1.0, 3.0, 5.0, 7.0]
        @test res.b             == [2.0, 4.0, 6.0, 8.0]

        # 2D result_size: the llh, log_posterior, and params objects should be reshaped to 2x2 matrices when we spcify result_size=(2, 2)
        res2d = Newtrinos.assemble_profile_results(opt_results, (2, 2))

        @test size(res2d.llh) == (2, 2)
        @test size(res2d.a)   == (2, 2)
        @test res2d.llh[1] == -10.0
        @test res2d.llh[4] == -9.0
    end

    @testset "scan" begin
        fwd(a, b) = MvNormal([a, b], I(2))
        likelihood = likelihoodof(splat(fwd), [1.0, 2.0])
        priors = distprod(a=Uniform(-3.0, 3.0), b=Uniform(-3.0, 3.0))
        params = (a=0.0, b=0.0)

        #1D scan: result should have 1D arrays for llh, log_posterior, and params, with length matching the number of scan points
        result = Newtrinos.scan(likelihood, priors, OrderedDict(:a => 5), params)

        @test result isa Newtrinos.NewtrinosResult
        @test result.meta["task"] == "scan"
        @test haskey(result.meta, "exec_time")
        @test length(result.axes.a) == length(result.values.llh) == length(result.values.log_posterior) == 5
        @test result.meta["task"] == "scan"
        # non-scanned param :b is stored as a Fill array at its fixed value
        @test all(==(0.0), result.values.b)
        # best LLH should be at the scan point closest to observed[1] = 1.0
        bf = Newtrinos.bestfit(result)
        @test abs(bf.a - 1.0) < 1.5

        #2D scan: result should have 2D arrays for llh, log_posterior, and params, with dimensions matching the scan grid
        result_2d = Newtrinos.scan(likelihood, priors, OrderedDict(:a => 3, :b => 4), params)

        @test result_2d isa Newtrinos.NewtrinosResult
        @test size(result_2d.values.llh) == (3, 4)
        @test length(result_2d.axes.a) == 3
        @test length(result_2d.axes.b) == 4

        # test that if gradient_map=true, the result includes gradient arrays 
        result_grad = Newtrinos.scan(likelihood, priors, OrderedDict(:a => 3, :b => 4), params; gradient_map=true)

        @test haskey(result_grad.values, :a_grad)
        @test haskey(result_grad.values, :b_grad)
        @test size(result_grad.values.a_grad) == size(result_grad.values.b_grad) == (3, 4)
        @test length(result_grad.values.a_grad) == 12 #3x4 entries in the scan grid
    end

    @testset "profile scans" begin

        @testset "profile" begin

            @testset "profile with nuisance optimization" begin
                fwd(a, b) = MvNormal([a, b], I(2))
                likelihood = likelihoodof(splat(fwd), [1.0, 0.0])
                priors = (a=Uniform(-3.0, 3.0), b=Uniform(-2.0, 2.0))
                params = (a=0.0, b=0.0)

                # "fix" only a to gridpoints -> can still optimize over b as a nuisance param -> profile scans over a but optimizes over b at each a value
                result_profile = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3), params)
                @test result_profile isa Newtrinos.NewtrinosResult
                @test length(result_profile.axes.a) == 3
                @test size(result_profile.values.llh) == (3,)
                @test result_profile.meta["task"] == "profile"
                @test all(isfinite, result_profile.values.llh)
                # at a=1.0 (->index 2) we should have the best LLH since it's closest to observed[1] = 1.0
                [@test result_profile.values.llh[2] > result_profile.values.llh[i] for i in 1:3 if i != 2]
            end 
            
            @testset "profile 2D grid dimensions" begin
                fwd(a, b, c) = MvNormal([a, b, c], I(3))
                likelihood = likelihoodof(splat(fwd), [1.0, 0.0, 0.5])
                priors = (a=Uniform(-3.0, 3.0), b=Uniform(-3.0, 3.0), c=Uniform(-2.0, 2.0))
                params = (a=0.0, b=0.0, c=0.0)
                
                # fix a,b to gridpoints -> profile over c 
                result = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3, :b => 7), params)

                @test size(result.values.llh) == (3, 7)
                @test length(result.axes.a) == 3
                @test length(result.axes.b) == 7
                @test size(result.values.c) == (3, 7)   # nuisance :c stored at each grid point
                @test result.meta["task"] == "profile"
                @test all(isfinite, result.values.llh)
                # best LLH at a=0.0 (index 2), b=0.0 (index 4) since those are closest to observed[1]=1.0 and observed[2]=0.0, respectively 
                [@test result.values.llh[2, 4] > result.values.llh[i, j] for i in 1:3 for j in 1:7 if (i, j) != (2, 4)]
            end

            @testset "profile with cached dir" begin 
                fwd(a, b) = MvNormal([a, b], I(2))
                likelihood = likelihoodof(splat(fwd), [1.0, 0.0])
                priors = (a=Uniform(-3.0, 3.0), b=Uniform(-2.0, 2.0))
                params = (a=0.0, b=0.0)

                mktempdir() do cache_dir
                    # "fix" only a to gridpoints -> can still optimize over b as a nuisance param -> profile scans over a but optimizes over b at each a value
                    result_profile = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3), params, cache_dir=cache_dir)
                    @test length(readdir(cache_dir)) == 3 # 3 cache files should be created for the 3 grid points in :a
                    result_profile2 = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3), params, cache_dir=cache_dir)
                    @test length(readdir(cache_dir)) == 3 # no new cache file should be created since inputs are the same   
                    new_params = (a=1.0, b=1.0)
                    result_profile3 = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3), new_params, cache_dir=cache_dir)
                    @test length(readdir(cache_dir)) == 6 # 3 + 3 new cache files should be created since params changed
                end 
            end 
        end

        @testset "profile fallback to scan" begin
            #fallback if all non-scanned priors are numbers, here: prior(a) = 0.0
            fwd(a, b) = MvNormal([a, b], I(2))
            likelihood = likelihoodof(splat(fwd), [1.0, 2.0])
            # :a is a plain Number → all non-scan priors are Numbers → scan fallback
            priors_nt = (a=0.0, b=Uniform(-3.0, 3.0))
            params = (a=0.0, b=0.0)

            # only fix b to gridpoints, :a is already fixed to a number prior → no nuisance params to optimize over → profile falls back to scan over b only
            result = Newtrinos.profile(likelihood, priors_nt, OrderedDict(:b => 5), params)

            @test result isa Newtrinos.NewtrinosResult
            @test length(result.axes.b) == 5
            @test length(result.values.llh) == 5
            @test result.meta["task"] == "scan" #meta data also shows fallback to scan
            @test haskey(result.meta, "exec_time")
            @test all(result.values.a .== 0.0) # non-scanned param :a is stored as a Fill array at its fixed value
            @test all(isfinite, result.values.llh)
            # b is closest to 2 at gridpoint b=1.5 (->index 4) -> at b=1.5 we should have the best LLH
            [@test result.values.llh[4] > result.values.llh[i] for i in 1:5 if i != 4]  

            
            #fallback if all params are scanned over -> no nuisance params to optimize over -> no profiling possible 
            fwd(a, b) = MvNormal([a, b], I(2))
            likelihood = likelihoodof(splat(fwd), [1.0, 0.0])
            priors = (a=Uniform(-3.0, 3.0), b=Uniform(-2.0, 2.0))
            params = (a=0.0, b=0.0)

            # "fix" a and b to gridpoints -> cannot optimize over any nuisance params -> profile falls back to scan over both a and b 
            result = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3, :b => 5), params)
            @test result isa Newtrinos.NewtrinosResult
            @test length(result.axes.a) == 3
            @test length(result.axes.b) == 5
            @test size(result.values.llh) == (3, 5)
            @test result.meta["task"] == "scan" 
            @test all(isfinite, result.values.llh)
            # b is closest to 0.0  at gridpoint b=0.0 (->index 3) and  a is closest to 1.0 at gridpoint a=1.0 (->index 2) 
            # -> at (a=1.0, b=0.0) , i.e.llh[2, 3], we should have the best LLH
            [@test result.values.llh[2, 3] > result.values.llh[i, j] for i in 1:3 for j in 1:5 if (i, j) != (2, 3)]  
        end
    end

    @testset "bestfit" begin

        @testset "bestfit 1D" begin
            axes  = (a=[1.0, 2.0, 3.0],)
            vals  = (llh = [-10.0, -5.0, -8.0], 
                    log_posterior = [-10.0, -5.0, -8.0],
                    b = [0.1, 0.2, 0.3],
            )
            result = Newtrinos.NewtrinosResult(axes=axes, values=vals)
            bf = Newtrinos.bestfit(result)

            # picks the maximum log_posterior index
            @test bf.log_posterior == -5.0
            @test bf.llh           == -5.0
            @test bf.a == 2.0
            @test bf.b == 0.2 
        end

        @testset "bestfit 2D" begin
            a_axis = [1.0, 2.0, 3.0]
            b_axis = [10.0, 20.0]
            lp = [-9.0  -1.0;   # best is at (1,2): a=1.0, b=20.0
                -3.0  -6.0;
                -7.0  -8.0]
            result = Newtrinos.NewtrinosResult(axes = (a=a_axis, b=b_axis), values = (log_posterior=lp, llh=lp))
            bf = Newtrinos.bestfit(result)
            
            @test bf.log_posterior == -1.0
            @test bf.a == 1.0    # row 1
            @test bf.b == 20.0   # col 2
        end

        @testset "bestfit with single point" begin
            result = Newtrinos.NewtrinosResult(axes = (a=[5.0],), values = (log_posterior=[-3.0], llh=[-3.0]))
            bf = Newtrinos.bestfit(result)

            @test bf.a             == 5.0
            @test bf.log_posterior == -3.0
        end

        @testset "bestfit ties: returns first maximum" begin
            result = Newtrinos.NewtrinosResult(axes = (a=[1.0, 2.0, 3.0],), values = (log_posterior=[-5.0, -5.0, -8.0], llh=[-5.0, -5.0, -8.0]))
            bf = Newtrinos.bestfit(result)

            # argmax returns the first occurrence on ties
            @test bf.a == 1.0
        end

        @testset "bestfit with results from profiling" begin
            fwd(a, b, c) = MvNormal([a, b, c], I(3))
            likelihood = likelihoodof(splat(fwd), [1.0, 0.0, 0.5])
            priors = (a=Uniform(-3.0, 3.0), b=Uniform(-3.0, 3.0), c=Uniform(-2.0, 2.0))
            params = (a=0.0, b=0.0, c=0.0)

            result = Newtrinos.profile(likelihood, priors, OrderedDict(:a => 3, :b => 7), params)
            bf = Newtrinos.bestfit(result)
            @test result.values.log_posterior[2, 4] == bf.log_posterior
            @test result.values.llh[2, 4] == bf.llh
            @test result.axes.a[2] == bf.a
            @test result.axes.b[4] == bf.b
        end
    end

end
