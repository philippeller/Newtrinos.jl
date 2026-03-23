using ForwardDiff
using LinearAlgebra

@testset "ForwardDiff Compatibility" begin

    @testset "PMNS matrix with Dual numbers" begin
        osc = Newtrinos.osc.configure()

        # Compute PMNS with Dual numbers by differentiating
        function pmns_element(x)
            p = merge(osc.params, (θ₁₂=x[1],))
            U = Newtrinos.osc.get_PMNS(p)
            real(U[1,1])^2  # |Ue1|^2
        end

        x0 = [osc.params.θ₁₂]
        g = ForwardDiff.gradient(pmns_element, x0)
        @test length(g) == 1
        @test isfinite(g[1])
        @test g[1] != 0.0  # should have non-zero derivative
    end

    @testset "Oscillation probability gradient" begin
        osc = Newtrinos.osc.configure()
        E = [2.0]   # GeV
        L = [300.0]  # km

        # Gradient of P(νμ→νμ) survival probability w.r.t. θ₂₃
        function survival_prob(x)
            p = merge(osc.params, (θ₂₃=x[1],))
            P = osc.osc_prob(E, L, p)
            P[1, 1, 2, 2]  # P(νμ→νμ) at first E, first L
        end

        x0 = [osc.params.θ₂₃]
        g = ForwardDiff.gradient(survival_prob, x0)
        @test length(g) == 1
        @test isfinite(g[1])

        # Verify with finite differences
        eps = 1e-6
        fd = (survival_prob([x0[1] + eps]) - survival_prob([x0[1] - eps])) / (2 * eps)
        @test g[1] ≈ fd atol=1e-4
    end

    @testset "Gradient w.r.t. mass splitting" begin
        osc = Newtrinos.osc.configure()
        E = [2.0]
        L = [500.0]

        function prob_vs_dm(x)
            p = merge(osc.params, (Δm²₃₁=x[1],))
            P = osc.osc_prob(E, L, p)
            P[1, 1, 1, 2]  # P(νe→νμ) at first E, first L
        end

        x0 = [osc.params.Δm²₃₁]
        g = ForwardDiff.gradient(prob_vs_dm, x0)
        @test isfinite(g[1])
        @test g[1] != 0.0

        # Verify with finite differences
        eps = 1e-9
        fd = (prob_vs_dm([x0[1] + eps]) - prob_vs_dm([x0[1] - eps])) / (2 * eps)
        @test g[1] ≈ fd atol=1e-3
    end

    @testset "NamedTuple gradient wrapper" begin
        osc = Newtrinos.osc.configure()
        E = [3.0]
        L = [1000.0]

        function neg_loglik(p)
            P = osc.osc_prob(E, L, p)
            -log(max(P[1, 1, 2, 2], 1e-30))  # -log P(νμ→νμ)
        end

        g = ForwardDiff.gradient(neg_loglik, osc.params)
        @test g isa NamedTuple
        @test keys(g) == keys(osc.params)
        @test all(isfinite.(values(g)))
    end

    @testset "xsec type stability with Dual" begin
        xs = Newtrinos.xsec.configure()

        function xsec_func(x)
            p = (nc_norm=x[1], nutau_cc_norm=x[2])
            # Should return Dual type for all branches
            a = xs.scale(:nue, :NC, p)
            b = xs.scale(:nutau, :CC, p)
            c = xs.scale(:numu, :CC, p)
            a + b + c
        end

        x0 = [1.0, 1.0]
        g = ForwardDiff.gradient(xsec_func, x0)
        @test g[1] ≈ 1.0  # d/d(nc_norm) of nc_norm + nutau + 1 = 1
        @test g[2] ≈ 1.0  # d/d(nutau_cc_norm) = 1
    end
end
