using Mooncake

@testset "Mooncake Reverse-Mode AD" begin

    @testset "Vacuum oscillation gradient" begin
        osc = Newtrinos.osc.configure()
        E = [2.0]
        L = [300.0]

        function vacuum_prob(x)
            p = NamedTuple{keys(osc.params)}(Tuple(x))
            P = osc.osc_prob(E, L, p)
            P[1, 1, 2, 2]  # P(νμ→νμ)
        end

        x0 = collect(Float64, values(osc.params))
        rule = Mooncake.build_rrule(vacuum_prob, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, vacuum_prob, x0)

        @test isfinite(val)
        @test all(isfinite, dx)

        # Verify against finite differences
        eps_fd = 1e-7
        for i in eachindex(x0)
            x_plus = copy(x0); x_plus[i] += eps_fd
            x_minus = copy(x0); x_minus[i] -= eps_fd
            fd = (vacuum_prob(x_plus) - vacuum_prob(x_minus)) / (2eps_fd)
            @test dx[i] ≈ fd atol=1e-4
        end
    end

    @testset "Matter oscillation gradient (SI)" begin
        cfg = Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI())
        osc_si = Newtrinos.osc.configure(cfg)
        el = Newtrinos.earth_layers.configure()
        layers = el.compute_layers()
        paths = el.compute_paths([-0.5], layers)
        E = [5.0]

        function matter_prob(x)
            p = NamedTuple{keys(osc_si.params)}(Tuple(x))
            P = osc_si.osc_prob(E, paths, layers, p)
            P[1, 1, 2, 2]
        end

        x0 = collect(Float64, values(osc_si.params))
        rule = Mooncake.build_rrule(matter_prob, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, matter_prob, x0)

        @test isfinite(val)
        @test all(isfinite, dx)

        # Verify against finite differences
        eps_fd = 1e-7
        for i in eachindex(x0)
            x_plus = copy(x0); x_plus[i] += eps_fd
            x_minus = copy(x0); x_minus[i] -= eps_fd
            fd = (matter_prob(x_plus) - matter_prob(x_minus)) / (2eps_fd)
            @test dx[i] ≈ fd rtol=1e-4
        end
    end

    @testset "Sterile matter oscillation gradient (DefaultEigen)" begin
        cfg = Newtrinos.osc.OscillationConfig(
            flavour=Newtrinos.osc.Sterile(),
            interaction=Newtrinos.osc.SI(),
            eigen_method=Newtrinos.osc.DefaultEigen()
        )
        osc_st = Newtrinos.osc.configure(cfg)
        el = Newtrinos.earth_layers.configure()
        layers = el.compute_layers()
        paths = el.compute_paths([-0.5], layers)
        E = [5.0]

        function sterile_matter_prob(x)
            p = NamedTuple{keys(osc_st.params)}(Tuple(x))
            P = osc_st.osc_prob(E, paths, layers, p)
            P[1, 1, 2, 2]
        end

        x0 = collect(Float64, values(osc_st.params))
        rule = Mooncake.build_rrule(sterile_matter_prob, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, sterile_matter_prob, x0)

        @test isfinite(val)
        @test all(isfinite, dx)

        # Verify against finite differences
        eps_fd = 1e-7
        for i in eachindex(x0)
            x_plus = copy(x0); x_plus[i] += eps_fd
            x_minus = copy(x0); x_minus[i] -= eps_fd
            fd = (sterile_matter_prob(x_plus) - sterile_matter_prob(x_minus)) / (2eps_fd)
            @test dx[i] ≈ fd rtol=1e-4
        end
    end

    @testset "DeepCore full likelihood gradient" begin
        dc = Newtrinos.deepcore.configure()
        params = Newtrinos.get_params((deepcore=dc,))
        likelihood = Newtrinos.generate_likelihood((deepcore=dc,))

        function neg_ll_dc(x)
            p = NamedTuple{keys(params)}(Tuple(x))
            Float64(-log(likelihood(p)))
        end

        x0 = collect(Float64, values(params))
        rule = Mooncake.build_rrule(neg_ll_dc, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, neg_ll_dc, x0)

        @test isfinite(val)
        @test all(isfinite, dx)
        @test val ≈ 950.706345 rtol=1e-6

        # Spot check a few gradients against finite differences
        eps_fd = 1e-6
        for i in [1, length(x0)÷2, length(x0)]  # first, middle, last param
            scale = max(1.0, abs(x0[i]))
            x_plus = copy(x0); x_plus[i] += eps_fd * scale
            x_minus = copy(x0); x_minus[i] -= eps_fd * scale
            fd = (neg_ll_dc(x_plus) - neg_ll_dc(x_minus)) / (2eps_fd * scale)
            @test dx[i] ≈ fd rtol=1e-4
        end
    end

    # Helper to test a full experiment likelihood gradient
    function test_experiment_gradient(name, exp_tuple, ref_val)
        params = Newtrinos.get_params(exp_tuple)
        likelihood = Newtrinos.generate_likelihood(exp_tuple)

        neg_ll = let params=params, likelihood=likelihood
            x -> begin
                p = NamedTuple{keys(params)}(Tuple(x))
                Float64(-log(likelihood(p)))
            end
        end

        x0 = collect(Float64, values(params))
        rule = Mooncake.build_rrule(neg_ll, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, neg_ll, x0)

        @test isfinite(val)
        @test all(isfinite, dx)
        @test val ≈ ref_val rtol=1e-6

        # Spot check gradients against finite differences
        eps_fd = 1e-6
        for i in [1, length(x0)÷2, length(x0)]
            scale = max(1.0, abs(x0[i]))
            x_plus = copy(x0); x_plus[i] += eps_fd * scale
            x_minus = copy(x0); x_minus[i] -= eps_fd * scale
            fd = (neg_ll(x_plus) - neg_ll(x_minus)) / (2eps_fd * scale)
            @test dx[i] ≈ fd rtol=1e-4
        end
    end

    @testset "Daya Bay likelihood gradient" begin
        test_experiment_gradient("dayabay",
            (dayabay=Newtrinos.dayabay.configure(),), 168.900033)
    end

    @testset "KamLAND likelihood gradient" begin
        test_experiment_gradient("kamland",
            (kamland=Newtrinos.kamland.configure(),), 63.111404)
    end

    @testset "MINOS likelihood gradient" begin
        test_experiment_gradient("minos",
            (minos=Newtrinos.minos.configure(),), 268.336328)
    end

    @testset "ORCA likelihood gradient" begin
        test_experiment_gradient("orca",
            (orca=Newtrinos.orca.configure(),), 1164.250608)
    end

    @testset "Super-K likelihood gradient" begin
        test_experiment_gradient("super_k",
            (super_k=Newtrinos.super_k.configure(),), 3706.336908)
    end

    @testset "COHERENT CsI likelihood gradient" begin
        test_experiment_gradient("coherent_csi",
            (coherent_csi=Newtrinos.coherent_csi.configure(),), 574.341603)
    end

    @testset "COHERENT LAr likelihood gradient" begin
        test_experiment_gradient("coherent_lAr",
            (coherent_lAr=Newtrinos.coherent_lAr.configure(),), 1754.994694)
    end
end
