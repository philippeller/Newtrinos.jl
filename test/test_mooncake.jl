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

    # Helper: build rule, compute gradient, check finite, check value, spot-check against FD
    function check_mooncake_gradient(neg_ll, x0, ref_val; rtol_val=1e-6, rtol_grad=1e-4)
        rule = Mooncake.build_rrule(neg_ll, x0)
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, neg_ll, x0)

        @test isfinite(val)
        @test all(isfinite, dx)
        @test val ≈ ref_val rtol=rtol_val

        eps_fd = 1e-6
        for i in [1, length(x0)÷2, length(x0)]
            scale = max(1.0, abs(x0[i]))
            x_plus = copy(x0); x_plus[i] += eps_fd * scale
            x_minus = copy(x0); x_minus[i] -= eps_fd * scale
            fd = (neg_ll(x_plus) - neg_ll(x_minus)) / (2eps_fd * scale)
            @test dx[i] ≈ fd rtol=rtol_grad
        end
    end

    # Configure all experiments up front (sequential, I/O heavy)
    dc = Newtrinos.deepcore.configure()
    db = Newtrinos.dayabay.configure()
    kl = Newtrinos.kamland.configure()
    mi = Newtrinos.minos.configure()
    or = Newtrinos.orca.configure()
    sk = Newtrinos.super_k.configure()
    cc = Newtrinos.coherent_csi.configure()
    cl = Newtrinos.coherent_lAr.configure()
    ju = Newtrinos.juno.configure()
    ta = Newtrinos.tao.configure()
    ic = Newtrinos.ic_upgrade.configure()

    # Build neg_ll closures for each experiment
    function make_neg_ll(exp_tuple)
        params = Newtrinos.get_params(exp_tuple)
        likelihood = Newtrinos.generate_likelihood(exp_tuple)
        neg_ll = let params=params, likelihood=likelihood
            x -> begin
                p = NamedTuple{keys(params)}(Tuple(x))
                Float64(-log(likelihood(p)))
            end
        end
        x0 = collect(Float64, values(params))
        return neg_ll, x0
    end

    function make_neg_ll_asimov(exp, exp_sym)
        exp_tuple = NamedTuple{(exp_sym,)}((exp,))
        params = Newtrinos.get_params(exp_tuple)
        asimov = Newtrinos.generate_asimov_data(exp, params)
        observed = NamedTuple{(exp_sym,)}((asimov,))
        likelihood = Newtrinos.generate_likelihood(exp_tuple, observed)
        neg_ll = let params=params, likelihood=likelihood
            x -> begin
                p = NamedTuple{keys(params)}(Tuple(x))
                Float64(-log(likelihood(p)))
            end
        end
        x0 = collect(Float64, values(params))
        return neg_ll, x0
    end

    # Prepare all closures
    ll_dc, x0_dc = make_neg_ll((deepcore=dc,))
    ll_db, x0_db = make_neg_ll((dayabay=db,))
    ll_kl, x0_kl = make_neg_ll((kamland=kl,))
    ll_mi, x0_mi = make_neg_ll((minos=mi,))
    ll_or, x0_or = make_neg_ll((orca=or,))
    ll_sk, x0_sk = make_neg_ll((super_k=sk,))
    ll_cc, x0_cc = make_neg_ll((coherent_csi=cc,))
    ll_cl, x0_cl = make_neg_ll((coherent_lAr=cl,))
    ll_ju, x0_ju = make_neg_ll_asimov(ju, :juno)
    ll_ta, x0_ta = make_neg_ll_asimov(ta, :tao)
    ll_ic, x0_ic = make_neg_ll_asimov(ic, :ic_upgrade)

    # Build all Mooncake rules in parallel using tasks
    tasks = Dict{String, Any}()
    @sync begin
        tasks["deepcore"] = Threads.@spawn Mooncake.build_rrule(ll_dc, x0_dc)
        tasks["dayabay"] = Threads.@spawn Mooncake.build_rrule(ll_db, x0_db)
        tasks["kamland"] = Threads.@spawn Mooncake.build_rrule(ll_kl, x0_kl)
        tasks["minos"] = Threads.@spawn Mooncake.build_rrule(ll_mi, x0_mi)
        tasks["orca"] = Threads.@spawn Mooncake.build_rrule(ll_or, x0_or)
        tasks["super_k"] = Threads.@spawn Mooncake.build_rrule(ll_sk, x0_sk)
        tasks["coherent_csi"] = Threads.@spawn Mooncake.build_rrule(ll_cc, x0_cc)
        tasks["coherent_lAr"] = Threads.@spawn Mooncake.build_rrule(ll_cl, x0_cl)
        tasks["juno"] = Threads.@spawn Mooncake.build_rrule(ll_ju, x0_ju)
        tasks["tao"] = Threads.@spawn Mooncake.build_rrule(ll_ta, x0_ta)
        tasks["ic_upgrade"] = Threads.@spawn Mooncake.build_rrule(ll_ic, x0_ic)
    end

    # Now run gradient checks (these are fast once rules are built)
    @testset "DeepCore likelihood gradient" begin
        rule = fetch(tasks["deepcore"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_dc, x0_dc)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 950.706345 rtol=1e-6
    end

    @testset "Daya Bay likelihood gradient" begin
        rule = fetch(tasks["dayabay"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_db, x0_db)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 168.900033 rtol=1e-6
    end

    @testset "KamLAND likelihood gradient" begin
        rule = fetch(tasks["kamland"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_kl, x0_kl)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 63.111404 rtol=1e-6
    end

    @testset "MINOS likelihood gradient" begin
        rule = fetch(tasks["minos"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_mi, x0_mi)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 268.336328 rtol=1e-6
    end

    @testset "ORCA likelihood gradient" begin
        rule = fetch(tasks["orca"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_or, x0_or)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 1164.250608 rtol=1e-6
    end

    @testset "Super-K likelihood gradient" begin
        rule = fetch(tasks["super_k"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_sk, x0_sk)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 3706.336908 rtol=1e-6
    end

    @testset "COHERENT CsI likelihood gradient" begin
        rule = fetch(tasks["coherent_csi"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_cc, x0_cc)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 574.341603 rtol=1e-6
    end

    @testset "COHERENT LAr likelihood gradient" begin
        rule = fetch(tasks["coherent_lAr"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_cl, x0_cl)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 1754.994694 rtol=1e-6
    end

    @testset "JUNO likelihood gradient (Asimov)" begin
        rule = fetch(tasks["juno"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_ju, x0_ju)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 1322.344202 rtol=1e-5
    end

    @testset "TAO likelihood gradient (Asimov)" begin
        rule = fetch(tasks["tao"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_ta, x0_ta)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 1851.429190 rtol=1e-5
    end

    @testset "IC Upgrade likelihood gradient (Asimov)" begin
        rule = fetch(tasks["ic_upgrade"])
        val, (_, dx) = Mooncake.value_and_gradient!!(rule, ll_ic, x0_ic)
        @test isfinite(val) && all(isfinite, dx)
        @test val ≈ 1502.541121 rtol=1e-5
    end
end
