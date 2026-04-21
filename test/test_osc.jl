using LinearAlgebra
using StaticArrays

@testset "Oscillation Physics" begin

    @testset "Configuration" begin
        # Default configuration
        osc = Newtrinos.osc.configure()
        @test osc isa Newtrinos.osc.Osc
        @test osc.cfg isa Newtrinos.osc.OscillationConfig
        @test osc.cfg.flavour isa Newtrinos.osc.ThreeFlavour
        @test osc.cfg.interaction isa Newtrinos.osc.Vacuum
        @test osc.cfg.propagation isa Newtrinos.osc.Basic
        @test osc.cfg.states isa Newtrinos.osc.All

        # Custom configuration
        cfg = Newtrinos.osc.OscillationConfig(
            flavour=Newtrinos.osc.ThreeFlavour(ordering=:IO),
            interaction=Newtrinos.osc.SI(),
            propagation=Newtrinos.osc.Damping(σₑ=0.05),
        )
        osc_io = Newtrinos.osc.configure(cfg)
        @test osc_io.cfg.flavour.ordering == :IO
        @test osc_io.params.Δm²₃₁ < 0  # inverted ordering
    end

    @testset "Parameters and Priors" begin
        # ThreeFlavour NO
        osc = Newtrinos.osc.configure()
        @test haskey(osc.params, :θ₁₂)
        @test haskey(osc.params, :θ₁₃)
        @test haskey(osc.params, :θ₂₃)
        @test haskey(osc.params, :δCP)
        @test haskey(osc.params, :Δm²₂₁)
        @test haskey(osc.params, :Δm²₃₁)
        @test osc.params.Δm²₃₁ > 0  # normal ordering

        # Priors have same keys as params
        @test keys(osc.priors) == keys(osc.params)

        # Sterile has extra parameters
        cfg_sterile = Newtrinos.osc.OscillationConfig(flavour=Newtrinos.osc.Sterile())
        osc_sterile = Newtrinos.osc.configure(cfg_sterile)
        @test haskey(osc_sterile.params, :Δm²₄₁)
        @test haskey(osc_sterile.params, :θ₁₄)
        @test haskey(osc_sterile.params, :θ₂₄)
        @test haskey(osc_sterile.params, :θ₃₄)

        # ADD has extra parameters
        cfg_add = Newtrinos.osc.OscillationConfig(flavour=Newtrinos.osc.ADD())
        osc_add = Newtrinos.osc.configure(cfg_add)
        @test haskey(osc_add.params, :m₀)
        @test haskey(osc_add.params, :ADD_radius)
    end

    @testset "PMNS Matrix" begin
        osc = Newtrinos.osc.configure()
        U = Newtrinos.osc.get_PMNS(osc.params)

        # PMNS should be 3x3
        @test size(U) == (3, 3)

        # PMNS should be unitary: U * U' ≈ I
        @test U * U' ≈ SMatrix{3,3}(I) atol=1e-12

        # U' * U ≈ I
        @test U' * U ≈ SMatrix{3,3}(I) atol=1e-12

        # With zero mixing angles, PMNS should be identity
        zero_params = (θ₁₂=0.0, θ₁₃=0.0, θ₂₃=0.0, δCP=0.0, Δm²₂₁=7.53e-5, Δm²₃₁=2.5e-3)
        U_zero = Newtrinos.osc.get_PMNS(zero_params)
        @test U_zero ≈ SMatrix{3,3}(I) atol=1e-14
    end

    @testset "Matrices function" begin
        osc = Newtrinos.osc.configure()
        U, h = osc.matrices(osc.params)

        # Eigenvalues should be ordered
        @test h[1] == 0.0  # first eigenvalue is zero (baseline subtracted)
        @test h[2] ≈ osc.params.Δm²₂₁
        @test h[3] ≈ osc.params.Δm²₃₁

        # U should still be unitary
        @test U * U' ≈ SMatrix{3,3}(I) atol=1e-12
    end

    @testset "Vacuum Oscillation Probabilities" begin
        osc = Newtrinos.osc.configure()

        E = [1.0, 5.0, 10.0]   # GeV
        L = [1000.0, 5000.0]    # km

        P = osc.osc_prob(E, L, osc.params)
        # P[n_E, n_L, in, out]: P[i,j,α,β] = P(να → νβ)

        @test size(P) == (length(E), length(L), 3, 3)

        # Probability conservation: sum over output flavours = 1
        for i in 1:length(E), j in 1:length(L), k in 1:3
            @test sum(P[i, j, k, :]) ≈ 1.0 atol=1e-10
        end

        # All probabilities should be between 0 and 1
        @test all(P .>= -1e-10)
        @test all(P .<= 1.0 + 1e-10)

        # At L=0, survival probability should be 1 (no oscillation)
        P_zero = osc.osc_prob(E, [0.0], osc.params)
        for i in 1:length(E)
            @test P_zero[i, 1, :, :] ≈ I(3) atol=1e-10
        end

        # With zero mass splittings, no oscillation
        no_osc_params = merge(osc.params, (Δm²₂₁=0.0, Δm²₃₁=0.0))
        P_no_osc = osc.osc_prob(E, L, no_osc_params)
        for i in 1:length(E), j in 1:length(L)
            @test P_no_osc[i, j, :, :] ≈ I(3) atol=1e-10
        end

        # Anti-neutrino probabilities should also conserve probability
        P_anti = osc.osc_prob(E, L, osc.params; anti=true)
        for i in 1:length(E), j in 1:length(L), k in 1:3
            @test sum(P_anti[i, j, k, :]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "CPT symmetry" begin
        osc = Newtrinos.osc.configure()
        E = [2.0, 5.0]
        L = [1000.0]

        P_nu = osc.osc_prob(E, L, osc.params; anti=false)
        P_nubar = osc.osc_prob(E, L, osc.params; anti=true)

        # CPT: P(να→νβ) = P(ν̄β→ν̄α) (transpose of anti)
        for i in 1:length(E), j in 1:length(L)
            @test P_nu[i, j, :, :] ≈ P_nubar[i, j, :, :]' atol=1e-10
        end
    end

    @testset "Antineutrino Matter Oscillations" begin
        # Test with SI interaction and non-zero δCP to catch double-conjugation bugs
        cfg = Newtrinos.osc.OscillationConfig(interaction=Newtrinos.osc.SI())
        osc = Newtrinos.osc.configure(cfg)
        earth = Newtrinos.earth_layers.configure()
        layers = earth.compute_layers()
        coszen = [-1.0, -0.5, -0.2]
        paths = earth.compute_paths(coszen, layers)

        E = [1.0, 5.0, 10.0]
        params = osc.params

        # Antineutrino probability conservation with matter effects
        P_anti = osc.osc_prob(E, paths, layers, params; anti=true)
        @test size(P_anti) == (length(E), length(coszen), 3, 3)
        for i in 1:length(E), j in 1:length(coszen), k in 1:3
            @test sum(P_anti[i, j, k, :]) ≈ 1.0 atol=1e-10
        end
        @test all(P_anti .>= -1e-10)
        @test all(P_anti .<= 1.0 + 1e-10)

        # Neutrino probability conservation with matter effects
        P_nu = osc.osc_prob(E, paths, layers, params; anti=false)
        for i in 1:length(E), j in 1:length(coszen), k in 1:3
            @test sum(P_nu[i, j, k, :]) ≈ 1.0 atol=1e-10
        end

        # With non-zero δCP, neutrino and antineutrino should differ (CP violation)
        @test !isapprox(P_nu, P_anti, atol=1e-6)
    end

    @testset "Select function" begin
        osc = Newtrinos.osc.configure()
        U, h = osc.matrices(osc.params)

        # All selector returns everything unchanged
        U_sel, h_sel, rest = Newtrinos.osc.select(U, h, Newtrinos.osc.All())
        @test U_sel == U
        @test h_sel == h
        @test rest ≈ 0.0

        # Cut with high cutoff returns everything
        U_sel, h_sel, rest = Newtrinos.osc.select(U, h, Newtrinos.osc.Cut(cutoff=1e10))
        @test length(h_sel) == 3
    end

    @testset "Damping propagation" begin
        cfg = Newtrinos.osc.OscillationConfig(propagation=Newtrinos.osc.Damping(σₑ=0.1))
        osc = Newtrinos.osc.configure(cfg)
        E = [1.0, 5.0]
        L = [1000.0]
        P = osc.osc_prob(E, L, osc.params)

        # Probability conservation still holds
        for i in 1:length(E), j in 1:length(L), k in 1:3
            @test sum(P[i, j, k, :]) ≈ 1.0 atol=1e-6
        end

        # All probabilities in valid range
        @test all(P .>= -1e-6)
        @test all(P .<= 1.0 + 1e-6)
    end

    @testset "Decoherent propagation" begin
        cfg = Newtrinos.osc.OscillationConfig(propagation=Newtrinos.osc.Decoherent(σₑ=0.1))
        osc = Newtrinos.osc.configure(cfg)
        E = [1.0, 5.0]
        L = [1000.0]
        P = osc.osc_prob(E, L, osc.params)

        # Probability conservation
        for i in 1:length(E), j in 1:length(L), k in 1:3
            @test sum(P[i, j, k, :]) ≈ 1.0 atol=1e-6
        end

        @test all(P .>= -1e-6)
        @test all(P .<= 1.0 + 1e-6)
    end
end
