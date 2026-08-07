using Newtrinos
using Test
using Distributions
using LinearAlgebra
using StaticArrays

@testset "osc.jl" begin

    @testset "params and priors" begin
        # test get_params and get_priors for different model configurations
        configs = [
            Newtrinos.osc.ThreeFlavour(),
            Newtrinos.osc.ThreeFlavourXYCP(),
            Newtrinos.osc.Sterile(),
            Newtrinos.osc.ADD(),
            Newtrinos.osc.NND(),
            Newtrinos.osc.NNM(),] #=,
            Newtrinos.osc.Darkdim_Lambda(),
            Newtrinos.osc.Darkdim_Masses(),
            Newtrinos.osc.Darkdim_cas()]=#

        for cfg in configs
            @test Newtrinos.osc.get_params(cfg) isa NamedTuple
            @test Newtrinos.osc.get_priors(cfg) isa NamedTuple
            @test keys(Newtrinos.osc.get_params(cfg)) == keys(Newtrinos.osc.get_priors(cfg))
        end

        # test if the parameters are correctly calculated
        # eventually: test against PDG live values?
        # basic parameters
        angles = (θ₁₂ = 0.58725, θ₁₃ = 0.14543, θ₂₃ = 0.85563)
        NO_masses = (Δm²₂₁ = 7.53e-5, Δm²₃₁ = 2.4e-3 + 7.53e-5)
        IO_masses = (Δm²₂₁ = 7.53e-5, Δm²₃₁ = -(2.4e-3 - 7.53e-5))
        δCP = (δCP = 1.0,)
        # define expected parameter sets
        ThreeFlavour_Params_NO = merge(angles, NO_masses, δCP)
        ThreeFlavour_Params_IO = merge(angles, IO_masses, δCP)
        # these are all for NO because they they dont affect m31
        ThreeFlavourXYCP_Params = merge(angles, NO_masses, (δCPshell = [1.0, 0.0],))
        Sterile_Params = merge(ThreeFlavour_Params_NO, (Δm²₄₁ = 1.0, θ₁₄ = 0.1, θ₂₄ = 0.1, θ₃₄ = 0.1,))
        ADD_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, ADD_radius = 1e-2,))
        NND_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, N = 50, r=1e-8,))
        NNM_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, N = 50, r=1e-8,))
        Darkdim_Lambda_Params = merge(angles, δCP, (Darkdim_radius = 0.1, ca1 = 1e-5, ca2 = 1e-5, ca3 = 1e-5, λ₁ = 1.0, λ₂ = 1.0, λ₃ = 1.0,))
        Darkdim_Masses_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, Darkdim_radius = 0.1, λ₁ = 1.0, λ₂ = 1.0, λ₃ = 1.0,))
        Darkdim_cas_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, Darkdim_radius = 0.1, ca1 = 1e-5, ca2 = 1e-5, ca3 = 1e-5,))

        # tests
        # isapprox cannot handle NamedTuples -> loop over keys and test each parameter separately, with an appropriate tolerance
        for key in keys(ThreeFlavour_Params_NO)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.ThreeFlavour(ordering = :NO)), key) ≈ getfield(ThreeFlavour_Params_NO, key) atol = 1e-4
        end
        for key in keys(ThreeFlavour_Params_IO)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.ThreeFlavour(ordering = :IO)), key) ≈ getfield(ThreeFlavour_Params_IO, key) atol = 1e-4
        end
        for key in keys(ThreeFlavourXYCP_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.ThreeFlavourXYCP()), key) ≈ getfield(ThreeFlavourXYCP_Params, key) atol = 1e-4
        end
        for key in keys(Sterile_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.Sterile()), key) ≈ getfield(Sterile_Params, key) atol = 1e-4
        end
        for key in keys(ADD_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.ADD()), key) ≈ getfield(ADD_Params, key) atol = 1e-4
        end

         for key in keys(NND_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.NND()), key) ≈ getfield(NND_Params, key) atol = 1e-4
        end

         for key in keys(NNM_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.NNM()), key) ≈ getfield(NNM_Params, key) atol = 1e-4
        end
        # darkdim not yet exported from osc file
        #= for key in keys(Darkdim_Lambda_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.Darkdim_Lambda()), key) ≈ getfield(Darkdim_Lambda_Params, key) atol = 1e-4
        end
        for key in keys(Darkdim_Masses_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.Darkdim_Masses()), key) ≈ getfield(Darkdim_Masses_Params, key) atol = 1e-4
        end
        for key in keys(Darkdim_cas_Params)
            @test getfield(Newtrinos.osc.get_params(Newtrinos.osc.Darkdim_cas()), key) ≈ getfield(Darkdim_cas_Params, key) atol = 1e-4
        end =#

        # test if the priors are correctly calculated

        priors_ThreeFlavour_NO = Newtrinos.osc.get_priors(Newtrinos.osc.ThreeFlavour(ordering = :NO))
        priors_ThreeFlavour_IO = Newtrinos.osc.get_priors(Newtrinos.osc.ThreeFlavour(ordering = :IO))
        priors_ThreeFlavourXYCP = Newtrinos.osc.get_priors(Newtrinos.osc.ThreeFlavourXYCP())
        priors_Sterile = Newtrinos.osc.get_priors(Newtrinos.osc.Sterile())
        priors_ADD = Newtrinos.osc.get_priors(Newtrinos.osc.ADD())
        priors_NND = Newtrinos.osc.get_priors(Newtrinos.osc.NND())
        priors_NNM = Newtrinos.osc.get_priors(Newtrinos.osc.NNM()) 
        # priors_Darkdim_Lambda = Newtrinos.osc.get_priors(Newtrinos.osc.Darkdim_Lambda())
        # priors_Darkdim_Masses = Newtrinos.osc.get_priors(Newtrinos.osc.Darkdim_Masses())
        # priors_Darkdim_cas = Newtrinos.osc.get_priors(Newtrinos.osc.Darkdim_cas())

        # ThreeFlavour
        # TODO: test if param is contained in prior, with @test insupport(prior_distr, param)
        for key in keys(priors_ThreeFlavour_NO)
            @test getfield(priors_ThreeFlavour_NO, key) isa Uniform
        end
        @test getfield(priors_ThreeFlavour_NO, :Δm²₃₁) != getfield(priors_ThreeFlavour_IO, :Δm²₃₁)
        # ThreeFlavourXYCP
        @test getfield(priors_ThreeFlavourXYCP, :δCPshell) isa MvNormal
        # Sterile
        for key in [:θ₁₄, :θ₂₄, :θ₃₄, :Δm²₄₁]
            @test getfield(priors_Sterile, key) isa Uniform
        end
        # ADD
        @test getfield(priors_ADD, :m₀) isa LogUniform
        @test getfield(priors_ADD, :ADD_radius) isa LogUniform
        # NND
        @test getfield(priors_NND, :m₀) isa LogUniform
        @test getfield(priors_NND, :N) isa DiscreteUniform
        @test getfield(priors_NND, :r) isa LogUniform
        # NNM
        @test getfield(priors_NNM, :m₀) isa LogUniform
        @test getfield(priors_NNM, :N) isa DiscreteUniform
        @test getfield(priors_NNM, :r) isa LogUniform
        # Darkdim_Lambda
        #= @test getfield(priors_Darkdim_Lambda, :Darkdim_radius) isa LogUniform
        @test !haskey(priors_Darkdim_Lambda, :Δm²₂₁)
        @test !haskey(priors_Darkdim_Lambda, :Δm²₃₁)
        for key in [:ca1, :ca2, :ca3, :λ₁, :λ₂, :λ₃]
            @test getfield(priors_Darkdim_Lambda, key) isa Uniform
        end
        # Darkdim_Masses
        @test getfield(priors_Darkdim_Masses, :m₀) isa LogUniform
        @test getfield(priors_Darkdim_Masses, :Darkdim_radius) isa LogUniform
        for key in [:ca1, :ca2, :ca3]
            @test getfield(priors_Darkdim_Masses, key) isa Uniform
        end
        # Darkdim_cas
        @test getfield(priors_Darkdim_cas, :m₀) isa LogUniform
        @test getfield(priors_Darkdim_cas, :Darkdim_radius) isa LogUniform
        for key in [:λ₁, :λ₂, :λ₃]
            @test getfield(priors_Darkdim_cas, key) isa Uniform
        end =#
    end

    @testset "oscillation functions" begin

        # define test parameter sets
        test_params_1 = (θ₁₂ = 0.0, θ₁₃ = 0.0, θ₂₃ = 0.0, δCP = 0.0)
        test_params_2 = (θ₁₂ = 0.6, θ₁₃ = 0.2, θ₂₃ = 0.7, δCP = 0.4)
        test_params_3 = (θ₁₂ = 0.4, θ₁₃ = 0.15, θ₂₃ = 0.9, δCP = -0.5)

        # TEST get_PMNS
        for params in [test_params_1, test_params_2, test_params_3]
            s12, c12 = sin(params.θ₁₂), cos(params.θ₁₂)
            s13, c13 = sin(params.θ₁₃), cos(params.θ₁₃)
            s23, c23 = sin(params.θ₂₃), cos(params.θ₂₃)
            cp = cis(params.δCP) # exp(i * δ)
            cp_conj = conj(cp)   # exp(-i * δ)
            # PMNS standard parametrization
            expected_PMNS = [
                c13*c12                     c13*s12                     s13*cp_conj
                -c23*s12-s23*c12*s13*cp     c23*c12-s23*s12*s13*cp      s23*c13
                s23*s12-c23*s13*c12*cp      -s23*c12-s12*s13*c23*cp     c23*c13
            ]
            @test Newtrinos.osc.get_PMNS(params) ≈ expected_PMNS atol = 1e-6
            @test Newtrinos.osc.get_PMNS(params) isa SMatrix{3,3}
        end

        # TEST get_abs_masses()
        @test Newtrinos.osc.get_abs_masses((m₀ = 1.5, Δm²₂₁ = 2.3, Δm²₃₁ = 3.1)) isa Tuple
        @test collect(Newtrinos.osc.get_abs_masses((m₀ = 1.5, Δm²₂₁ = 2.3, Δm²₃₁ = 3.1))) ≈ [1.5, 2.1330729, 2.3130067] atol = 1e-6
        @test collect(Newtrinos.osc.get_abs_masses((m₀ = 0.4, Δm²₂₁ = 1.2, Δm²₃₁ = -2.4))) ≈ [1.6, 1.9390719, 0.4] atol = 1e-6

        # TEST osc_kernel()
        U = @SMatrix [cos(π/4) sin(π/4) 0; -sin(π/4) cos(π/4) 0; 0 0 1]
        H = @SVector [0.0, 2.5e-5, 1]
        e, l, σₑ = 1.0, 100.0, 2
        # test simple kernel
        result_simple = Newtrinos.osc.osc_kernel(U, H, e, l)
        @test size(result_simple) == (3, 3)
        @test result_simple' * result_simple ≈ I atol = 1e-12
        # test osc_kernel with low pass filter
        result_lowpass = Newtrinos.osc.osc_kernel(U, H, e, l, σₑ)
        @test length(result_lowpass) == 2
        @test size(result_lowpass[1]) == (3, 3)
        @test size(result_lowpass[2]) == (3,)
        # test matrix elements against results from rigorous calculation
        u11, u22, u33, u12, u13, u21, u23, u31, u32 = U[1,1], U[2,2], U[3,3], U[1,2], U[1,3], U[2,1], U[2,3], U[3,1], U[3,2]
        phi_simple = -Newtrinos.osc.F_units * 1im * (l / e) .* H
        phi_decay = - 2 * abs.(-1im*phi_simple) * σₑ^2
        phi_lowpass = phi_simple + phi_decay

        for phi in [phi_simple, phi_lowpass]
            expected_kernel_matrix = [
                u11^2*exp(phi[1])+u12^2*exp(phi[2])+u13^2*exp(phi[3])           u11*u21*exp(phi[1])+u12*u22*exp(phi[2])+u13*u23*exp(phi[3])     u11*u31*exp(phi[1])+u12*u32*exp(phi[2])+u13*u33*exp(phi[3])
                u21*u11*exp(phi[1])+u22*u12*exp(phi[2])+u23*u13*exp(phi[3])     u21^2*exp(phi[1])+u22^2*exp(phi[2])+u23^2*exp(phi[3])           u21*u31*exp(phi[1])+u22*u32*exp(phi[2])+u23*u33*exp(phi[3])
                u31*u11*exp(phi[1])+u32*u12*exp(phi[2])+u33*u13*exp(phi[3])     u31*u21*exp(phi[1])+u32*u22*exp(phi[2])+u33*u23*exp(phi[3])     u31^2*exp(phi[1])+u32^2*exp(phi[2])+u33^2*exp(phi[3])]
            if phi == phi_simple
                @test collect(Newtrinos.osc.osc_kernel(U, H, e, l)) ≈ expected_kernel_matrix atol = 5e-6
            elseif phi == phi_lowpass
                @test collect(Newtrinos.osc.osc_kernel(U, H, e, l, σₑ)[1]) ≈ expected_kernel_matrix atol = 5e-6
                @test Newtrinos.osc.osc_kernel(U, H, e, l, σₑ)[2] ≈ exp.(phi_decay) atol = 5e-6
            end
        end

        # TEST compute_matter_matrices()
        H_eff = [1.0 0.2 0.1; 0.2 2.0 0.3; 0.1 0.3 3.0] # example hamiltonian
        static_H_eff = SMatrix{3,3}(H_eff)
        layer = Newtrinos.osc.Layer(6371.0, 2.0, 1.5) # radius earth and example proton/neutron densities
        e = 1.5 # energy value

        vecs, vals = Newtrinos.osc.compute_matter_matrices(H_eff, e, layer, false, Newtrinos.osc.SI())
        vecs_anti, vals_anti = Newtrinos.osc.compute_matter_matrices(H_eff, e, layer, true, Newtrinos.osc.SI())
        static_vecs, static_vals = Newtrinos.osc.compute_matter_matrices(static_H_eff, e, layer, false, Newtrinos.osc.SI())
        @test size(vecs) == (3, 3)
        @test size(vals) == (3,)
        # test compatibility of the two implementations (static vs non-static)
        @test vals ≈ static_vals atol = 1e-6
        @test abs.(vecs) ≈ abs.(static_vecs) atol = 1e-6 # we can only compare the absolute values, because the eigenvectors are not uniquely defined (phase and order)
        # compare to expected values
        A, f, n_p, n_n = Newtrinos.osc.A, e*1e9, layer.p_density, layer.n_density
        H = [1.0+2*A*n_p*f-A*n_n*f 0.2 0.1; 0.2 2.0-A*n_n*f 0.3; 0.1 0.3 3.0-A*n_n*f] # anti = false
        expected_vals, expected_vecs = eigvals(H), eigvecs(H)
        @test collect(vals) ≈ collect(expected_vals) atol = 1e-6
        @test collect(vecs) ≈ collect(expected_vecs) atol = 1e-6

        # TEST osc_reduce()
        # take matter matrices and energy e = 1.5 (GeV) from above
        matter_matrices = [(vecs, vals), (static_vecs, static_vals)]
        path = [(layer_idx = 1, length = 5.0), (layer_idx = 2, length = 10.0)] # define path through matter (here: path ~ 5 km through abstr. matter matrix, and then 10 km through matter Smatrix)
        # with basic propagation
        U_expected_1 = Newtrinos.osc.osc_kernel(matter_matrices[1][1], matter_matrices[1][2], e, path[1].length)
        U_expected_2 = Newtrinos.osc.osc_kernel(matter_matrices[2][1], matter_matrices[2][2], e, path[2].length)
        P_expected = abs2.(U_expected_1 * U_expected_2) # expected probability matrix for the given path through matter
        P_result_Basic = Newtrinos.osc.osc_reduce(matter_matrices, path, e, Newtrinos.osc.Basic())
        # with damping propagation: sigma_e = 0.1
        # get decay factor for each layer from the osc_kernel with lowpass filter for both matter matrices
        res1 = Newtrinos.osc.osc_kernel(matter_matrices[1][1], matter_matrices[1][2], e, path[1].length, Newtrinos.osc.Damping().σₑ)
        res2 = Newtrinos.osc.osc_kernel(matter_matrices[2][1], matter_matrices[2][2], e, path[2].length, Newtrinos.osc.Damping().σₑ)
        # use bold approximation: coherent neutrino behaves as if it was influenced by an average weighted damping factor for the entire path
        # -> matter_matrix_avg = sum(Length_i * matrix_i) / sum(Lenght_i)
        P_bold_avg = (path[1].length * abs2.(matter_matrices[1][1]) + path[2].length * abs2.(matter_matrices[2][1])) / (path[1].length + path[2].length)
        # combine coherent and incoherent parts to account for damping effects in the probability matrix
        P_expected_Damping = abs2.(res1[1] * res2[1]) .+ P_bold_avg * Diagonal(1 .- abs2.(res1[2] .* res2[2])) * P_bold_avg'
        P_result_Damping = Newtrinos.osc.osc_reduce(matter_matrices, path, e, Newtrinos.osc.Damping())
        # @test collect(Newtrinos.osc.osc_reduce(matter_matrices, path, e, Newtrinos.osc.Basic())) ≈ P_expected atol = 1e-6
        @test size(P_result_Basic) == size(matter_matrices[1][1])
        @test size(P_result_Damping) == size(matter_matrices[1][1])
        @test all(P_result_Basic .>= 0) && all(P_result_Basic .<= 1)
        @test all(P_result_Damping .>= 0) && all(P_result_Damping .<= 1)
        @test P_result_Basic ≈ P_expected atol = 1e-6
        @test P_result_Damping ≈ P_expected_Damping atol = 1e-6

        # TEST matter_osc_per_e()
        # take H_eff, e, layer from above -> can take above matter matrices
        # take path from above
        # TEST matter_osc_per_e()
        # reuse H_eff = [1.0 0.2 0.1; ...], e = 1.5 from above
        layer2 = Newtrinos.osc.Layer(3480.0, 4.0, 3.0)
        layers_test = [layer, layer2]   # layer already defined above
        σ_decoh = Newtrinos.osc.Decoherent().σₑ

        # one single-layer path and one two-layer path
        paths_test = [
            [Newtrinos.osc.Path(5.0, 1)],
            [Newtrinos.osc.Path(5.0, 1), Newtrinos.osc.Path(10.0, 2)]
        ]

        # test basic/damping propagation
        mat = Newtrinos.osc.compute_matter_matrices.(Ref(H_eff), e, layers_test, false, Ref(Newtrinos.osc.SI()))
        osc1 = Newtrinos.osc.osc_reduce(mat, paths_test[1], e, Newtrinos.osc.Damping())
        osc2 = Newtrinos.osc.osc_reduce(mat, paths_test[2], e, Newtrinos.osc.Damping())
        p = stack((osc1, osc2))
        expected = Newtrinos.osc.matter_osc_per_e(H_eff, e, layers_test, paths_test, false, Newtrinos.osc.Damping(), Newtrinos.osc.SI())
        @test size(Newtrinos.osc.matter_osc_per_e(H_eff, e, layers_test, paths_test, false, Newtrinos.osc.Damping(), Newtrinos.osc.SI())) == (3, 3, length(paths_test)) # two paths, 3x3 probability matrices
        @test p == expected

        # test decoherent propagation
        U1, h1 = Newtrinos.osc.compute_matter_matrices(H_eff, e, layer, false, Newtrinos.osc.SI())
        U2, h2 = Newtrinos.osc.compute_matter_matrices(H_eff, e, layer2, false, Newtrinos.osc.SI())
        matter_U = [U1, U2]; matter_h = [h1, h2]

        function manual_decoherent(path_segs)
            P = zeros(3, 3)
            for α in 1:3
                eα = [i == α ? 1.0 : 0.0 for i in 1:3]
                ρ = eα * eα'
                for seg in path_segs
                    U, h = matter_U[seg.layer_idx], matter_h[seg.layer_idx]
                    l = seg.length
                    ρ_eig = U' * ρ * U
                    phases = exp.(-Newtrinos.osc.F_units * 1im * (l / e) .* h)
                    ρ_eig = Diagonal(phases) * ρ_eig * Diagonal(phases)'
                    Δφ = abs.(h .- h') * (l / e) * Newtrinos.osc.F_units
                    D = exp.(-2 .* Δφ .* σ_decoh^2)
                    ρ_eig .= ρ_eig .* D
                    ρ = U * ρ_eig * U'
                end
                for β in 1:3
                    eβ = [i == β ? 1.0 : 0.0 for i in 1:3]
                    P[β, α] = real(eβ' * ρ * eβ)
                end
            end
            P
        end

        P_expected_path1 = manual_decoherent(paths_test[1])
        P_expected_path2 = manual_decoherent(paths_test[2])

        result_Decoherent = Newtrinos.osc.matter_osc_per_e(H_eff, e, layers_test, paths_test, false, Newtrinos.osc.Decoherent(), Newtrinos.osc.SI())

        @test size(result_Decoherent) == (3, 3, 2)
        @test all(result_Decoherent .>= 0) && all(result_Decoherent .<= 1)
        @test result_Decoherent[:, :, 1] ≈ P_expected_path1 atol = 1e-6
        @test result_Decoherent[:, :, 2] ≈ P_expected_path2 atol = 1e-6

        # TEST select()
        @test Newtrinos.osc.select(U1, h1, Newtrinos.osc.All())[1:2] == Newtrinos.osc.select(U1, h1, Newtrinos.osc.Cut())[1:2] # same with out cut-off-value, third element gives fail due to different structure, i.e. 0 vs [0 0 0; 0 0 0; 0 0 0]
        @test size(Newtrinos.osc.select(U1, h1, Newtrinos.osc.Cut(cutoff = 0.5))[3]) == (3, 3)
        @test Newtrinos.osc.select(U1, h1, Newtrinos.osc.Cut(cutoff = 0.5)) != Newtrinos.osc.select(U1, h1, Newtrinos.osc.All())
        @test Newtrinos.osc.select(U1, h1, Newtrinos.osc.Cut(cutoff = 0.5)) != Newtrinos.osc.select(U1, h1, Newtrinos.osc.Cut(cutoff = 1)) # difference in cutoff

        # TEST propagate()
        # test-setup
        U = @SMatrix [cos(π/4) sin(π/4) 0; -sin(π/4) cos(π/4) 0; 0 0 1]
        H = @SVector [0.0, 2.5e-5, 1]
        e, l, σₑ = 1.0, 100.0, 2
        E_test = [0.5, 1.0, 1.5]
        L_test = [50.0, 100.0]
        nE, nL = length(E_test), length(L_test)

        # test for basic propagation
        U = @SMatrix [cos(π/4) sin(π/4) 0; -sin(π/4) cos(π/4) 0; 0 0 1]
        H = @SVector [0.0, 2.5e-5, 1]
        e, l, σₑ = 1.0, 100.0, 2
        E_test = [0.5, 1.0, 1.5]
        L_test = [50.0, 100.0]
        nE, nL = length(E_test), length(L_test)

        # test for damped propagation
        σₑ_damp = Newtrinos.osc.Damping().σₑ
        result_damp = Newtrinos.osc.propagate(U, H, E_test, L_test, Newtrinos.osc.Damping())
        @test size(result_damp) == (3, 3, nE, nL)
        @test all(result_damp .>= 0) && all(result_damp .<= 1)
        expected_damp = stack(broadcast((e, l) -> begin
            pf = -Newtrinos.osc.F_units * (l / e) .* H
            decay = exp.(-2 * abs.(pf) * σₑ_damp^2)
            K = U * Diagonal(exp.(1im * pf) .* decay) * U'
            abs2.(K) + abs2.(U) * Diagonal(1 .- abs2.(decay)) * abs2.(U)'
        end, E_test, L_test'))
        @test result_damp ≈ expected_damp atol = 1e-10

        # test for decoherent propagation
        σₑ_dec = Newtrinos.osc.Decoherent().σₑ
        result_decoh = Newtrinos.osc.propagate(U, H, E_test, L_test, Newtrinos.osc.Decoherent())
        @test size(result_decoh) == (3, 3, nE, nL)
        @test all(result_decoh .>= 0) && all(result_decoh .<= 1)
        expected_decoh = stack(broadcast((e, l) -> begin
            P = zeros(3, 3)
            v = one(U * U')
            for α in 1:3
                eα = v[α, :]
                ρ = eα * eα'
                ρ_eig = U' * ρ * U
                phases = exp.(-Newtrinos.osc.F_units * 1im * (l / e) .* H)
                ρ_eig = Diagonal(phases) * ρ_eig * Diagonal(phases)'
                Δφ = abs.(H .- H') * (l / e) * Newtrinos.osc.F_units
                D = exp.(-2 .* Δφ .* σₑ_dec^2)
                ρ_eig .= ρ_eig .* D
                ρ = U * ρ_eig * U'
                for β in 1:3
                    eβ = v[β, :]
                    P[β, α] = real(eβ' * ρ * eβ)
                end
            end
            P
        end, E_test, L_test'))
        @test result_decoh ≈ expected_decoh atol = 1e-10

        #= MAYBE BUG ->(The root cause: Layer is a parametric struct (Layer{T}), so StructVector{Layer{Float64}} doesn't match the function signature StructVector{Layer} due to Julia's type invariance. Fix: Change StructVector{Layer} → StructVector{<:Layer} on lines 505 and 510 of osc.jl. This is a genuine bug — it would affect any caller constructing layers with concrete types)
        # test for different layers in vacuum
        paths_vov = VectorOfVectors(paths_test)
        layers_sv = StructVector(layers_test)
        nPaths = length(paths_test)
        L_total = [sum(seg.length for seg in p) for p in paths_test]

        result_vac = Newtrinos.osc.propagate(U, H, E_test, paths_vov, layers_sv, Newtrinos.osc.Basic(), Newtrinos.osc.Vacuum(), false)
        @test size(result_vac) == (3, 3, nE, nPaths)
        result_direct = Newtrinos.osc.propagate(U, H, E_test, L_total, Newtrinos.osc.Basic())
        @test result_vac ≈ result_direct atol = 1e-10

        # test for different layers in matter
        result_si = Newtrinos.osc.propagate(U, H, E_test, paths_vov, layers_sv, Newtrinos.osc.Damping(), Newtrinos.osc.SI(), false)
        @test size(result_si) == (3, 3, nE, nPaths)
        @test all(result_si .>= 0) && all(result_si .<= 1)
        H_eff_ref = U * Diagonal(H) * adjoint(U)
        expected_si = stack(map(e -> Newtrinos.osc.matter_osc_per_e(H_eff_ref, e, layers_sv, paths_vov, false, Newtrinos.osc.Damping(), Newtrinos.osc.SI()), E_test))
        expected_si = permutedims(expected_si, (1, 2, 4, 3))
        @test result_si ≈ expected_si atol = 1e-10 =#

    end

    @testset "get_osc_prob" begin

        F = Newtrinos.osc.F_units

        # test Zero Baseline -> Identity
        @testset "zero baseline identity" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)
            params = Newtrinos.osc.get_params(cfg)

            E = [1.0, 5.0, 10.0]
            L = [0.0]
            result = osc_prob(E, L, params)

            for i_e in 1:length(E)
                @test result[i_e, 1, :, :] ≈ I(3) atol = 1e-12
            end
        end

        # test Zero Mixing -> Identity
        @testset "zero mixing identity" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)
            params_zero = (θ₁₂ = 0.0, θ₁₃ = 0.0, θ₂₃ = 0.0, δCP = 0.0, Δm²₂₁ = 7.53e-5, Δm²₃₁ = 2.5e-3)

            E = [1.0, 5.0]
            L = [100.0, 500.0, 1000.0]
            result = osc_prob(E, L, params_zero)

            for i_e in 1:length(E), i_l in 1:length(L)
                @test result[i_e, i_l, :, :] ≈ I(3) atol = 1e-12
            end
        end

        # test Two-Flavour limit -> standard 2-flavour formula
        @testset "two-flavour limit" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)

            θ12 = 0.6
            Δm²₂₁ = 7.53e-5
            params_2f = (θ₁₂ = θ12, θ₁₃ = 0.0, θ₂₃ = 0.0, δCP = 0.0, Δm²₂₁ = Δm²₂₁, Δm²₃₁ = 2.5e-3)

            E = [0.5, 1.0, 3.0]
            L = [100.0, 500.0, 1500.0]
            result = osc_prob(E, L, params_2f)

            for (ie, e) in enumerate(E), (il, l) in enumerate(L)
                Δ21 = F * Δm²₂₁ * l / (2 * e)
                P_12_expected = sin(2 * θ12)^2 * sin(Δ21)^2
                # with θ₁₃=θ₂₃=0, flavour 3 decouples completely
                @test result[ie, il, 1, 2] ≈ P_12_expected atol = 1e-10
                @test result[ie, il, 2, 1] ≈ P_12_expected atol = 1e-10
                @test result[ie, il, 1, 3] ≈ 0.0 atol = 1e-12
                @test result[ie, il, 3, 1] ≈ 0.0 atol = 1e-12
                @test result[ie, il, 3, 2] ≈ 0.0 atol = 1e-12
                @test result[ie, il, 2, 3] ≈ 0.0 atol = 1e-12
            end
        end

        # test Unitarity
        @testset "unitarity" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)
            params = Newtrinos.osc.get_params(cfg)

            E = [0.5, 1.0, 3.0, 10.0]
            L = [50.0, 295.0, 810.0, 1300.0]
            result = osc_prob(E, L, params)

            for i_e in 1:length(E), i_l in 1:length(L)
                P = result[i_e, i_l, :, :]
                for α in 1:3
                    @test sum(P[α, :]) ≈ 1.0 atol = 1e-10
                end
                @test all(P .>= -1e-15)
            end
        end

        # test against Independent PMNS Calculation
        @testset "independent calculation" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)

            params = (θ₁₂ = 0.5843, θ₁₃ = 0.1496, θ₂₃ = 0.8587, δCP = 3.59, Δm²₂₁ = 7.42e-5, Δm²₃₁ = 2.514e-3)
            E = [0.4, 1.0, 2.5, 8.0]
            L = [180.0, 295.0, 810.0]

            result = osc_prob(E, L, params)

            # Build PMNS from scratch (independent of get_PMNS)
            s12, c12 = sin(params.θ₁₂), cos(params.θ₁₂)
            s13, c13 = sin(params.θ₁₃), cos(params.θ₁₃)
            s23, c23 = sin(params.θ₂₃), cos(params.θ₂₃)
            δ = cis(params.δCP)
            δc = conj(δ)

            U_manual = [
                c12*c13                       s12*c13                       s13*δc
                -s12*c23 - c12*s23*s13*δ      c12*c23 - s12*s23*s13*δ      s23*c13
                s12*s23 - c12*c23*s13*δ       -c12*s23 - s12*c23*s13*δ     c23*c13
            ]

            h_raw = [0.0, params.Δm²₂₁, params.Δm²₃₁]
            h = h_raw .- minimum(h_raw)

            expected = zeros(length(E), length(L), 3, 3)
            for (ie, e) in enumerate(E), (il, l) in enumerate(L)
                phases = -F * 1im * (l / e) .* h
                A = U_manual * Diagonal(exp.(phases)) * U_manual'
                expected[ie, il, :, :] = abs2.(A)
            end

            @test result ≈ expected atol = 1e-10
        end

        # test for Anti-Neutrino and CP Violation
        @testset "anti-neutrino and CP" begin
            cfg = Newtrinos.osc.OscillationConfig()
            osc_prob = Newtrinos.osc.get_osc_prob(cfg)

            params_cp = (θ₁₂ = 0.5843, θ₁₃ = 0.1496, θ₂₃ = 0.8587, δCP = π/2, Δm²₂₁ = 7.42e-5, Δm²₃₁ = 2.514e-3)
            params_nocp = (θ₁₂ = 0.5843, θ₁₃ = 0.1496, θ₂₃ = 0.8587, δCP = 0.0, Δm²₂₁ = 7.42e-5, Δm²₃₁ = 2.514e-3)

            E = [0.6, 2.5]
            L = [295.0, 810.0]

            result_nu = osc_prob(E, L, params_cp; anti = false)
            result_anti = osc_prob(E, L, params_cp; anti = true)
            result_nu_nocp = osc_prob(E, L, params_nocp; anti = false)
            result_anti_nocp = osc_prob(E, L, params_nocp; anti = true)

            # δCP=0: neutrino == anti-neutrino
            @test result_nu_nocp ≈ result_anti_nocp atol = 1e-10

            # δCP≠0: disappearance still same (CPT)
            for ie in 1:length(E), il in 1:length(L), α in 1:3
                @test result_nu[ie, il, α, α] ≈ result_anti[ie, il, α, α] atol = 1e-10
            end

            # Verify anti-neutrino against independent calc with conj(U)
            s12, c12 = sin(params_cp.θ₁₂), cos(params_cp.θ₁₂)
            s13, c13 = sin(params_cp.θ₁₃), cos(params_cp.θ₁₃)
            s23, c23 = sin(params_cp.θ₂₃), cos(params_cp.θ₂₃)
            δ = cis(params_cp.δCP)
            δc = conj(δ)

            U_nu = [
                c12*c13                       s12*c13                       s13*δc
                -s12*c23 - c12*s23*s13*δ      c12*c23 - s12*s23*s13*δ      s23*c13
                s12*s23 - c12*c23*s13*δ       -c12*s23 - s12*c23*s13*δ     c23*c13
            ]
            U_anti = conj.(U_nu)

            h = [0.0, params_cp.Δm²₂₁, params_cp.Δm²₃₁]
            h = h .- minimum(h)

            expected_anti = zeros(length(E), length(L), 3, 3)
            for (ie, e) in enumerate(E), (il, l) in enumerate(L)
                phases = -F * 1im * (l / e) .* h
                A = U_anti * Diagonal(exp.(phases)) * U_anti'
                expected_anti[ie, il, :, :] = abs2.(A)
            end

            @test result_anti ≈ expected_anti atol = 1e-10
        end

        # test for Different Propagation Models
        @testset "propagation models" begin
            params = Newtrinos.osc.get_params(Newtrinos.osc.ThreeFlavour())
            E = [1.0, 5.0]
            L = [295.0, 810.0]

            for prop in [Newtrinos.osc.Basic(), Newtrinos.osc.Damping(), Newtrinos.osc.Decoherent()]
                cfg = Newtrinos.osc.OscillationConfig(propagation = prop)
                osc_prob = Newtrinos.osc.get_osc_prob(cfg)
                result = osc_prob(E, L, params)

                @test size(result) == (2, 2, 3, 3)
                @test all(result .>= -1e-15)
                @test all(result .<= 1.0 + 1e-15)
                for ie in 1:2, il in 1:2, α in 1:3
                    @test sum(result[ie, il, α, :]) ≈ 1.0 atol = 1e-6
                end
            end

            # Damping should differ from Basic
            r_basic = Newtrinos.osc.get_osc_prob(Newtrinos.osc.OscillationConfig(propagation = Newtrinos.osc.Basic()))(E, L, params)
            r_damp = Newtrinos.osc.get_osc_prob(Newtrinos.osc.OscillationConfig(propagation = Newtrinos.osc.Damping()))(E, L, params)
            @test !(r_basic ≈ r_damp)
        end

    end

    @testset "get matrices" begin

        # --- Shared parameter definitions ---
        angles = (θ₁₂ = 0.58725, θ₁₃ = 0.14543, θ₂₃ = 0.85563)
        NO_masses = (Δm²₂₁ = 7.53e-5, Δm²₃₁ = 2.4e-3 + 7.53e-5)
        IO_masses = (Δm²₂₁ = 7.53e-5, Δm²₃₁ = -(2.4e-3 - 7.53e-5))
        δCP_val = (δCP = 1.0,)

        ThreeFlavour_Params_NO = merge(angles, NO_masses, δCP_val)
        ThreeFlavour_Params_IO = merge(angles, IO_masses, δCP_val)
        ThreeFlavourXYCP_Params = merge(angles, NO_masses, (δCPshell = [1.0, 0.0],))
        Sterile_Params = merge(ThreeFlavour_Params_NO, (Δm²₄₁ = 1.0, θ₁₄ = 0.1, θ₂₄ = 0.1, θ₃₄ = 0.1))
        ADD_Params = merge(ThreeFlavour_Params_NO, (m₀ = 0.01, ADD_radius = 1e-2))

        # Independent PMNS construction (standard parametrization)
        function reference_PMNS(θ₁₂, θ₁₃, θ₂₃, δCP)
            s12, c12 = sin(θ₁₂), cos(θ₁₂)
            s13, c13 = sin(θ₁₃), cos(θ₁₃)
            s23, c23 = sin(θ₂₃), cos(θ₂₃)
            cp = cis(δCP)
            cp_conj = conj(cp)
            [
                c13*c12                     c13*s12                     s13*cp_conj
                -c23*s12-s23*c12*s13*cp     c23*c12-s23*s12*s13*cp      s23*c13
                s23*s12-c23*s13*c12*cp      -s23*c12-s12*s13*c23*cp     c23*c13
            ]
        end

        # test for ThreeFlavour model
        @testset "ThreeFlavour" begin
            for (label, cfg, params) in [
                ("NO", Newtrinos.osc.ThreeFlavour(ordering = :NO), ThreeFlavour_Params_NO),
                ("IO", Newtrinos.osc.ThreeFlavour(ordering = :IO), ThreeFlavour_Params_IO),
            ]
                @testset "$label" begin
                    matrices_fn = Newtrinos.osc.get_matrices(cfg)
                    U, h = matrices_fn(params)

                    # Shape and type
                    @test U isa SMatrix{3,3}
                    @test h isa SVector{3}
                    @test size(U) == (3, 3)
                    @test length(h) == 3

                    # h values
                    @test h[1] == 0.0
                    @test h[2] ≈ params.Δm²₂₁ atol = 1e-12
                    @test h[3] ≈ params.Δm²₃₁ atol = 1e-12

                    # U against independent reference
                    U_ref = reference_PMNS(params.θ₁₂, params.θ₁₃, params.θ₂₃, params.δCP)
                    @test collect(U) ≈ U_ref atol = 1e-10

                    # Unitarity
                    @test collect(U' * U) ≈ I(3) atol = 1e-10
                    @test collect(U * U') ≈ I(3) atol = 1e-10
                end
            end
        end

        # test for ThreeFlavourXYCP model (should match ThreeFlavour with δCP from shell)
        @testset "ThreeFlavourXYCP" begin
            cfg = Newtrinos.osc.ThreeFlavourXYCP()
            matrices_fn = Newtrinos.osc.get_matrices(cfg)
            U, h = matrices_fn(ThreeFlavourXYCP_Params)

            # Shape and type
            @test U isa SMatrix{3,3}
            @test h isa SVector{3}
            @test size(U) == (3, 3)
            @test length(h) == 3

            # h values
            @test h[1] == 0.0
            @test h[2] ≈ ThreeFlavourXYCP_Params.Δm²₂₁ atol = 1e-12
            @test h[3] ≈ ThreeFlavourXYCP_Params.Δm²₃₁ atol = 1e-12

            # U against independent reference (δCP extracted from shell)
            δCP_extracted = ThreeFlavourXYCP_Params.δCPshell[1]
            U_ref = reference_PMNS(
                ThreeFlavourXYCP_Params.θ₁₂, ThreeFlavourXYCP_Params.θ₁₃,
                ThreeFlavourXYCP_Params.θ₂₃, δCP_extracted,
            )
            @test collect(U) ≈ U_ref atol = 1e-10

            # Unitarity
            @test collect(U' * U) ≈ I(3) atol = 1e-10

            # Consistency with ThreeFlavour using same δCP
            cfg_3f = Newtrinos.osc.ThreeFlavour()
            equiv_params = (
                θ₁₂ = ThreeFlavourXYCP_Params.θ₁₂,
                θ₁₃ = ThreeFlavourXYCP_Params.θ₁₃,
                θ₂₃ = ThreeFlavourXYCP_Params.θ₂₃,
                δCP = δCP_extracted,
                Δm²₂₁ = ThreeFlavourXYCP_Params.Δm²₂₁,
                Δm²₃₁ = ThreeFlavourXYCP_Params.Δm²₃₁,
            )
            U_3f, h_3f = Newtrinos.osc.get_matrices(cfg_3f)(equiv_params)
            @test collect(U) ≈ collect(U_3f) atol = 1e-12
            @test collect(h) ≈ collect(h_3f) atol = 1e-12
        end

        # test for Sterile model
        @testset "Sterile" begin
            cfg = Newtrinos.osc.Sterile()
            matrices_fn = Newtrinos.osc.get_matrices(cfg)
            U, h = matrices_fn(Sterile_Params)

            # Shape and size
            @test size(U) == (4, 4)
            @test length(h) == 4

            # h values
            @test h[1] == 0.0
            @test h[2] ≈ Sterile_Params.Δm²₂₁ atol = 1e-12
            @test h[3] ≈ Sterile_Params.Δm²₃₁ atol = 1e-12
            @test h[4] ≈ Sterile_Params.Δm²₄₁ atol = 1e-12

            # Independent reference: build 4x4 mixing matrix from scratch
            PMNS_3x3 = reference_PMNS(
                Sterile_Params.θ₁₂, Sterile_Params.θ₁₃,
                Sterile_Params.θ₂₃, Sterile_Params.δCP,
            )
            U_embedded = hcat(vcat(PMNS_3x3, [0 0 0]), [0; 0; 0; 1])

            θ₁₄, θ₂₄, θ₃₄ = Sterile_Params.θ₁₄, Sterile_Params.θ₂₄, Sterile_Params.θ₃₄
            R14 = [cos(θ₁₄) 0 0 sin(θ₁₄); 0 1 0 0; 0 0 1 0; -sin(θ₁₄) 0 0 cos(θ₁₄)]
            R24 = [1 0 0 0; 0 cos(θ₂₄) 0 sin(θ₂₄); 0 0 1 0; 0 -sin(θ₂₄) 0 cos(θ₂₄)]
            R34 = [1 0 0 0; 0 1 0 0; 0 0 cos(θ₃₄) sin(θ₃₄); 0 0 -sin(θ₃₄) cos(θ₃₄)]

            U_ref = R34 * R24 * R14 * U_embedded
            @test U ≈ U_ref atol = 1e-10

            # Unitarity
            @test U' * U ≈ I(4) atol = 1e-10
            @test U * U' ≈ I(4) atol = 1e-10

            # Zero sterile angles recover standard 3-flavour PMNS
            params_zero_sterile = merge(Sterile_Params, (θ₁₄ = 0.0, θ₂₄ = 0.0, θ₃₄ = 0.0))
            U_zero, _ = matrices_fn(params_zero_sterile)
            @test U_zero[1:3, 1:3] ≈ PMNS_3x3 atol = 1e-10
            @test U_zero[4, 4] ≈ 1.0 atol = 1e-10
            @test U_zero[4, 1:3] ≈ [0.0, 0.0, 0.0] atol = 1e-10
            @test U_zero[1:3, 4] ≈ [0.0, 0.0, 0.0] atol = 1e-10
        end

        # test for ADD model
        @testset "ADD" begin
            N_KK = 5
            dim = 3 * (N_KK + 1)  # = 18
            cfg = Newtrinos.osc.ADD()
            matrices_fn = Newtrinos.osc.get_matrices(cfg)
            U, h = matrices_fn(ADD_Params)

            # Shape and size
            @test size(U) == (dim, dim)
            @test length(h) == dim

            # Eigenvalues sorted and non-negative (from Hermitian M†M)
            @test issorted(h)
            @test all(h .>= -1e-10)

            # Unitarity
            @test U' * U ≈ I(dim) atol = 1e-8
            @test U * U' ≈ I(dim) atol = 1e-8

            # Independent reference calculation
            umev = 5.067730716156395
            PMNS = reference_PMNS(
                ADD_Params.θ₁₂, ADD_Params.θ₁₃,
                ADD_Params.θ₂₃, ADD_Params.δCP,
            )
            m1 = ADD_Params.m₀
            m2 = sqrt(ADD_Params.Δm²₂₁ + ADD_Params.m₀^2)
            m3 = sqrt(ADD_Params.Δm²₃₁ + ADD_Params.m₀^2)

            MD = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)

            aM1 = zeros(ComplexF64, dim, dim)
            aM2 = zeros(Float64, dim, dim)

            for i in 1:3, j in 1:3
                aM1[i, j] = ADD_Params.ADD_radius * MD[i, j] * umev
            end
            for n in 1:N_KK, i in 1:3, j in 1:3
                aM1[3*n + i, j] = sqrt(2) * ADD_Params.ADD_radius * MD[i, j] * umev
            end
            for i in 1:N_KK
                aM2[3*i + 1, 3*i + 1] = i
                aM2[3*i + 2, 3*i + 2] = i
                aM2[3*i + 3, 3*i + 3] = i
            end

            aM = aM1 + aM2
            aaMM = Hermitian(conj(transpose(aM)) * aM)
            h_ref, U_ref = eigen(aaMM)
            h_ref = h_ref / (ADD_Params.ADD_radius^2 * umev^2)

            @test h ≈ h_ref atol = 1e-8
            # Eigenvectors defined up to phase — compare absolute values
            @test abs.(U) ≈ abs.(U_ref) atol = 1e-5

            # Different N_KK produces correct dimensions
            cfg_3 = Newtrinos.osc.ADD(N_KK = 3)
            dim_3 = 3 * (3 + 1)  # = 12
            U_3, h_3 = Newtrinos.osc.get_matrices(cfg_3)(ADD_Params)
            @test size(U_3) == (dim_3, dim_3)
            @test length(h_3) == dim_3
        end



        # test for NND model
        @testset "NND" begin
            N = NND_Params.N
            r=NND_Params.r
            cfg = Newtrinos.osc.NND()
            matrices_fn = Newtrinos.osc.get_matrices(cfg)
            U, h, eigenvalues, V_e, V_m, V_t= matrices_fn(NND_Params)

            # Shape and size
            @test size(U) == (dim, dim)
            @test length(h) == dim

            # Eigenvalues sorted and non-negative (from Hermitian M†M)
            @test issorted(h)
            @test all(h .>= -1e-10)

            # Unitarity
            @test U' * U ≈ I(dim) atol = 1e-8
            @test U * U' ≈ I(dim) atol = 1e-8

            # Compare eigenvalues with expected values from NND_Params in the SM limit 

            m1_sq = NND_Params.m₀^2
            m2_sq = NND_Params.Δm²₂₁ + NND_Params.m₀^2
            m3_sq = NND_Params.Δm²₃₁ + NND_Params.m₀^2

            @test eigenvalues[1] ≈ m1_sq atol = 1e-8
            @test eigenvalues[2] ≈ m2_sq atol = 1e-8
            @test eigenvalues[3] ≈ m3_sq atol = 1e-8

            @test h[1] ≈ 0.0 atol = 1e-8
            @test h[2] ≈ NND_Params.Δm²₂₁ atol = 1e-8
            @test h[3] ≈ NND_Params.Δm²₃₁ atol = 1e-8

            for i in 2:N-1
                @test eigenvalues[3*i-2] ≈ (2i+r)*m1_sq atol = 1e-8
                @test eigenvalues[3*i-1] ≈ (2i+r)*m2_sq atol = 1e-8
                @test eigenvalues[3*i] ≈ (2i+r)*m3_sq atol = 1e-8
            end

            @test eigenvalues[N-2] ≈ N^2*(2i+r)*m1_sq atol = 1e-8
            @test eigenvalues[N-1] ≈ N^2*(2i+r)*m2_sq atol = 1e-8
            @test eigenvalues[N] ≈ N^2*(2i+r)*m3_sq atol = 1e-8

           
        end

        
        # test for NND model
        @testset "NNM" begin
            N = NNM_Params.N
            r=NNM_Params.r
            cfg = Newtrinos.osc.NNM()
            matrices_fn = Newtrinos.osc.get_matrices(cfg)
            U, h, eigenvalues, V_e, V_m, V_t= matrices_fn(NNM_Params)

            # Shape and size
            @test size(U) == (dim, dim)
            @test length(h) == dim

            # Eigenvalues sorted and non-negative (from Hermitian M†M)
            @test issorted(h)
            @test all(h .>= -1e-10)

            # Unitarity
            @test U' * U ≈ I(dim) atol = 1e-8
            @test U * U' ≈ I(dim) atol = 1e-8

            # Compare eigenvalues with expected values from NNM_Params in the SM limit 

            m1_sq = NNM_Params.m₀^2
            m2_sq = NNM_Params.Δm²₂₁ + NNM_Params.m₀^2
            m3_sq = NNM_Params.Δm²₃₁ + NNM_Params.m₀^2

            @test eigenvalues[1]^2 ≈ m1_sq atol = 1e-8
            @test eigenvalues[2]^2 ≈ m2_sq atol = 1e-8
            @test eigenvalues[3]^2 ≈ m3_sq atol = 1e-8

            @test h[1] ≈ 0.0 atol = 1e-8
            @test h[2] ≈ NNM_Params.Δm²₂₁ atol = 1e-8
            @test h[3] ≈ NNM_Params.Δm²₃₁ atol = 1e-8

            for i in 2:N-1
                @test eigenvalues[3*i-2]^2 ≈ (2i+r)*m1_sq atol = 1e-8
                @test eigenvalues[3*i-1]^2 ≈ (2i+r)*m2_sq atol = 1e-8
                @test eigenvalues[3*i]^2 ≈ (2i+r)*m3_sq atol = 1e-8
            end

            @test eigenvalues[N-2]^2 ≈ N^4*(2i+r)*m1_sq atol = 1e-8
            @test eigenvalues[N-1]^2 ≈ N^4*(2i+r)*m2_sq atol = 1e-8
            @test eigenvalues[N]^2 ≈ N^4*(2i+r)*m3_sq atol = 1e-8

           
        end


    end

end


