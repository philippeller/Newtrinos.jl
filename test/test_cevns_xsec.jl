using LinearAlgebra
using Distributions
using Newtrinos
using SpecialFunctions
using Test

#define test isotopes with known properties for testing
#isotope = (proton number Z, neutron number N, mass (MeV), fraction of isotope wrt to abundance of other isotopes, nominal neutron radius Rn_nom in fm, unique key Rn_key)
cs133 = (Z=55, N=78, mass=123.8, fraction=0.5, Rn_nom=4.83, Rn_key=:Rn_Cs133) # Cesium-133
i127  = (Z=53, N=74, mass=118.2, fraction=0.5, Rn_nom=4.75, Rn_key=:Rn_I127) # Iodine-127
ar40 = (Z = 18, N = 22, mass = 37.211, fraction = 1.0, Rn_nom = 3.47, Rn_key = :Rn_Ar40) # Argon-40
ge76 = (Z=32, N=44, mass=71.922, fraction=1.0, Rn_nom=4.0, Rn_key=:Rn_Ge76) # Germanium-76
isotopes_test = [cs133, i127, ar40, ge76]

@testset "cevns cross sections" begin
    
    @testset "default params within prior support" begin
        params, priors = Newtrinos.cevns_xsec.build_params_and_priors(isotopes_test)
        for key in keys(params)
            @test Distributions.insupport(priors[key], params[key])
            #@test params[key] >= 0.0 , !! some params can be negative !!
        end
    end

    @testset "configure" begin
        xsec = Newtrinos.cevns_xsec.configure(isotopes_test, [1.0, 5.0, 10.0], [10.0, 20.0, 50.0])
        @test xsec isa Newtrinos.cevns_xsec.CevnsXsec
        @test keys(xsec.params) == keys(xsec.priors)
        @test xsec.diff_xsec isa Function
    end

    @testset "asset collection" begin
        assets = Newtrinos.cevns_xsec.get_assets(isotopes_test, [0,1,5,10], [0,5,10,15,20])
        
        #has correct types and keys
        @test assets.isotopes isa Dict
        @test haskey(assets, :isotopes)
        @test haskey(assets, :er_centers)
        @test haskey(assets, :enu_centers)

        #correct allocation and dimensions
        @test length(assets.isotopes) == 4
        @test assets.isotopes[:Rn_Cs133].mass == 123.8
        @test assets.isotopes[:Rn_Ar40].Z == 18
        @test assets.er_centers == [0, 1, 5, 10]
        @test assets.enu_centers == [0, 5, 10, 15, 20]

        #empty assets edge case
        empty_assets = Newtrinos.cevns_xsec.get_assets([], [], [])
        @test isempty(empty_assets.isotopes)
        @test isempty(empty_assets.er_centers)
        @test isempty(empty_assets.enu_centers)

    end

    @testset "form factor function" begin
        #calculate specific cases 
        #edge cases where at least on input is 0
        @test Newtrinos.cevns_xsec.ffsq(0, 0, 0) == 1.0 
        @test Newtrinos.cevns_xsec.ffsq(0,1,1) == 1.0
        @test Newtrinos.cevns_xsec.ffsq(1,0,1) == 1.0
        @test Newtrinos.cevns_xsec.ffsq(1, 1, 0) ≈ 0.9999583961 atol=1e-6

        #typical cases with known values
        @test Newtrinos.cevns_xsec.ffsq(1, 1, 1) ≈ 0.9999481150 atol=1e-6
        @test Newtrinos.cevns_xsec.ffsq(10, 123.8, 4.83) ≈ 0.70133086 atol=1e-6 #Cs133: m=123.8 MeV, Rn=4.83 fm, E=10 MeV
        @test Newtrinos.cevns_xsec.ffsq(1000, 37.211, 3.47) ≈ 0.00030909 atol=1e-6 #Ar40: m=37.211 MeV, Rn=3.47 fm, E=1000 MeV
        @test Newtrinos.cevns_xsec.ffsq(0.001, 71.922, 4.0) ≈ 0.99998518 atol=1e-6 #Ge76: m=71.922 MeV, Rn=4.0 fm, E=0.001 MeV
    end

    @testset "differential cross section calculation" begin
        er, enu = [0,1,5,10], [5,10,15,20,500000]
        params, priors = Newtrinos.cevns_xsec.build_params_and_priors(isotopes_test)
        nupar = (118.2, 53, 74) # (mn, Z, N)
        Rn_key = :Rn_I127
        Newtrinos.cevns_xsec.ds(er, enu, params, nupar, Rn_key)

        #structure of output
        @test size(Newtrinos.cevns_xsec.ds(er,enu,params, nupar, Rn_key)) == (length(er), length(enu)) #returned array has the correct grid dimensions
        @test all(Newtrinos.cevns_xsec.ds(er,enu,params, nupar, Rn_key) .>= 0.0) #cross section should be non-negative
        
        #edge cases
        @test all(isnan.(Newtrinos.cevns_xsec.ds(er,[0],params, nupar, Rn_key))) #get NaN for enu = 0 as ds->infinity
        @test all(isapprox.(Newtrinos.cevns_xsec.ds([0.0], enu, params, nupar, Rn_key), 0.0, atol=1e-10)) #get 0 for er = 0 as no recoil energy
        @test all(Newtrinos.cevns_xsec.ds([1], [1], params, (0,0,0), Rn_key) .== 0.0) #get 0 values for nupar = (0,0,0) as this is unphysical and should yield zero cross section

        #specific cases with known expected values 
        # Input Values for explicit calculation(Argon-40) 
        er = [2.0]        # MeV
        enu = [20.0]      # MeV
        mN = 37211.0      # MeV (~40 amu)
        Z = 18
        N = 22
        rn = 3.47         # fm
        sw2 = 0.231
        gf = 1.1663787e-11 # MeV^-2

        qwsq = (N - (1 - 4 * sw2) * Z)^2
        ffsq = Newtrinos.cevns_xsec.ffsq(er[1], mN, rn)

        # Standard Model Kinematics (where a=b=c=d=0)
        SM_params = (
            sin2thetaW = sw2,
            cevns_xsec_a = 0.0,
            cevns_xsec_b = 0.0,
            cevns_xsec_c = 0.0,
            cevns_xsec_d = 0.0,
            Rn_Ar40 = rn
        )

        term1 = qwsq
        term2 = -qwsq * (er[1] / enu[1]) # This is base2 in code
        term3 = -qwsq * (mN * er[1] / (2 * enu[1]^2)) # This is base3 in code
        term4 = qwsq * (er[1] / enu[1])^2 # This is base4 in code

        #expected result
        xf_expected_SM = term1 + term2 + term3 + term4
        expected_result_SM = (gf^2 / (4 * pi) * ffsq * mN) * xf_expected_SM

        # function result
        nupar = (mN, Z, N)
        actual_SM = Newtrinos.cevns_xsec.ds(er, enu, SM_params, nupar, :Rn_Ar40)
        @test actual_SM[1, 1] ≈ expected_result_SM atol=1e-10

        #NSI-modified kinematics (random example non-zero NSI parameters)
        NSI_params = (
            sin2thetaW = sw2,
            cevns_xsec_a = 0.1, 
            cevns_xsec_b = 0.05,
            cevns_xsec_c = -0.02,
            cevns_xsec_d = 0.01,
            Rn_Ar40 = rn
        )

        term1_NSI = qwsq * (1 + NSI_params.cevns_xsec_a)
        term2_NSI = -qwsq * (1 + NSI_params.cevns_xsec_b) * (er[1] / enu[1])
        term3_NSI = -qwsq * (1 + NSI_params.cevns_xsec_c) * (mN * er[1] / (2 * enu[1]^2))
        term4_NSI = qwsq * (1 + NSI_params.cevns_xsec_d) * (er[1] / enu[1])^2

        #expected result with NSI
        xf_expected_NSI = term1_NSI + term2_NSI + term3_NSI + term4_NSI
        expected_result_NSI = (gf^2 / (4 * pi) * ffsq * mN) * xf_expected_NSI   
        actual_NSI = Newtrinos.cevns_xsec.ds(er, enu, NSI_params, nupar, :Rn_Ar40)
        @test actual_NSI[1, 1] ≈ expected_result_NSI atol=1e-10
        
    end

    @testset "cross section collection" begin 
        #test various versions of get_diff_xsec
        
        er, enu = [0,1,5,10], [5,10,15,20,500000]
        params, priors = Newtrinos.cevns_xsec.build_params_and_priors(isotopes_test)
        nupar = (118.2, 53, 74) # (mn, Z, N)
        Rn_key = :Rn_I127
        assets = Newtrinos.cevns_xsec.get_assets(isotopes_test, er, enu)

        #get_diff_xsec_lar and get_diff_xsec_csi get consistent results with direct ds calculation
        expected = Newtrinos.cevns_xsec.ds(er, enu, params, nupar, Rn_key)
        get_lar = Newtrinos.cevns_xsec.get_diff_xsec_lar()(er, enu, params, nupar, Rn_key)
        get_csi = Newtrinos.cevns_xsec.get_diff_xsec_csi()(er, enu, params, nupar, Rn_key)
        @test get_lar == expected 
        @test get_csi == expected

        #get_diff_xsec 
        results= Newtrinos.cevns_xsec.get_diff_xsec(assets)(params)
        @test results[:Rn_I127] == Newtrinos.cevns_xsec.ds(er, enu, params, (118.2, 53, 74), :Rn_I127)
        @test results[:Rn_Cs133] == Newtrinos.cevns_xsec.ds(er, enu, params, (123.8, 55, 78), :Rn_Cs133)
        @test results[:Rn_Ar40] == Newtrinos.cevns_xsec.ds(er, enu, params, (37.211, 18, 22), :Rn_Ar40)
        @test results[:Rn_Ge76] == Newtrinos.cevns_xsec.ds(er, enu, params, (71.922, 32, 44), :Rn_Ge76)
    end
end