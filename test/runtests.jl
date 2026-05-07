using Newtrinos
using Test

@testset "Newtrinos.jl" begin
    @testset "physics" begin
        include("test_osc.jl")
        include("test_earth_layers.jl")
        include("test_xsec.jl")
        include("osc_unit_tests.jl")
        include("test_snsflux.jl")
        include("test_atmflux.jl")
        include("test_cevns_xsec.jl")
        include("test_fast_eigen_3x3.jl")
    end

    @testset "analysis" begin
        include("test_helpers.jl")
        include("test_analysis.jl")
        include("test_autodiff.jl")
        include("test_regression.jl")
    end
    
    #@testset "experiments" begin 
    #end 
end
