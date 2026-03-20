using Newtrinos
using Test

@info "Running tests with $(Threads.nthreads()) threads"

@testset "Newtrinos.jl" begin
    include("test_helpers.jl")
    include("test_osc.jl")
    include("test_earth_layers.jl")
    include("test_xsec.jl")
    include("test_analysis.jl")
    include("test_autodiff.jl")
    include("test_regression.jl")

    if get(ENV, "NEWTRINOS_TEST_MOONCAKE", "") == "1"
        include("test_mooncake.jl")
    else
        @info "Skipping Mooncake tests (set NEWTRINOS_TEST_MOONCAKE=1 to enable)"
    end
end
