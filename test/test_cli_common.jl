using Test
using Newtrinos
using Accessors
using FileIO
using CairoMakie

include("../src/analysis/cli_common.jl")

@testset "CLI Common" begin

    @testset "configure_experiments (defaults)" begin
        experiments = configure_experiments(["dayabay"])
        @test haskey(experiments, :dayabay)
        @test experiments.dayabay isa Newtrinos.Experiment
    end

    @testset "configure_experiments (physics override)" begin
        osc = Newtrinos.osc.configure()
        osc_mod = @reset osc.params.θ₁₂ = 0.3   # differs from default ~0.588
        physics = (osc = osc_mod,)
        experiments = configure_experiments(["dayabay"], physics)
        @test experiments.dayabay isa Newtrinos.Experiment
        @test experiments.dayabay.physics.osc.params.θ₁₂ ≈ 0.3
    end

    @testset "save_result" begin
        result = Newtrinos.NewtrinosResult(
            axes=(x=[1.0, 2.0, 3.0],),
            values=(llh=[-10.0, -5.0, -8.0], log_posterior=[-10.0, -5.0, -8.0])
        )
        mktempdir() do tmpdir
            name = joinpath(tmpdir, "test_save")
            save_result(result, name)
            @test isfile(name * ".jld2")
            loaded = FileIO.load(name * ".jld2")
            @test haskey(loaded, "result")
            @test loaded["result"] isa Newtrinos.NewtrinosResult
            @test loaded["result"].axes == result.axes
            @test loaded["result"].values.llh == result.values.llh
            @test loaded["result"].values.log_posterior == result.values.log_posterior
        end
    end

    # didnt really figure out how to test the plotting function without actually creating files, 
    # and looking at them manually, so here we are. 
    # The tests just test that it creates a file and doesn't error out.
    
    @testset "plot_result (1D)" begin
        result = Newtrinos.NewtrinosResult(
            axes=(x=[1.0, 2.0, 3.0],),
            values=(llh=[-10.0, -5.0, -8.0], log_posterior=[-10.0, -5.0, -8.0])
        )
        vars_to_scan = (x=[1.0, 2.0, 3.0],)
        mktempdir() do tmpdir
            name = joinpath(tmpdir, "test_plot")
            plot_result(result, name, vars_to_scan)
            @test isfile(name * ".png")
        end
    end

    @testset "plot_result (with title)" begin
        result = Newtrinos.NewtrinosResult(
            axes=(x=[1.0, 2.0, 3.0],),
            values=(llh=[-10.0, -5.0, -8.0], log_posterior=[-10.0, -5.0, -8.0])
        )
        vars_to_scan = (x=[1.0, 2.0, 3.0],)
        mktempdir() do tmpdir
            name = joinpath(tmpdir, "test_plot_title")
            plot_result(result, name, vars_to_scan; title="Test Title")
            @test isfile(name * ".png")
        end
    end

end
