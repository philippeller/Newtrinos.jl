using Documenter
using Newtrinos

gh_repo = get(ENV, "GITHUB_REPOSITORY", "philippeller/Newtrinos.jl")
repo_owner = split(gh_repo, "/")[1]

makedocs(
    sitename = "Newtrinos.jl",
    modules = [Newtrinos],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://$(repo_owner).github.io/Newtrinos.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md", 
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "Architecture" => "tutorials/architecture.md",
            "Neutrino physics" => "tutorials/neutrino_physics.md",
            "Experiments" => "tutorials/experiments.md",
            "Assemble a physics model" => "tutorials/physics_model.md",
            "configure an experiment" => "tutorials/configure_experiment.md",
            "analyse data" => "tutorials/analyse_data.md",
        ], 
        "examples" => [ 
            "Single Experiment" => "examples/single_experiment.md",
            "Joint Analysis" => "examples/joint_analysis.md",
            "Custom Physics" => "examples/custom_physics.md",
            "CLI Reference" => "examples/cli.md",
            "Performance" => "examples/performance.md",
        ],
        "API Documentation" => [
            "Types" => "api/types.md",
            "Physics" => "api/physics.md",
            "Analysis" => "api/analysis.md",
            "internal" => "api/internal.md",
        ],
        "Contribution guidelines" => "contribution_guidelines.md",

    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/$(gh_repo).git",
    devbranch = "main",
    #push_preview = true,
)
