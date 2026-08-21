using Documenter
using Newtrinos

gh_repo = get(ENV, "GITHUB_REPOSITORY", "Newtrinos-org/Newtrinos.jl")
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
        "Getting Started" => [
            "Installation" => "installation.md", 
            "First example" => "getting_started.md",
        ],
        "Tutorials" => [
            "Architecture" => "tutorials/architecture.md",
            "Neutrino physics" => "tutorials/neutrino_physics.md",
            "Assemble a physics model" => "tutorials/physics_model.md",
            "Experiments" => "tutorials/experiments.md",
            "Configure an experiment" => "tutorials/configure_experiment.md",
            "Analyse data" => "tutorials/analyse_data.md",
            "Analysis CLI Reference" => "tutorials/cli.md",
            "Automatic Differentiation" => "tutorials/autodiff.md"
        ], 
        "Examples" => [ 
            "Single Experiment" => "examples/single_experiment.md", 
            "Custom Physics" => "examples/custom_physics.md",
            "Joint Analysis" => "examples/joint_analysis.md",
            "Performance" => "examples/performance.md",
        ],
        "API Documentation" => [
            "Types" => "api/types.md",
            "Physics" => "api/physics.md",
            "Analysis" => "api/analysis.md",
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
