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
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "Single Experiment" => "tutorials/single_experiment.md",
            "Joint Analysis" => "tutorials/joint_analysis.md",
            "Custom Physics" => "tutorials/custom_physics.md",
        ],
        "Manual" => [
            "Architecture" => "manual/architecture.md",
            "Experiments" => "manual/experiments.md",
            "CLI Reference" => "manual/cli.md",
            "Performance" => "manual/performance.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Analysis" => "api/analysis.md",
            "Physics" => "api/physics.md",
        ],
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/$(gh_repo).git",
    devbranch = "main",
    #push_preview = true,
)
