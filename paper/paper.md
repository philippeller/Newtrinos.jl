---
title: 'Newtrinos.jl: A Julia Package for Global Analysis of Neutrino Data'
tags:
  - julia
  - neutrino
  - oscillation
  - global fit
  - global analysis
  - joint fit
  - joint likelihood
  - differential Programming
authors:
  - name: Philipp Eller
    orcid: 0000-0001-6354-5209
    affiliation: 1
  - name: David Schultheiß
    orcid: 0009-0000-3027-684X
    affiliation: 1
affiliations:
 - name: Technical University of Munich, Germany
   index: 1
date: 28 October 2025
bibliography: paper.bib
---

# Summary
*Newtrinos.jl* is an open-source Julia package for performing global analyses of neutrino data. It provides a modular, three-layer architecture that separates physics models, experiment descriptions, and statistical inference into independent modules. This allows researchers to freely combine experiments and test them against a variety of theoretical models. New experiments and physics models can be added without modifying core code.

Full statistical forward models including all relevant systematic uncertainties are implemented for each experimental dataset, defining both the likelihood and the data-generating process and enabling modern statistical inference workflows.

The framework is composable, making it straightforward to construct joint likelihoods over multiple datasets. Physics and nuisance parameters can be merged, correlated, or decorrelated across experiments to ensure consistency in the joint fit.

The package supports Frequentist and Bayesian inference and is parallelizable across CPU threads or distributed workers. Written entirely in Julia, all models are automatically differentiable, enabling exact gradient computation.

# Statement of Need
As neutrino physics transitions into an era of high-precision measurements, global fits have become essential for extracting neutrino properties and probing physics beyond the Standard Model [@Capozzi:2025wyn; @Esteban:2024eli; @deSalas:2020pgw]. By combining disparate datasets, they break parameter degeneracies that limit single-experiment analyses. Consequently, they are poised to address urgent open questions such as the neutrino mass ordering and CP violation in the lepton sector. The first joint fit between T2K and NOvA demonstrated that combining distinct experimental signatures is crucial to resolving such degeneracies and tensions [@T2K:2025wet].

Performing a global neutrino fit requires assembling diverse, experiment-specific detector models, neutrino flux models, matter density profiles, interaction cross sections, et cetera, into a coherent inference pipeline, while rigorously accounting for large numbers of correlated systematic uncertainties. Varying availability of public data and documentation further limits dataset compatibility, making global fits a substantial undertaking that can render certain analyses infeasible [@Capozzi:2025wyn]. Existing frameworks that have enabled successful studies are typically proprietary and closed-source, making results difficult to verify and the software impossible to reuse or adapt.

A further challenge is the computational burden of high-dimensional parameter spaces with correlated systematics. As the number of experiments grows, traditional statistical methods become increasingly intractable. Automatic differentiation and gradient-based inference offer promising paths forward. However, existing frameworks (see Related Work) were developed before these techniques became widely adopted and therefore rely primarily on derivative-free optimization or finite-difference gradient estimates such as those used by MINUIT [@James:1975dr].  

*Newtrinos.jl* addresses these limitations with a fully open-source, extensible framework built within the high-performance Julia ecosystem.

# Key Features

Key features include:

- **Out-of-the-box usability**: The package includes experimental data, configuration files, and plotting tools, making it straightforward to reproduce results, verify correctness, and conduct custom analyses on existing models and experiments.

- **Modular architecture**: Experiments, physics models, and analysis methods are implemented as independent components with clearly separated concerns. The physics layer provides theory predictions and physical parameters; the experiment layer configures individual experiments with their parameters, priors, and forward models; the analysis layer provides inference methods for sampling, profiling, or scanning. Each layer treats the others as black boxes, enabling flexible composition. Layers communicate through two well-defined interfaces: NamedTuples of parameters and priors adhering to the Distributions.jl standard, and callable functions stored in structs. Theory extensions can be added via Julia's multiple dispatch without modifying existing code, and parameters flow through the pipeline automatically.

- **Full statistical models**: The package constructs full statistical forward models for each experiment, encoding the complete mapping from physical parameters to expected observations. Systematic uncertainties enter as nuisance parameters rather than being absorbed into simplified $\chi^2$ approximations, preserving the full statistical information. This enables profiling or marginalisation of joint likelihoods, Bayesian posterior estimation, and the generation of pseudo-data and prior and posterior predictive checks.

- **Consistent parameter handling**: All physics and nuisance parameters across configured experiments are collected into a single, flat NamedTuple via `get_params` and `get_priors` with `safe_merge`. Wrapper functions support arbitrary parameter renaming and (de)correlation between experiments. Prior distributions, including correlated priors via covariance matrices, are managed in the same unified structure and compose naturally into a joint prior. Individual parameters can be overridden or conditioned at analysis time without modifying the underlying modules, keeping configurations transparent and reproducible.

- **Automatic differentiation**: The complete forward model chain — from oscillation probability calculation to likelihood evaluation — is compatible with *ForwardDiff.jl* out of the box, with Mooncake.jl also successfully tested. Exact, efficient gradients enable advanced optimisation and inference methods such as those provided by [@Schulz:2021BAT].

- **Scalability**: The package is built for large-scale inference tasks with tens of experiments and hundreds of parameters. Julia's just-in-time compilation delivers high performance without sacrificing expressiveness. Profile likelihood scans parallelise transparently across CPU threads or distributed workers via Distributed.jl, and forward-mode automatic differentiation keeps gradient-based inference tractable as the parameter space grows.


# Workflow Overview

*Newtrinos.jl* is designed as a modular pipeline where components communicate strictly through functional interfaces: physics modules expose callables used inside experiment forward models, and experiments expose forward model callables consumed by the analysis layer, without either inspecting the other's internals. A typical analysis proceeds as follows:

1. **Configure physics**: Select and instantiate theoretical models — for example, a three-flavour oscillation model with non-standard matter interactions, an Earth density model, an atmospheric flux model, and a cross-section model. Any model can be swapped independently, e.g. replacing a three-flavour model with a BSM model including sterile neutrinos.

2. **Configure experiments**: Select one or more experiment modules, each encapsulating its detector response, systematic uncertainties, and observed data. Instantiate them via the `configure` method with the chosen physics model. Experiments are independent and can be freely combined, typically sharing a common physics model.

3. **Build the joint likelihood**: Pass the collection of configured experiments to `generate_likelihood`, which composes their individual forward models and likelihood functions into a single joint likelihood.

4. **Collect parameters and priors**: Use `get_params` and `get_priors` to collect and merge all physics and nuisance parameters across experiments into a unified NamedTuple. Parameters and priors can be modified as needed, e.g. for likelihood conditioning via Accessors.jl.

5. **Run inference**: Choose an analysis method from the provided tools. For example, run a profile likelihood scan by calling `profile` on the joint likelihood over a chosen parameter grid to obtain a `NewtrinosResult` containing the grid coordinates, likelihood values, optimised nuisance parameters at each point, and run metadata.

6. **Visualize and export results**: Plot confidence contours or best-fit data/MC comparisons using the built-in plotting utilities based on Makie.jl, and save results to disk.

![Typical workflow for global analyses of neutrino data with *Newtrinos.jl*. The layers communicate through two well-defined interfaces: NamedTuples of parameters and priors, and callable functions stored in structs. \label{workflow}](Workflow_Diagram.pdf)

# Availability
*Newtrinos.jl* is open-source and freely available under the MIT License at https://github.com/philippeller/Newtrinos.jl .

# Related Work
Several existing software projects address related but distinct use cases:

- **GLoBES** [@Huber:2004ka; @Huber:2007ji]: Simulates long-baseline experiments but does not support full global fits.
- **GAMBIT** [@GAMBIT:2017yxo]: A general-purpose global fitting framework with some support for neutrino data, but not tailored for neutrino physics.
- **PEANUTS** [@Gonzalo:2023mdh]: Focused on solar neutrino modelling.
- **PhyLiNO** [@Hellwig:2025jxe]: A high-performance framework for reactor neutrino data.
- **PISA** [@IceCube:2018ikn]: Designed for atmospheric neutrino analyses.

*Newtrinos.jl* complements these efforts by focusing on the neutrino sector, offering a simple, extensible, and efficient design. It is currently the only framework in this domain supporting automatic differentiation. The software has been used in [@Ettengruber:2024fcq], [@Kozynets:2024xgt], [@Eller:2025lsh], and [@Eller:2026urd].

# Acknowledgements
This work was supported by Germany's Federal Ministry of Research, Technology and Space (BMFTR) within the ErUM-Data programme under grant FKZ 05D25PC1 (DEMOS consortium), and partially by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC-2094/2 – 390783311, the SFB 1258– 283604770, and NFDI 39/1.