---
title: 'Newtrinos.jl: A Julia Package for Global Analysis of Neutrino data'
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
Newtrinos.jl is an open-source Julia package for performing global analyses of neutrino data. It provides a modular, three-layer architecture that separates theoretical physics models, experiment configurations, and statistical inference methods. This allows researchers to freely combine data from a wide range of supported neutrino experiments and test them against a variety of theoretical models. New experiments and physics models can be easily added without modifying core code, making the package expandable. The Out-of-the-box usability of the package makes it easy to use and appealing for a broad range of researchers in the field. For the inference methods, full statistical likelihoods, that include all relevant systematic uncertainties as nuisance parameters, are constructed for each experiment alone and then are composed into a joint likelihood over the complete dataset. The unified parameter handling system automatically merges physics and nuisance parameters across the modules and between the experiments, ensuring consistency and eliminating redundancy in the joint fit. For inference, the package provides profile likelihood scanning, Bayesian posterior sampling, and Asimov-based sensitivity projections, all parallelisable across CPU threads or distributed workers. Written entirely in Julia, the complete forward model chain is compatible with ForwardDiff.jl out of the box, enabling exact gradient computation making it suitable for modern scientific workflows.

# Statement of Need
As neutrino physics transitions into an era of high-precision measurements, global fits of experimental neutrino data have become essential for understanding and extracting neutrino properties and probing new physics beyond the Standard Model [@Capozzi:2025wyn; @Esteban:2024eli; @deSalas:2020pgw]. While single-experiment analyses provide highly precise measurements of specific physical channels, global analyses complement them by systematically combining disparate datasets to break parameter degeneracies and maximize overall statistical sensitivity. Thus, urgent questions in the field, such as the Neutrino Mass Ordering or CP violation in the lepton sector, are also addressed in global fits. Recent efforts underscore this necessity; for example, the first joint fit between the T2K and NOvA experiments demonstrated that combining distinct experimental signatures is crucial to breaking parameter degeneracies [@T2K:2025wet].

Performing a global neutrino fit requires collecting diverse, experiment-specific detector models, beam fluxes, and matter density profiles and then build an inference . Furthermore, the process is complicated by the need to rigorously account for a vast number of correlated systematic uncertainties and nuisance parameters for each contained experiment to ensure the statistical consistency of the joint data. On top of that, the relatively bad availability of public data and information regarding the different experiments complicates the process even further. So global fits require huge efforts and thus, analyses sometimes seem to be unfeasible. For example the analysis in [@Capozzi:2025wyn] excludes atmospheric neutrino experiments entirely due to such limitations, stating "For atmospheric neutrinos, currently involving hundreds of bins, dozens of systematic uncertainties, and refined statistical separation of event classes by flavor proxies, the construction of $\chi^2$ maps based on public information has become eventually unfeasible outside the experimental collaborations”.

While the existing studies have enabled many successful studies, they often rely on proprietary, closed-source implementations. This makes their results difficult to verify and the software impossible to reuse or adapt. Additionally, within those implementations the experiment definitions, systematic nuisances, and physics models are often deeply intertwined. These frameworks often rely on a multi-language architecture, typically using C++ or Fortran for the performance-critical numerical engine and Python for the user interface, which complicates maintenance and creates language barriers for community contributions and transparency. All these points make the modification of underlying physics models and combination of different experiments within the framework even more difficult.

Another issue is the computational burden associated with managing high-dimensional parameter spaces and correlated systematic uncertainties. As the dimensionality of this parameter space grows with more experiments, traditional statistical methods become increasingly intractable. Recent advances in differentiable programming and gradient-based inference, like automatic differentiation, offer promising approaches for exploring such parameter spaces efficiently. However, the existing analysis frameworks were developed before automatic differentiation became widely available and therefore rely primarily on derivative-free optimization or finite-difference gradient estimates, such as those employed by MINUIT [@James:1975dr]. While these approaches are robust and well established, they scale poorly and become prohibitively slow as more nuisance parameters are introduced. Consequently, these frameworks hit a strict computational ceiling. Retrofitting an end-to-end automatic differentiation functionality in the existing packages is very difficult due to the lack of differentiality across programming language barriers and the presence of non-differentiable pipeline parts like lookup tables or interpolation grids. So unfortunately, no established neutrino global fit framework currently is able to utilize automatic differentiation.

To address all those critical problems, we introduce *Newtrinos.jl*. Built within the high-performance Julia ecosystem, this package provides a fully open-source, extensible framework explicitly designed to overcome the structural and computational limitations of legacy tools.

# Key Features
*Newtrinos.jl* takes a different approach compared to established global fit frameworks. It is fully open-source, modular, and computationally efficient. Written entirely in the Julia programming language, it offers high performance and a clear structure optimized for combining and expanding diverse theoretical models and experimental datasets.
Key features include:

* **Out-of-the-box usability**: The package includes the experimental data, configuration files, and plotting tools, making it easy to reproduce results and verify correctness. The straightforward workflow also makes it easy to do own analyses on existing models and experiments.

* **Modular architecture**: Experiments, physics models, and analysis methods are implemented as independent components with mutually separated concerns. The physics model contains the physics theory predictions and physical parameters and priors. In the Experiment layer one configures the different experiments and their parameters, priors, and forward models. The Analysis layer contains the inference methods for sampling, profiling, or scanning. Each layer treats the other layers as black boxes, allowing flexible composition for statistical data analysis of neutrino data. The layers communicate through two well-defined interfaces: NamedTuples of parameters and priors, and callable functions stored in structs. A general pipeline is described in the workflow section. Adding functionalities to existing methods in the source code is also very convenient using the multiple dispatch functionality of Julia. Thus, a theory extension can just be dispatched as an additional method and the parameters automatically flow in the pipeline. This modular design philoshopy allows that a large number of experiments can be added, combined, and tested against arbitrary theories without modifying core code.

* **Full likelihood support**: The package constructs full statistical likelihoods for each experiment, encoding the complete mapping from physical parameters to expected observations via each experiment's forward model. Systematic uncertainties enter as nuisance parameters directly in the likelihood rather than being absorbed into simplified $\chi^2$ approximations, preserving the full statistical information. This allows the joint likelihood to be profiled or marginalised over all nuisance parameters simultaneously, ensuring that the combined analysis is statistically consistent and that parameter correlations are correctly propagated across experiments.

* **Consistent parameter handling**: All physics and nuisance parameters across every configured experiment are collected into a single, flat NamedTuple via `get_params` and `get_priors`, with `safe_merge` enforcing that shared parameters are represented exactly once and conflict-free. When two experiments share a physics parameter under different names, a Wrapper renames parameters seen by the outer analysis while transparently translating back to the original names before calling the inner experiment's forward_model and plot functions. Prior distributions, including correlated priors via covariance matrices, are managed in the same unified structure and compose naturally into a joint prior over the full parameter space. Individual parameters can be overridden, fixed, or conditionally marginalised at analysis time without modifying the underlying experiment or physics modules, keeping the configuration transparent and reproducible.

* **Automatic differentiation**: Since the package is written entirely in Julia, the complete forward model chain - from oscillation probability calculation to likelihood evaluation - is compatible with *ForwardDiff.jl* out of the box. Consequently, Gradients can be computed accurately and efficiently, enabling advanced statistical methods and optimization techniques for inference, such as provided in [@Schulz:2021BAT].

* **Scalability**: The package is built for large-scale inference tasks, making it suitable for modern global fit applications featuring tens of experiments and hundreds of parameters. Julia's just-in-time compilation delivers high performance without sacrificing expressiveness. Profile likelihood scans parallelise transparently across CPU threads or distributed workers, and the efficient forward-mode AutoDiff gradient computation keeps gradient-based inference tractable even as the nuisance parameter space grows with additional experiments.

* **Sensitivity projections**: Asimov datasets — pseudo-data matching the model prediction — can be generated for any configured experiment via generate_asimov_data, enabling projected sensitivity studies and expected-limit calculations under any signal hypothesis without requiring real observations. Combined with the profile likelihood or Bayesian sampling tools, this supports both the design of future experiments and the end-to-end validation of analysis pipelines on mock data prior to unblinding.

# Workflow Overview
Newtrinos.jl is designed as a modular pipeline where components communicate strictly through functional interfaces. Physics modules provide callables that experiments use inside their callable forward models. Experiments expose the forward model callables, which are then used by the analysis layer to build the joint likelihood - without ever inspecting the internals of either.
A typical analysis with *Newtrinos.jl* proceeds in the following steps:

1. **Configure physics**: Select and instantiate a theoretical model (e.g., standard three-flavour oscillations, sterile neutrinos, or non-standard interactions) by composing the relevant physics modules: oscillation model, Earth density model, atmospheric flux, and cross-section model.

2. **Configure experiments**: Select one or more experiment modules, each encapsulating its detector response, systematic uncertainties, and observed data. Instantiate them by calling the `configure` method of each experiment with the specified physics model. Experiments are independent components and can be freely combined.

3. **Build the joint likelihood**: Pass the collection of configured experiments to `generate_likelihood`, which composes their individual forward models and likelihood functions into a single joint likelihood.

4. **Collect parameters and priors**: Use `get_params` and `get_priors` to automatically collect and merge all physics and nuisance parameters across experiments into a unified NamedTuple. Individual priors can be overridden or fixed to constant values at this stage.

5. **Run inference**: Choose an analysis method from the provided analysis tools. For example, run a profile likelihood scan by calling the `profile` method on the joint likelihood and a chosen parameter grid and obtain a `NewtrinosResult` containing the scan grid coordinates, the likelihood values and optimized nuisance parameters at each point, and meta data for the run. 

6. **Visualize and export results**: Plot confidence contours or best-fit data/MC comparisons using the built-in plotting utilities, and save results to disk for later use.

# Availability
*Newtrinos.jl* is open-source and freely available under the MIT-License. It can be downloaded from the GitHub repository available at https://github.com/philippeller/Newtrinos.jl .

# Related Work
Several software projects in the community address related but distinct use cases:

* **GLoBES** [@Huber:2004ka; @Huber:2007ji]: Simulates long-baseline experiments but does not support full global fits.
* **GAMBIT** [@GAMBIT:2017yxo]: A general-purpose global fitting framework with some support for neutrino data, but not tailored for neutrino physics.
* **PEANUTS** [@Gonzalo:2023mdh]: Focused on solar neutrino modeling.
* **PhyLiNO** [@Hellwig:2025jxe]: High-performance framework developed for reactor neutrino data.
* **PISA** [@IceCube:2018ikn]: Designed for atmospheric neutrino analyses.

*Newtrinos.jl* complements these efforts by focusing specifically on the global fitting aspect, offering a simple, extensible, and efficient design. It provides methods for analysing the full neutrino sector enabling analyses for all kind of neutrino experiments. Futhermore, it is currently the only framework in this domain that supports automatic differentiation. The software has been used in the results presented in [@Ettengruber:2024fcq], [@Kozynets:2024xgt], [@Eller:2025lsh], and [eller2026atmosphericneutrinooscillationspicture].

# Acknowledgements
This work was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy– EXC-2094– 390783311, and the SFB 1258– 283604770.