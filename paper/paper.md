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
*Newtrinos.jl* is an open-source Julia package for performing global analyses of neutrino data. It is designed to be easy to use, fast, and flexible. The software allows researchers to combine data from different experiments and test a variety of theoretical models. It also allows adding experiments and theoretical models via Julias multiple dispatch without critical changes to the source code. It supports automatic differentiation and is written entirely in Julia, making it suitable for modern scientific workflows.

# Statement of Need
As neutrino physics transitions into an era of high-precision measurements, global fits of experimental neutrino data have become essential for understanding and extracting neutrino properties and probing new physics beyond the Standard Model [@Capozzi:2025wyn; @Esteban:2024eli; @deSalas:2020pgw]. While single-experiment analyses provide highly precise measurements of specific physical channels, global analyses complement them by systematically combining disparate datasets to break parameter degeneracies and maximize overall statistical sensitivity. Thus, urgent questions in the field, such as the Neutrino Mass Ordering or CP violation in the lepton sector, are also addressed in global fits. Recent efforts underscore this necessity; for example, the first joint fit between the T2K and NOvA experiments demonstrated that combining distinct experimental signatures is crucial to breaking parameter degeneracies [@T2K:2025wet].

Performing a global neutrino fit requires managing the massive computational overhead of calculating oscillation probabilities across diverse, experiment-specific detector models, beam fluxes, and matter density profiles. Furthermore, the process is complicated by the need to rigorously account for a vast number of correlated systematic uncertainties and nuisance parameters for each contained experiment to ensure the statistical consistency of the joint data. On top of that, the relatively bad availability of public data and information regarding the different experiments complicates the process even further. So global fits require huge efforts and thus, analyses sometimes seem to be unfeasible. For example the analysis in [@Capozzi:2025wyn] excludes atmospheric neutrino experiments entirely due to such limitations, stating "For atmospheric neutrinos, currently involving hundreds of bins, dozens of systematic uncertainties, and refined statistical separation of event classes by flavor proxies, the construction of $\chi^2$ maps based on public information has become eventually unfeasible outside the experimental collaborations”.

The existing studies rely on proprietary, closed-source implementations, which makes their results difficult to verify and the software impossible to reuse or adapt. Additionally, within those implementations the experiment definitions, systematic nuisances, and oscillation physics are often deeply intertwined. These frameworks often rely on a multi-language architecture, typically using C++ or Fortran for the performance-critical numerical engine and Python for the user interface, which complicates maintenance and creates language barriers for community contributions and transparency. All these points make the modification of underlying physics models and combination of different experiments within the framework even more difficult.

Another issue is the computational burden associated with managing high-dimensional parameter spaces and correlated systematic uncertainties. As the dimensionality of this parameter space grows with more experiments, traditional statistical methods become increasingly intractable. For instance, executing frequentist profiling or implementing rigorous coverage corrections, such as the Feldman-Cousins procedure, requires performing millions of multi-dimensional likelihood minimizations and each individual minimization step demands the repeated evaluation of neutrino survival and appearance probabilities. There exist new modern gradient based inference algorithms that rely on automatic differentiation (e.g., HMC [@Neal:2011hmc] or MALA [@Roberts:1996mala]) that tackle these problems. However, the existing frameworks typically rely on finite-difference approximations (as employed, e.g., in MINUIT [@James:1975dr]) to calculate gradients during minimization, a technique that scales poorly and becomes prohibitively slow as more nuisance parameters are introduced. Consequently, these frameworks hit a strict computational ceiling.

To address all those critical problems, we introduce *Newtrinos.jl*. Built within the high-performance Julia ecosystem, this package provides a fully open-source, extensible framework explicitly designed to overcome the structural and computational limitations of legacy tools.

# Key Features
*Newtrinos.jl* takes a different approach compared to established global fit frameworks. It is fully open-source, modular, and computationally efficient. Written entirely in the Julia programming language, it offers high performance and a clear structure optimized for combining diverse theoretical models and experimental datasets.
Key features include:

* **Out-of-the-box usability**: The package includes experimental data, configuration files, and plotting tools, making it easy to reproduce results and verify correctness.
* **Modular architecture**: Experiments, physics models, and analysis methods are implemented as independent components with mutually separated concerns. This design philoshopy allows that a large number of experiments can be added, combined, and tested against arbitrary theories without modifying core code.
* **Full likelihood support**: Analyses are based on complete likelihood functions that include all relevant systematic uncertainties, improving the reliability and flexibility of fits.
* **Consistent parameter handling**: Model parameters and priors, including correlations, are managed in a unified and transparent way.
* **Automatic differentiation**: Gradients can be computed accurately and efficiently, enabling advanced statistical methods and optimization techniques for inference, such as provided in [@Schulz:2021BAT].
* **Scalability**: Built for speed and large-scale inference tasks, making it suitable for modern global fit applications featuring tens of experiments and hundrets of parameters.

# Workflow Overview
A typical analysis with *Newtrinos.jl* proceeds in the following steps:

1. **Configure physics**: Select and instantiate a theoretical model (e.g., standard three-flavour oscillations, sterile neutrinos, or non-standard interactions) by composing the relevant physics modules — oscillation engine, Earth density model, atmospheric flux, and cross-section model.

2. **Configure experiments**: Instantiate one or more experiment modules, each encapsulating its detector response, systematic uncertainties, and observed data. Experiments are independent components and can be freely combined.

3. **Build the joint likelihood**: Pass the collection of configured experiments to `generate_likelihood`, which composes their individual forward models and likelihood functions into a single joint likelihood.

4. **Collect parameters and priors**: Use `get_params` and `get_priors` to automatically merge all physics and nuisance parameters across experiments into a unified NamedTuple. Individual priors can be overridden or fixed at this stage.

5. **Run inference**: Choose an analysis method — a parameter scan (grid evaluation), a profile likelihood (per-point nuisance optimization, with optional multithreading or distributed execution), or Bayesian posterior sampling — and obtain a `NewtrinosResult`.

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

*Newtrinos.jl* complements these efforts by focusing specifically on the global fitting aspect, offering a simple, extensible, and efficient design. It is currently the only framework in this domain that supports automatic differentiation. The software has been used in the results presented in [@Ettengruber:2024fcq], [@Kozynets:2024xgt], and [@Eller:2025lsh].

# Acknowledgements
This work was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy– EXC-2094– 390783311, and the SFB 1258– 283604770.