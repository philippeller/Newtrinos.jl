---
title: 'Newtrinos.jl: A Julia Package for Global Neutrino Oscillation Fits'
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
*Newtrinos.jl* is a Julia package for performing global fits of neutrino oscillation data. It is designed to be easy to use, fast, and flexible. The software allows researchers to combine data from different experiments and test a variety of theoretical models. It supports automatic differentiation and is written entirely in Julia, making it suitable for modern scientific workflows.

# Statement of Need
Global fits of neutrino oscillation data are essential for understanding neutrino properties and testing new physics [@Capozzi:2025wyn; @Esteban:2024eli; @deSalas:2020pgw]. These existing studies rely on proprietary, closed-source implementations, which makes their results difficult to verify and the software impossible to reuse. Some of these efforts also face technical limitations that restrict the complexity of datasets they can include. For example, the analysis in [@Capozzi:2025wyn] excludes atmospheric neutrino experiments entirely due to such limitations.
Another recent study highlights the necessity of combining different datasets in performing a first joint fit between the T2K and the NOvA experiment [@T2K:2025wet].
It is likely, that urgent questions in the field, such as the Neutrino Mass Ordering or CP violation in the lepton sector, can be addressed in global fits in the coming years.

# Key Features
*Newtrinos.jl* takes a different approach compared to established global fit frameworks. It is fully open-source, modular, and computationally efficient. Written entirely in the Julia programming language, it offers high performance and a clear structure optimized for combining diverse theoretical models and experimental datasets.
Key features include:

* **Out-of-the-box usability**: The package includes experimental data, configuration files, and plotting tools, making it easy to reproduce results and verify correctness.
* **Modular architecture**: Experiments, physics models, and analysis methods are implemented as independent components with mutually separated concerns. This design philoshopy allows that a large number of experiments can be added, combined, and tested against arbitrary theories without modifying core code.
* **Full likelihood support**: Analyses are based on complete likelihood functions that include all relevant systematic uncertainties, improving the reliability and flexibility of fits.
* **Consistent parameter handling**: Model parameters and priors, including correlations, are managed in a unified and transparent way.
* **Automatic differentiation**: Gradients can be computed accurately and efficiently, enabling advanced statistical methods and optimization techniques for inference, such as provided in [@Schulz:2021BAT].
* **Scalability**: Built for speed and large-scale inference tasks, making it suitable for modern global fit applications featuring tens of experiments and hundrets of parameters.

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