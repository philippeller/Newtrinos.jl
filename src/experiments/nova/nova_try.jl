module nova

using LinearAlgebra
using Distributions
using UnROOT
using BAT
using DataStructures
using CairoMakie
using Logging
using Statistics
import ..Newtrinos

@kwdef struct novaExperiment <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

# Helper function to safely extract data from ROOT histograms
function extract_histogram_data(hist)
    """Safely extract bin contents from ROOT histogram"""
    if hasfield(typeof(hist), :fArray)
        if hasfield(typeof(hist), :fN) && hist.fN > 0
            return hist.fArray[1:hist.fN]
        else
            return hist.fArray
        end
    else
        # Alternative access method for different ROOT histogram types
        return hist.weights
    end
end

function extract_histogram_edges(hist)
    """Extract bin edges from ROOT histogram axis"""
    if hasfield(typeof(hist), :fXaxis)
        axis = hist.fXaxis
        if hasfield(typeof(axis), :fXbins) && !isempty(axis.fXbins.fArray)
            return axis.fXbins.fArray
        else
            # Uniform binning case
            nbins = axis.fNbins
            xmin = axis.fXmin
            xmax = axis.fXmax
            return range(xmin, xmax, length=nbins+1)
        end
    else
        throw(ArgumentError("Cannot extract edges from histogram"))
    end
end

function configure(physics)
    physics = (;physics.osc, physics.xsec)
    assets = get_assets(physics)
    return novaExperiment(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_assets(physics; datadir = @__DIR__)
    @info "Loading NOvA data"
    
    # Load data from ROOT files (as in original Python code)
    data_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_histograms.root"))
    mc_file = ROOTFile(joinpath(datadir, "NOvA_2020_data_release_predictions_with_systs_all_hists.root"))
    
    # Energy binning for electron neutrino analysis
    energy_edges = range(0.5, 4.5, length=9)
    
    # Load electron neutrino data
    nue_data = load_nue_data(data_file, mc_file, energy_edges)
    nuebar_data = load_nuebar_data(data_file, mc_file, energy_edges)
    
    # Load muon neutrino data by quartiles
    numu_data, numubar_data = load_numu_data(data_file, mc_file)

   

     observed_nested = (
         nue_data = nue_data.observed,
         nuebar_data = nuebar_data.observed, 
         numu_data = numu_data.observed,
         numubar_data = numubar_data.observed    
      )

     
    # Flatten observed data to match forward model structure
    observed_flat = Vector{Float64}()
    # NUE segments (same order as forward model)
    #@info "NUE segment 1 length: $(length(nue_data.observed.segment1))"
    append!(observed_flat, nue_data.observed.segment1)
    #@info "NUE segment 2 length: $(length(nue_data.observed.segment2))"
    append!(observed_flat, nue_data.observed.segment2)
    #@info "NUE segment 3 length: $(length(nue_data.observed.segment3))"
    append!(observed_flat, nue_data.observed.segment3)
    
    # NUEBAR segments (same order as forward model)
    #@info "NUEBAR segment 1 length: $(length(nuebar_data.observed.segment1))"
    append!(observed_flat, nuebar_data.observed.segment1)
    #@info "NUEBAR segment 2 length: $(length(nuebar_data.observed.segment2))"
    append!(observed_flat, nuebar_data.observed.segment2)
    #@info "NUEBAR segment 3 length: $(length(nuebar_data.observed.segment3))"
    append!(observed_flat, nuebar_data.observed.segment3)
    
    # NUMU observed data - explicit loop over 4 quartiles (matching forward model)
  
    #@info "NUMU observed type: $(typeof(numu_obs)), length: $(length(numu_obs))"
    for i in 1:4
        quartile_obs = numu_data.quartiles[i]["observed"]
       # @info "  NUMU observed quartile $i: $(typeof(quartile_obs)), length: $(length(quartile_obs))"
        append!(observed_flat, quartile_obs)
    end
    
    # NUMUBAR observed data - explicit loop over 4 quartiles (matching forward model)
    
    #@info "NUMUBAR observed type: $(typeof(numubar_obs)), length: $(length(numubar_obs))"
    for i in 1:4
        quartile_obs = numubar_data.quartiles[i]["observed"]
     #   @info "  NUMUBAR observed quartile $i: $(typeof(quartile_obs)), length: $(length(quartile_obs))"
        append!(observed_flat, quartile_obs)
    end
    
    #@info "Total observed data length: $(length(observed_flat))"
    
    observed= observed_flat

    # NOvA baseline and matter density
    L = 810.0  # km
    density = 2.84 * 0.5  # g/cm³ * Z/A ratio
    
    assets = (
        L = L,
        density = density,
        energy_edges = energy_edges,
        nue_data = nue_data,
        nuebar_data = nuebar_data,
        numu_data = numu_data,
        numubar_data = numubar_data,
        # Smearing parameters for each quartile
        numu_smearing = [0.078, 0.092, 0.104, 0.115],
        numubar_smearing = [0.085, 0.089, 0.097, 0.102],
        # Energy scale and bias parameters
        numu_e_scale = 1.05,
        numu_e_bias = 0.0,
        nue_e_scale = 0.65,
        nue_e_bias = 0.02,
        observed = observed
    )
    
    # Files are automatically closed by UnROOT when going out of scope
    return assets
end

function load_nue_data(data_file, mc_file, energy_edges)
    """Load electron neutrino data with 3 segments - robust version"""
    
    println("=== LOADING NUE DATA ===")
    
    # Get neutrino data histogram
    if !haskey(data_file, "neutrino_mode_nue")
        error("Key 'neutrino_mode_nue' not found in data file. Available keys: $(keys(data_file))")
    end
    
    data_hist = data_file["neutrino_mode_nue"]
    
    # Extract observed data
    
    if haskey(data_hist, :fN)
        data_values = data_hist[:fN]
        println("Using :fN for neutrino data")
    else
        error("Cannot find data in neutrino histogram. Available keys: $(keys(data_hist))")
    end
    
    # Extract segments (matching Python logic)
    observed1 = data_values[2:9]      # 8 elements
    observed2 = data_values[11:18]    # 8 elements  
    observed3 = vcat(data_values[19:21], zeros(5))  # 8 elements: 3 data + 5 zeros
    
    # Load Monte Carlo components for FHC 
    mc_components = Dict{String, Vector{Float64}}()  
    
    if haskey(mc_file, "prediction_components_nue_fhc")
        mc_dir = mc_file["prediction_components_nue_fhc"]
        println("Found MC directory: prediction_components_nue_fhc")
        
        for component_name in keys(mc_dir)
            #println("Processing neutrino MC component: ", component_name)
            component_hist = mc_dir[component_name]
            
            # Extract MC data
            mc_values = nothing
            
            if haskey(component_hist, :fN) && length(component_hist[:fN]) > 0
                mc_values = component_hist[:fN]
                println("Using :fN field, length: $(length(mc_values))")
            else
                println("No suitable data field found for neutrino component ", component_name)
                continue
            end
            
            if length(mc_values) >= 21
                clean_name = string(component_name)
                
                mc_components[clean_name * "1"] = mc_values[1:8]
                mc_components[clean_name * "2"] = mc_values[9:16]
                mc_components[clean_name * "3"] = vcat(mc_values[17:21], zeros(3))  # 8 elements: 5 data + 3 zeros
                
                println("Created: $(clean_name)1, $(clean_name)2, $(clean_name)3")
            else
                println("MC values too short: $(length(mc_values)) (expected >= 21)")
            end
        end
    else
        println("Warning: 'prediction_components_nue_fhc' not found in MC file")
       # println("Available MC keys: ", keys(mc_file))
    end
    
   # println("Created NUE MC components: $(keys(mc_components))")
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
 )

end

function load_nuebar_data(data_file, mc_file, energy_edges)
    """Load antineutrino data with 3 segments - robust version"""
    
    println("=== LOADING NUEBAR DATA ===")
    
    # Get antineutrino data histogram
    if !haskey(data_file, "antineutrino_mode_nue")
        error("Key 'antineutrino_mode_nue' not found in data file. Available keys: $(keys(data_file))")
    end
    
    data_hist = data_file["antineutrino_mode_nue"]
   
    if haskey(data_hist, :fN)
        data_values = data_hist[:fN]
        println("Using :fN for antineutrino data")
    else
        error("Cannot find data in antineutrino histogram. Available keys: $(keys(data_hist))")
    end
  
    observed1 = data_values[2:9]      # 8 elements
    observed2 = data_values[11:18]    # 8 elements  
    observed3 = vcat(data_values[19:21], zeros(5))  # 8 elements: 3 data + 5 zeros
    
    # Load Monte Carlo components for RHC 
    mc_components = Dict{String, Vector{Float64}}()  

    if haskey(mc_file, "prediction_components_nue_rhc")
        mc_dir = mc_file["prediction_components_nue_rhc"]
        println("Found MC directory: prediction_components_nue_rhc")
        
        for component_name in keys(mc_dir)
           
            component_hist = mc_dir[component_name]
            
            # Extract MC data 
            mc_values = nothing
            
            if haskey(component_hist, :fN) && length(component_hist[:fN]) > 0
                mc_values = component_hist[:fN]
                println("Using :fN field, length: $(length(mc_values))")
        
            else
                println("No suitable data field found for antineutrino component ", component_name)
                continue
            end
            
            if length(mc_values) >= 21
                clean_name = string(component_name)
                
               
                mc_components[clean_name * "1"] = mc_values[1:8]
                mc_components[clean_name * "2"] = mc_values[9:16]
                mc_components[clean_name * "3"] = vcat(mc_values[17:21], zeros(3))  # 8 elements: 5 data + 3 zeros

                println("Created: $(clean_name)1, $(clean_name)2, $(clean_name)3")
            else
                println(" MC values too short: $(length(mc_values)) (expected >= 21)")
            end
        end
    else
        println("Warning: 'prediction_components_nue_rhc' not found in MC file")
      #  println("Available MC keys: ", keys(mc_file))
    end
    
    #println("Created NUEBAR MC components: $(keys(mc_components))")
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )

end


function load_numu_data(data_file, mc_file)
    """Load muon neutrino data by quartiles with proper ROOT histogram handling"""
    
    println("=== LOADING NUMU DATA ===")
  
    # Get energy binning from first quartile
    first_hist = data_file["neutrino_mode_numu_quartile1"]
    energy_edges = extract_energy_edges(first_hist)
    n_bins = length(energy_edges) - 1
    
    #println(" Energy binning: $(n_bins) bins from $(energy_edges[1]) to $(energy_edges[end]) GeV")
    
    # Initialize storage
    expected_components = ["NoOscillations_Signal", "Oscillated_Signal", "NoOscillations_Total_beam_bkg", "Cosmic_bkg","Oscillated_Total_pred"]
    numu_total = Dict(comp => zeros(Float64, n_bins) for comp in expected_components)
    numubar_total = Dict(comp => zeros(Float64, n_bins) for comp in expected_components)
    numu_observed_total = zeros(Float64, n_bins)  
    numubar_observed_total = zeros(Float64, n_bins)  

    numu_quartiles = []
    numubar_quartiles = []
    
    # Process each quartile

    for q in 1:4
        # Load neutrino data
        neutrino_data = load_quartile_data(data_file, mc_file, q, "neutrino", n_bins)
        push!(numu_quartiles, neutrino_data)
        
        # Load antineutrino data  
        antineutrino_data = load_quartile_data(data_file, mc_file, q, "antineutrino", n_bins)
        push!(numubar_quartiles, antineutrino_data)
        
        # Accumulate totals
        accumulate_totals!(numu_total, neutrino_data, expected_components, "neutrino", q)
        accumulate_totals!(numubar_total, antineutrino_data, expected_components, "antineutrino", q)
    
    end
        
    numu_observed_total = sum([q["observed"] for q in numu_quartiles])
    numubar_observed_total = sum([q["observed"] for q in numubar_quartiles])
    
    # Return results
    numu_data = (
        quartiles = numu_quartiles,
        total = numu_total,
        observed = numu_observed_total,  # Average observed data across quartiles
        energy_edges = energy_edges

    )
    
    numubar_data = (
        quartiles = numubar_quartiles,
        total = numubar_total,
        observed = numubar_observed_total,  
        energy_edges = energy_edges
    )
    
    return numu_data, numubar_data
end


function extract_energy_edges(histogram)
    """Extract energy bin edges from ROOT histogram"""
    
    if haskey(histogram, :fXaxis_fXbins)
        return histogram[:fXaxis_fXbins]
    end
end

function extract_histogram_contents(histogram, hist_name)
    """Extract bin contents from ROOT histogram - specifically targeting fN field"""
    
    # Try to extract fN field
    if haskey(histogram, :fN)
        data = histogram[:fN]
        if isa(data, AbstractVector{<:Real}) && !isempty(data)
            println("Found :fN field: length=$(length(data)), sum=$(sum(data))")
            return Float64.(data), "fN"
        else
            println(" Found :fN but it's not a valid numeric vector: $(typeof(data))")
        end
    end
end

function extract_physics_bins(raw_data, n_bins, source_name)
    """Extract physics bins from ROOT histogram data"""
    
    raw_length = length(raw_data)
    
    if raw_length == n_bins + 2
        # Standard ROOT format: [underflow, bin1, bin2, ..., binN, overflow]
        physics_data = raw_data[2:end-1]
        println("    → Extracted $(length(physics_data)) physics bins (skipped under/overflow)")
    elseif raw_length == n_bins
        # Just physics bins
        physics_data = raw_data
        println("    → Using all $(length(physics_data)) bins as physics bins")
    elseif raw_length > n_bins
        # Take first n_bins (assuming no underflow bin)
        physics_data = raw_data[1:n_bins]
        println("    → Extracted first $(length(physics_data)) bins from $(raw_length) total")
    else
        # Too few bins - pad with zeros
        physics_data = vcat(raw_data, zeros(n_bins - raw_length))
        println("   Only $(raw_length) bins available, padded to $(n_bins)")
    end
    
    return physics_data
end

function load_quartile_data(data_file, mc_file, quartile, mode, n_bins)
    """Load observed and MC data for a single quartile"""
    
    mode_prefix = mode == "neutrino" ? "neutrino_mode" : "antineutrino_mode"
    beam_mode = mode == "neutrino" ? "fhc" : "rhc"
    
    quartile_data = Dict{String, Any}()
    
    # Load observed data
    obs_hist_key = "$(mode_prefix)_numu_quartile$(quartile)"
    
    if haskey(data_file, obs_hist_key)
        obs_hist = data_file[obs_hist_key]

        if haskey(obs_hist, :fN)
            data_values = obs_hist[:fN]
            observed_data = data_values[2:end-1] 
            println("Direct extraction: length=$(length(observed_data)), sum=$(sum(observed_data))")
        end
        
        quartile_data["observed"] = observed_data
       
    end
    # Load MC components
    mc_key = "prediction_components_numu_$(beam_mode)_Quartile$(quartile)"
    
    
    if haskey(mc_file, mc_key)
        mc_dir = mc_file[mc_key]
        
        if isa(mc_dir, UnROOT.ROOTDirectory)
         
            for component_name in keys(mc_dir)
                component_hist = mc_dir[component_name]
                
                
                    if haskey(component_hist, :fN)
                        data_values = component_hist[:fN]
                        mc_data = data_values[2:end-1]
                        println("Direct MC extraction for $(component_name): length=$(length(mc_data)), sum=$(sum(mc_data))")
                        # Add this debug code in your load_quartile_data function:
                        println("Quartile $quartile, Component: NoOscillations_Total_beam_bkg, Sum: $(sum(mc_data))")
                    end
                    
            
        
                 quartile_data[string(component_name)] = mc_data
            
            end
        end
    
    end

    first_hist = data_file["neutrino_mode_numu_quartile1"]
    energy_edges = extract_energy_edges(first_hist)
    n_bins = length(energy_edges) - 1
    quartile_data["energy_edges"] = energy_edges  # Pass this as parameter

    # Check energy binning info
    println("Energy edges: $(quartile_data["energy_edges"])")
    println("Number of energy bins: $(length(quartile_data["energy_edges"]) - 1)")

    return quartile_data
end



function accumulate_totals!(totals_dict, quartile_data, expected_components, mode, quartile)
    """Accumulate quartile data into totals"""
    
    for component in expected_components
        if haskey(quartile_data, component)
            data_vec = quartile_data[component]
            if length(data_vec) == length(totals_dict[component])
                totals_dict[component] .+= data_vec
                component_sum = sum(data_vec)
                if component_sum > 0
                    println(" Added $(component) to $(mode) total: +$(component_sum)")
                end
            end
       end
    end   
end


function smearnorm(energies, probabilities, percent, width=10, e_scale=1.0, e_bias=0.0)
    """
    Apply energy resolution smearing to oscillation probabilities.
    Convolves probabilities with a boxcar function to model detector resolution.
    """
    n = length(probabilities)
    out = zeros(n)
    
    for i in 1:n
        norm = 0.0
        for j in max(1, i - width):min(n, i + width)
            coeff = 1.0
            norm += coeff
            out[i] += coeff * probabilities[j]
        end
        out[i] /= norm
    end
    
    return out
end

function calculate_energy_edges(energy_centers)

    """Calculate bin edges from bin centers for logarithmic binning"""
    
    if length(energy_centers) < 2
        throw(ArgumentError("Need at least 2 energy centers"))
    end
    if any(energy_centers .<= 0)
        throw(ArgumentError("Energy centers must be positive"))
    end
    
    edges = zeros(length(energy_centers) + 1)
    
    # Interior edges as geometric mean
    edges[2:end-1] = sqrt.(energy_centers[2:end] .* energy_centers[1:end-1])
    
    # First and last edges maintaining logarithmic spacing
    log_spacing = log(energy_centers[2] / energy_centers[1])
    edges[1] = energy_centers[1] / exp(log_spacing/2)
    edges[end] = energy_centers[end] * exp(log_spacing/2)
    
    return edges
end

function rebin_energy_spectrum(input_data, edges, e_min=0.5, e_max=4.5, num_bins=8)
    """
    Rebin spectrum from irregular energy bins to regular bins.
    Handles partial bin overlaps properly.
    """
    # Handle masked arrays or missing data
    data = replace(input_data, missing => 0.0, NaN => 0.0)
    
    # Convert edges to array if needed
    edge_values = isa(edges, AbstractArray) ? edges : collect(edges)
    
    # Create new equally spaced bin edges
    new_edges = range(e_min, e_max, length=num_bins + 1)
    new_counts = zeros(num_bins)
    
    # For each input bin
    for i in 1:(length(edge_values)-1)
        old_e_low = edge_values[i]
        old_e_high = edge_values[i+1]
        old_width = old_e_high - old_e_low
        
        # Skip bins outside range
        if old_e_high < e_min || old_e_low > e_max
            continue
        end
        
        # For each new bin
        for j in 1:num_bins
            new_e_low = new_edges[j]
            new_e_high = new_edges[j+1]
            
            # Calculate overlap
            overlap_low = max(old_e_low, new_e_low)
            overlap_high = min(old_e_high, new_e_high)
            
            if overlap_high > overlap_low
                overlap = overlap_high - overlap_low
                fraction = overlap / old_width
                new_counts[j] += data[i] * fraction
            end
        end
    end
    
    return new_counts
end

function fast_predictions_new(signal, backgrounds, norm_factor=1; condense_to_bin3=false)
    """
    Efficiently combine signal and background components with normalization.
    Option to condense all values into bin 3 for systematic uncertainties.
    """
    total = copy(signal)
    
    # Add backgrounds
    for bg in backgrounds
        total .+= bg
    end
    
    # Apply normalization
    total .*= norm_factor
    
    # Condense to bin 3 if requested
    if condense_to_bin3 && length(total) > 2
        condensed_sum = sum(total)
        fill!(total, 0.0)
        total[3] = condensed_sum
    end
    
    return total
end

function make_numu_predictions(params, physics, assets)
    """Calculate muon neutrino disappearance predictions for all quartiles"""

    # At the beginning, check what you're getting
  
    L = [assets.L]
    density = [assets.density]
    
    # Energy grid for oscillation calculation
    energy_grid = exp.(range(log(0.1), log(10.0), length=100))
    
    # Calculate oscillation probabilities for neutrinos
    p_nu = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                               L, params; anti=false)
    p_nu_survival = p_nu[:, 1, 2, 2]  # νμ → νμ survival probability
    
    # Calculate for antineutrinos
    p_nubar = physics.osc.osc_prob(energy_grid * assets.numu_e_scale .+ assets.numu_e_bias, 
                                  L, params; anti=true)
    p_nubar_survival = p_nubar[:, 1, 2, 2]  # ν̄μ → ν̄μ survival probability
    
    # Apply smearing and make predictions for each quartile
    predictions = Dict{String, Vector{Vector{Float64}}}()
    predictions["numu"] = []
    predictions["numubar"] = []
    
    for i in 1:4

        p_smeared = smearnorm(energy_grid, p_nu_survival, assets.numu_smearing[i], 4, 
                            assets.numu_e_scale, assets.numu_e_bias)
        
        # Rebin to detector energy bins
        quartile_data = assets.numu_data.quartiles[i]
        
            # Neutrino quartile

        p_rebinned = rebin_energy_spectrum(p_smeared, energy_grid, 0.5, 10.0, 
                                         length(quartile_data["observed"]))
        p_rebinned ./= sum(p_rebinned)  # Normalize
        
        # Calculate prediction
        prediction = (quartile_data["NoOscillations_Signal"] .* p_rebinned .+
                     quartile_data["NoOscillations_Total_beam_bkg"] .+
                     quartile_data["Cosmic_bkg"]) 
        push!(predictions["numu"], prediction)
        # Antineutrino quartile
        p_smeared_bar = smearnorm(energy_grid, p_nubar_survival, assets.numubar_smearing[i], 4,
                                assets.numu_e_scale, assets.numu_e_bias)
        
        quartile_data_bar = assets.numubar_data.quartiles[i]
        p_rebinned_bar = rebin_energy_spectrum(p_smeared_bar, energy_grid, 0.5, 10.0,
                                             length(quartile_data_bar["observed"]))
        p_rebinned_bar ./= sum(p_rebinned_bar)
        
        prediction_bar = (quartile_data_bar["NoOscillations_Signal"] .* p_rebinned_bar .+
                         quartile_data_bar["NoOscillations_Total_beam_bkg"] .+
                         quartile_data_bar["Cosmic_bkg"]) 
        push!(predictions["numubar"], prediction_bar)
      
    end
    
    return predictions
end

function make_nue_predictions(params, physics, assets)
    """Calculate electron neutrino appearance predictions"""
    
    L = [assets.L]
    density = [assets.density]
    
    # Energy grid for oscillation calculation
    energy_grid = range(0.5, 4.5, length=100)
    energy_edges = calculate_energy_edges(energy_grid)
    
    # Calculate νμ → νe oscillation probabilities
    p_nu = physics.osc.osc_prob(energy_grid * assets.nue_e_scale .+ assets.nue_e_bias,
                               L, params; anti=false)
    p_nu_appearance = p_nu[:, 1, 2, 1]  # νμ → νe probability
    
    p_nubar = physics.osc.osc_prob(energy_grid * assets.nue_e_scale .+ assets.nue_e_bias,
                                  L, params; anti=true)
    p_nubar_appearance = p_nubar[:, 1, 2, 1]  # ν̄μ → ν̄e probability
    
    # Apply smearing
    p_nu_smeared = smearnorm(energy_grid, p_nu_appearance, 1.0, 5, 
                           assets.nue_e_scale, assets.nue_e_bias)
    p_nubar_smeared = smearnorm(energy_grid, p_nubar_appearance, 1.0, 5,
                              assets.nue_e_scale, assets.nue_e_bias)
    
    # Calculate signal from muon neutrino flux
    numu_total_bins = length(assets.numu_data.total["NoOscillations_Signal"])
    signal_nu = rebin_energy_spectrum(p_nu_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nu ./= sum(signal_nu)
    signal_nu .*= assets.numu_data.total["NoOscillations_Signal"]
    
    signal_nubar = rebin_energy_spectrum(p_nubar_smeared, energy_edges, 0.5, 4.5, numu_total_bins)
    signal_nubar ./= sum(signal_nubar)
    signal_nubar .*= assets.numubar_data.total["NoOscillations_Signal"]
    
    # Rebin to electron neutrino analysis bins
    signal_nue = rebin_energy_spectrum(signal_nu, assets.numu_data.energy_edges, 0.5, 4.5, 8)
    signal_nuebar = rebin_energy_spectrum(signal_nubar, assets.numubar_data.energy_edges, 0.5, 4.5, 8)
    
    # Make predictions for each segment
    predictions = Dict{String, Dict{String, Vector{Float64}}}()
    predictions["nue"] = Dict{String, Vector{Float64}}()
    predictions["nuebar"] = Dict{String, Vector{Float64}}()
        # Add this line before the error at line 730
   # println("Available NUE MC keys: ", keys(assets.nue_data.mc_components))
    #println("Looking for key: Wrong_sign_bkg1")
    
    for segment in 1:3
        # Neutrino segment
        backgrounds_nu = [
            assets.nue_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nue_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nue_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        condense_mode = (segment == 3)
        prediction_nu = fast_predictions_new(signal_nue, backgrounds_nu,
                                           condense_to_bin3=condense_mode)
        
        predictions["nue"]["segment$(segment)"] = prediction_nu
        
        # Antineutrino segment
        backgrounds_nubar = [
            assets.nuebar_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nuebar_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nuebar_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        prediction_nubar = fast_predictions_new(signal_nuebar, backgrounds_nubar, 
                                              condense_to_bin3=condense_mode)
        
        predictions["nuebar"]["segment$(segment)"] = prediction_nubar
    end
    
    return predictions
end

function get_forward_model(physics, assets)

    function forward_model(params)
        exp_events_numu = make_numu_predictions(params, physics, assets)
        exp_events_nue = make_nue_predictions(params, physics, assets)
        
        # everything into a single vector
        all_predictions = Vector{Float64}()
        
        # NUE segments 
        append!(all_predictions, exp_events_nue["nue"]["segment1"])
        append!(all_predictions, exp_events_nue["nue"]["segment2"])
        append!(all_predictions, exp_events_nue["nue"]["segment3"])
        
        # NUEBAR segments 
        append!(all_predictions, exp_events_nue["nuebar"]["segment1"])
        append!(all_predictions, exp_events_nue["nuebar"]["segment2"])
        append!(all_predictions, exp_events_nue["nuebar"]["segment3"])
        
            
        # NUMU data - loop over 4 quartiles directly
        numu_data = exp_events_numu["numu"]
        #@info "NUMU data type: $(typeof(numu_data)), length: $(length(numu_data))"
        for i in 1:4
            quartile_data = numu_data[i]
         #    @info "  NUMU quartile $i: $(typeof(quartile_data)), length: $(length(quartile_data))"
            append!(all_predictions, quartile_data)
        end
        
        # NUMUBAR data - loop over 4 quartiles directly  
        numubar_data = exp_events_numu["numubar"]
       # @info "NUMUBAR data type: $(typeof(numubar_data)), length: $(length(numubar_data))"
        for i in 1:4
            quartile_data = numubar_data[i]
         #   @info "  NUMUBAR quartile $i: $(typeof(quartile_data)), length: $(length(quartile_data))"
            append!(all_predictions, quartile_data)
        end
        
        #@info "Total predictions length: $(length(all_predictions))"
     
        
        # Return vector of Poisson distributions
        return Poisson.(max.(all_predictions, 1e-10))
    end
    return forward_model
end


function get_plot(physics, assets)

    function plot(params)
        f = Figure(resolution=(1400, 1200)) 
        
        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)
        
        # Plot muon neutrino disappearance by quartile
        for i in 1:4
            ax = Axis(f[1, i], title="νμ Quartile $(i)")
            
            observed = assets.numu_data.quartiles[i]["observed"]
            predicted = numu_predictions["numu"][i]
            predicted_errors = sqrt.(max.(predicted, 0))  # Poisson errors for predictions
            energy_centers = (assets.numu_data.energy_edges[1:end-1] .+ 
                            assets.numu_data.energy_edges[2:end]) ./ 2
            
            # Get MC histogram data directly
            mc_histogram = assets.numu_data.quartiles[i]["Oscillated_Total_pred"]
            
            # Plot ALL THREE components
            scatter!(ax, energy_centers, observed, color=:black, label="Observed Data", markersize=6)
            stairs!(ax, energy_centers, mc_histogram, color=:gray, linewidth=3, label="MC Histogram")
            errorbars!(ax, energy_centers, predicted, predicted_errors, color=:red, linewidth=2)
            lines!(ax, energy_centers, predicted, color=:red, linewidth=2, label="Calculated Prediction")
            
            ax.xlabel = "Energy (GeV)"
            ax.ylabel = "Events"
            axislegend(ax, position=:rt, labelsize=4)
        end

        # Plot muon antineutrino disappearance by quartile
        for i in 1:4
            ax_bar = Axis(f[2, i], title="ν̄μ Quartile $(i)")

            observed = assets.numubar_data.quartiles[i]["observed"]
            predicted = numu_predictions["numubar"][i]
            predicted_errors = sqrt.(max.(predicted, 0))  # Poisson errors for predictions
            energy_centers = (assets.numubar_data.energy_edges[1:end-1] .+ 
                            assets.numubar_data.energy_edges[2:end]) ./ 2

            # Get MC histogram data directly for numubar
            mc_histogram = assets.numubar_data.quartiles[i]["Oscillated_Total_pred"]

           
            scatter!(ax_bar, energy_centers, observed, color=:black, label="Observed Data", markersize=6)
            stairs!(ax_bar, energy_centers, mc_histogram, color=:gray, linewidth=3, label="MC Histogram")
            errorbars!(ax_bar, energy_centers, predicted, predicted_errors, color=:red, linewidth=2)
            lines!(ax_bar, energy_centers, predicted, color=:red, linewidth=2, label="Calculated Prediction")

            ax_bar.xlabel = "Energy (GeV)"
            ax_bar.ylabel = "Events"
            axislegend(ax_bar, position=:rt, labelsize=4)
        end
        
      
        ax_numu_total = Axis(f[1, 5], title="νμ Total (All Quartiles)")
        
        # Sum observed data across all quartiles
        observed_total_numu = assets.numu_data.observed
     
        # Sum predicted data across all quartiles
        predicted_total_numu = sum([numu_predictions["numu"][i] for i in 1:4])
        
        predicted_total_numu_errors = sqrt.(max.(predicted_total_numu, 0))
        
        # Get total MC histogram
        mc_total_numu = assets.numu_data.total["Oscillated_Total_pred"]
   
        energy_centers = (assets.numu_data.energy_edges[1:end-1] .+ 
                        assets.numu_data.energy_edges[2:end]) ./ 2
        
        # Plot total components
        scatter!(ax_numu_total, energy_centers, observed_total_numu, color=:black, 
                label="Observed Total", markersize=8)
        stairs!(ax_numu_total, energy_centers, mc_total_numu, color=:gray, 
                linewidth=4, label="MC Total")
        errorbars!(ax_numu_total, energy_centers, predicted_total_numu, 
                predicted_total_numu_errors, color=:red, linewidth=3)
        lines!(ax_numu_total, energy_centers, predicted_total_numu, color=:red, 
            linewidth=3, label="Calculated Total")
        
        ax_numu_total.xlabel = "Energy (GeV)"
        ax_numu_total.ylabel = "Events"
        axislegend(ax_numu_total, position=:rt, labelsize=4)
        
      
        ax_numubar_total = Axis(f[2, 5], title="ν̄μ Total (All Quartiles)")
        
        
        observed_total_numubar = assets.numubar_data.observed

        # Sum predicted data across all quartiles
        predicted_total_numubar = sum([numu_predictions["numubar"][i] for i in 1:4])

        predicted_total_numubar_errors = sqrt.(max.(predicted_total_numubar, 0))
        
        # Get total MC histogram
        mc_total_numubar = assets.numubar_data.total["Oscillated_Total_pred"]
   
        energy_centers = (assets.numubar_data.energy_edges[1:end-1] .+ 
                        assets.numubar_data.energy_edges[2:end]) ./ 2
        
        # Plot total components
        scatter!(ax_numubar_total, energy_centers, observed_total_numubar, color=:black, 
                label="Observed Total", markersize=8)
        stairs!(ax_numubar_total, energy_centers, mc_total_numubar, color=:gray, 
                linewidth=4, label="MC Total")
        errorbars!(ax_numubar_total, energy_centers, predicted_total_numubar, 
                predicted_total_numubar_errors, color=:red, linewidth=3)
        lines!(ax_numubar_total, energy_centers, predicted_total_numubar, color=:red, 
            linewidth=3, label="Calculated Total")
        
        ax_numubar_total.xlabel = "Energy (GeV)"
        ax_numubar_total.ylabel = "Events"
        axislegend(ax_numubar_total, position=:rt, labelsize=4)
        
        # Plot electron neutrino appearance
        for seg in 1:3
            ax_nue = Axis(f[3, seg], title="νμ → νe Segment $seg")
            
            # Get data for this segment
            if seg == 1
                observed_nue = assets.nue_data.observed.segment1
                predicted_nue = nue_predictions["nue"]["segment1"]
                mc_nue = assets.nue_data.mc_components["Total_pred1"]
            elseif seg == 2
                observed_nue = assets.nue_data.observed.segment2
                predicted_nue = nue_predictions["nue"]["segment2"]
                mc_nue = assets.nue_data.mc_components["Total_pred2"]
            else # seg == 3
                observed_nue = assets.nue_data.observed.segment3
                predicted_nue = nue_predictions["nue"]["segment3"]
                mc_nue = assets.nue_data.mc_components["Total_pred3"]
            end
            
            predicted_nue_errors = sqrt.(max.(predicted_nue, 0))  # Poisson errors for predictions
            energy_centers = (assets.nue_data.energy_edges[1:end-1] .+ assets.nue_data.energy_edges[2:end]) ./ 2
        
            energy_centers_m = energy_centers .+ 0.25  

            scatter!(ax_nue, energy_centers, observed_nue, color=:black, label="Observed Data", markersize=6)
            stairs!(ax_nue, energy_centers_m, mc_nue, color=:gray, linewidth=3, label="MC Histogram")
            errorbars!(ax_nue, energy_centers, predicted_nue, predicted_nue_errors, color=:blue, linewidth=2)
            lines!(ax_nue, energy_centers, predicted_nue, color=:blue, linewidth=2, label="Calculated Prediction")

            ax_nue.xlabel = "Energy (GeV)"
            ax_nue.ylabel = "Events"
            axislegend(ax_nue, position=:rt, labelsize=4)
        end
        
        # Plot antineutrino appearance - 3 separate segments  
        for seg in 1:3
            ax_nuebar = Axis(f[4, seg], title="ν̄μ → ν̄e Segment $seg")
            
            # Get data for this segment
            if seg == 1
                observed_nuebar = assets.nuebar_data.observed.segment1
                predicted_nuebar = nue_predictions["nuebar"]["segment1"]
                mc_nuebar = assets.nuebar_data.mc_components["Total_pred1"]
            elseif seg == 2
                observed_nuebar = assets.nuebar_data.observed.segment2
                predicted_nuebar = nue_predictions["nuebar"]["segment2"]
                mc_nuebar = assets.nuebar_data.mc_components["Total_pred2"]
            else # seg == 3
                observed_nuebar = assets.nuebar_data.observed.segment3
                predicted_nuebar = nue_predictions["nuebar"]["segment3"]
                mc_nuebar = assets.nuebar_data.mc_components["Total_pred3"]
            end
            
            predicted_nuebar_errors = sqrt.(max.(predicted_nuebar, 0))  # Poisson errors for predictions
            energy_centers = (assets.nuebar_data.energy_edges[1:end-1] .+ assets.nuebar_data.energy_edges[2:end]) ./ 2
            energy_centers_m = energy_centers .+ 0.25  
            
            scatter!(ax_nuebar, energy_centers, observed_nuebar, color=:black, label="Observed Data", markersize=6)
            stairs!(ax_nuebar, energy_centers_m, mc_nuebar, color=:gray, linewidth=3, label="MC Histogram")
            errorbars!(ax_nuebar, energy_centers, predicted_nuebar, predicted_nuebar_errors, color=:green, linewidth=2)
            lines!(ax_nuebar, energy_centers, predicted_nuebar, color=:green, linewidth=2, label="Calculated Prediction")

            ax_nuebar.xlabel = "Energy (GeV)"
            ax_nuebar.ylabel = "Events"
            axislegend(ax_nuebar, position=:rt, labelsize=4)
        end
        
        return f
    end

 return plot

end    


end  # module Nova