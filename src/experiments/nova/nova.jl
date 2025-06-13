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
        nue_e_bias = 0.02
    )
    
    # Files are automatically closed by UnROOT when going out of scope
    return assets
end

function load_nue_data(data_file, mc_file, energy_edges)
    """Load electron neutrino data with 3 segments"""
    
    # Find the correct key first
    println("Available keys in data_file:")
    data_keys = collect(keys(data_file))
    println(data_keys)
    
    # Look for neutrino data key
    nue_key = nothing
    for key in data_keys
        key_str = string(key)
        if occursin("nue", lowercase(key_str))
            nue_key = key
            println("Found nue key: ", key)
            break
        end
    end
    
    if nue_key === nothing
        # If no "nue" key found, try the first available key for debugging
        nue_key = data_keys[1]
        println("No 'nue' key found, using first key: ", nue_key)
    end
    
    # Read observed data histogram
    data_hist = data_file[nue_key]
    println("Type of data_hist: ", typeof(data_hist))
    println("Keys in data_hist: ", keys(data_hist))
    
    # Extract data values - AVOID using .fN completely
    if haskey(data_hist, :fArray)
        data_values = data_hist[:fArray]
        println("Using symbol key :fArray")
    elseif haskey(data_hist, "fArray")
        data_values = data_hist["fArray"]
        println("Using string key \"fArray\"")
    else
        # Look for any array-like field
        for key in keys(data_hist)
            val = data_hist[key]
            if isa(val, Vector) && length(val) > 20  # Assuming we need at least 21 elements
                data_values = val
                println("Using array field: ", key)
                break
            end
        end
        if !@isdefined(data_values)
            error("Cannot find histogram data array. Available keys: $(keys(data_hist))")
        end
    end
    
    println("Length of data_values: ", length(data_values))
    println("First few values: ", data_values[1:min(10, length(data_values))])
    
    # Extract segments (make sure we have enough data)
    if length(data_values) >= 21
        observed1 = data_values[2:9]
        observed2 = data_values[11:18]  
        observed3 = vcat(data_values[19:21], zeros(5))
    else
        error("Not enough data values. Expected at least 21, got $(length(data_values))")
    end
    
    # Load Monte Carlo components
    mc_components = Dict{String, Vector{Vector{Float64}}}()
    
    println("\nProcessing MC file...")
    mc_keys = collect(keys(mc_file))
    println("Available MC keys: ", mc_keys)
    
    for key in mc_keys
        key_str = string(key)
        if occursin("prediction", lowercase(key_str)) && occursin("nue", lowercase(key_str))
            println("Processing MC key: ", key)
            component_name = replace(key_str, r"_\d+$" => "")
            mc_hist = mc_file[key]
            
            # Extract MC values the same way - NO .fN access
            if haskey(mc_hist, :fArray)
                mc_values = mc_hist[:fArray]
            elseif haskey(mc_hist, "fArray")
                mc_values = mc_hist["fArray"]
            else
                # Look for any vector field
                mc_values = nothing
                for mc_key in keys(mc_hist)
                    val = mc_hist[mc_key]
                    if isa(val, Vector) && length(val) > 20
                        mc_values = val
                        break
                    end
                end
                if mc_values === nothing
                    println("Warning: Cannot find MC data for key $key")
                    continue
                end
            end
            
            if length(mc_values) >= 21
                mc_components[component_name * "1"] = mc_values[1:8]
                mc_components[component_name * "2"] = mc_values[9:16]
                mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
            else
                println("Warning: MC values too short for key $key")
            end
        end
    end
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end

function load_nuebar_data(data_file, mc_file, energy_edges)
    """Load electron antineutrino data with 3 segments"""
    
    # Read observed data from ROOT histogram
    data_hist = data_file["antineutrino_mode_nue;1"]  # Note the ;1 from Python code
    data_values = data_hist.fN > 0 ? data_hist.fArray : data_hist.fArray[1:data_hist.fN]
    
    observed1 = data_values[2:9]
    observed2 = data_values[11:18]
    observed3 = vcat(data_values[19:21], zeros(5))
    
    # Load Monte Carlo components
    mc_components = Dict{String, Vector{Vector{Float64}}}()
    
    # Get all keys from MC file for nue RHC predictions
    for key in keys(mc_file)
        if startswith(string(key), "prediction_components_nue_rhc")
            component_name = replace(string(key), r"_\d+$" => "")
            mc_hist = mc_file[key]
            mc_values = mc_hist.fN > 0 ? mc_hist.fArray : mc_hist.fArray[1:mc_hist.fN]
            
            mc_components[component_name * "1"] = mc_values[1:8]
            mc_components[component_name * "2"] = mc_values[9:16]
            mc_components[component_name * "3"] = vcat(mc_values[17:21], zeros(3))
        end
    end
    
    return (
        observed = (segment1 = observed1, segment2 = observed2, segment3 = observed3),
        mc_components = mc_components,
        energy_edges = energy_edges
    )
end
function load_numu_data(data_file, mc_file)
    """Load muon neutrino data by quartiles"""
    
    numu_quartiles = []
    numubar_quartiles = []
    
    # Initialize totals
    numu_total = Dict{String, Vector{Float64}}()
    numubar_total = Dict{String, Vector{Float64}}()
    
    # Get energy edges from first quartile histogram
    first_quartile_hist = data_file["neutrino_mode_numu_quartile1"]
    
    # Extract energy edges using the correct path (discovered from exploration)
    energy_edges = nothing
    
    # The exploration showed that fXaxis_fXbins is directly on the histogram
    if haskey(first_quartile_hist, :fXaxis_fXbins)
        energy_edges = first_quartile_hist[:fXaxis_fXbins]
        println("✅ Found energy edges in fXaxis_fXbins")
        println("   Length: ", length(energy_edges))
        println("   Values: ", energy_edges)
    elseif haskey(first_quartile_hist, "fXaxis_fXbins")
        energy_edges = first_quartile_hist["fXaxis_fXbins"]
        println("✅ Found energy edges in fXaxis_fXbins (string key)")
        println("   Length: ", length(energy_edges))
        println("   Values: ", energy_edges)
    else
        # Fallback: construct from fXaxis_fNbins, fXaxis_fXmin, fXaxis_fXmax
        println("⚠️  fXaxis_fXbins not found, constructing from axis parameters")
        
        nbins = get(first_quartile_hist, :fXaxis_fNbins, get(first_quartile_hist, "fXaxis_fNbins", 19))
        xmin = get(first_quartile_hist, :fXaxis_fXmin, get(first_quartile_hist, "fXaxis_fXmin", 0.0))
        xmax = get(first_quartile_hist, :fXaxis_fXmax, get(first_quartile_hist, "fXaxis_fXmax", 5.0))
        
        println("   Constructing from: nbins=$(nbins), xmin=$(xmin), xmax=$(xmax)")
        energy_edges = collect(range(xmin, xmax, length=nbins+1))
        println("   Constructed edges: ", energy_edges)
    end
    
    # Final fallback if we still don't have energy edges
    if energy_edges === nothing
        println("❌ Could not extract energy edges, using default")
        energy_edges = collect(range(0.0, 5.0, length=20))  # 19 bins based on exploration
    end
    
    # From exploration: we have 19 bins and data arrays with 21 elements
    # The data structure is likely: [underflow, bin1, bin2, ..., bin19, overflow]
    n_bins = length(energy_edges) - 1  # This should be 19
    data_length = 21  # Based on your debug output
    
    println("Energy edges: ", energy_edges)
    println("Number of bins: ", n_bins)
    println("Expected data array length: ", data_length)
    
    # Data extraction indices: skip underflow (index 1), take bins 2:20
    data_start_idx = 2
    data_end_idx = n_bins + 1  # This will be 20, so indices 2:20 = 19 elements
    
    println("Data extraction: indices [$(data_start_idx):$(data_end_idx)] = $(data_end_idx - data_start_idx + 1) elements")
    
    # Initialize total arrays with correct bin count
    for component in ["NoOscillations_Signal", "Oscillated_Signal", "NoOscillations_Total_beam_bkg", "Cosmic_bkg"]
        numu_total[component] = zeros(n_bins)
        numubar_total[component] = zeros(n_bins)
    end
    
    for q in 1:4
        # Load neutrino quartile
        quartile_data = Dict{String, Any}()
        
        # Load observed data for this quartile
        quartile_hist = data_file["neutrino_mode_numu_quartile$(q)"]
        
        # Extract observed data - skip underflow/overflow bins
        if haskey(quartile_hist, :fSumw2)
            raw_data = quartile_hist[:fSumw2]
            println("Quartile $(q) fSumw2 length: ", length(raw_data))
            
            if length(raw_data) == data_length
                # Extract the physics bins (skip underflow at index 1, skip overflow at index 21)
                observed_data = raw_data[data_start_idx:data_end_idx]
                println("   Extracted $(length(observed_data)) physics bins")
            else
                println("   Warning: Unexpected data length $(length(raw_data)), expected $(data_length)")
                # Try to extract what we can
                if length(raw_data) >= n_bins
                    observed_data = raw_data[1:n_bins]
                else
                    observed_data = vcat(raw_data, zeros(n_bins - length(raw_data)))
                end
            end
        elseif haskey(quartile_hist, "fSumw2")
            raw_data = quartile_hist["fSumw2"]
            println("Quartile $(q) fSumw2 (string) length: ", length(raw_data))
            
            if length(raw_data) == data_length
                observed_data = raw_data[data_start_idx:data_end_idx]
                println("   Extracted $(length(observed_data)) physics bins")
            else
                println("   Warning: Unexpected data length $(length(raw_data)), expected $(data_length)")
                if length(raw_data) >= n_bins
                    observed_data = raw_data[1:n_bins]
                else
                    observed_data = vcat(raw_data, zeros(n_bins - length(raw_data)))
                end
            end
        else
            println("Warning: Cannot find fSumw2 for neutrino quartile $(q)")
            observed_data = zeros(n_bins)
        end
        
        quartile_data["observed"] = observed_data
        quartile_data["energy_edges"] = energy_edges
        
        println("Quartile $(q) observed data length: ", length(observed_data))
        println("Observed data: ", observed_data)
        
        # Load MC components for this quartile
        quartile_mc_key = "prediction_components_numu_fhc_Quartile$(q-1)"
        
        if haskey(mc_file, quartile_mc_key)
            mc_dir = mc_file[quartile_mc_key]
            
            if isa(mc_dir, UnROOT.ROOTDirectory)
                # MC is a directory with components
                for component_name in keys(mc_dir)
                    component_hist = mc_dir[component_name]
                    
                    # Extract MC data using same logic as observed data
                    mc_data = nothing
                    if haskey(component_hist, :fSumw2)
                        raw_mc_data = component_hist[:fSumw2]
                        if length(raw_mc_data) == data_length
                            mc_data = raw_mc_data[data_start_idx:data_end_idx]
                        elseif length(raw_mc_data) >= n_bins
                            mc_data = raw_mc_data[1:n_bins]
                        else
                            mc_data = vcat(raw_mc_data, zeros(n_bins - length(raw_mc_data)))
                        end
                    elseif haskey(component_hist, "fSumw2")
                        raw_mc_data = component_hist["fSumw2"]
                        if length(raw_mc_data) == data_length
                            mc_data = raw_mc_data[data_start_idx:data_end_idx]
                        elseif length(raw_mc_data) >= n_bins
                            mc_data = raw_mc_data[1:n_bins]
                        else
                            mc_data = vcat(raw_mc_data, zeros(n_bins - length(raw_mc_data)))
                        end
                    else
                        println("Warning: No data found for MC component $(component_name) in quartile $(q)")
                        mc_data = zeros(n_bins)
                    end
                    
                    quartile_data[string(component_name)] = mc_data
                    println("Quartile $(q), MC component $(component_name): length=$(length(mc_data))")
                end
            else
                # Single histogram case
                if haskey(mc_dir, :fSumw2)
                    raw_mc_data = mc_dir[:fSumw2]
                    if length(raw_mc_data) == data_length
                        quartile_data[quartile_mc_key] = raw_mc_data[data_start_idx:data_end_idx]
                    else
                        quartile_data[quartile_mc_key] = raw_mc_data[1:min(n_bins, length(raw_mc_data))]
                    end
                elseif haskey(mc_dir, :fArray)
                    fN = get(mc_dir, :fN, length(mc_dir[:fArray]))
                    n_entries = isa(fN, Vector) ? fN[1] : fN
                    quartile_data[quartile_mc_key] = n_entries > 0 ? mc_dir[:fArray][1:min(n_bins, n_entries)] : zeros(n_bins)
                end
            end
        else
            println("Warning: MC key $(quartile_mc_key) not found")
        end
        
        push!(numu_quartiles, quartile_data)
        
        # Accumulate totals
        for component in ["NoOscillations_Signal", "Oscillated_Signal", "NoOscillations_Total_beam_bkg", "Cosmic_bkg"]
            if haskey(quartile_data, component)
                data_vec = quartile_data[component]
                if length(data_vec) == n_bins
                    numu_total[component] += data_vec
                    println("   Added $(component) to total: length=$(length(data_vec))")
                else
                    println("   Warning: $(component) has wrong length $(length(data_vec)), expected $(n_bins)")
                end
            end
        end
        
        # Load antineutrino quartile (same logic)
        quartile_data_bar = Dict{String, Any}()
        
        quartile_hist_bar = data_file["antineutrino_mode_numu_quartile$(q)"]
        
        # Extract observed antineutrino data
        if haskey(quartile_hist_bar, :fSumw2)
            raw_data = quartile_hist_bar[:fSumw2]
            if length(raw_data) == data_length
                observed_data_bar = raw_data[data_start_idx:data_end_idx]
            elseif length(raw_data) >= n_bins
                observed_data_bar = raw_data[1:n_bins]
            else
                observed_data_bar = vcat(raw_data, zeros(n_bins - length(raw_data)))
            end
        elseif haskey(quartile_hist_bar, "fSumw2")
            raw_data = quartile_hist_bar["fSumw2"]
            if length(raw_data) == data_length
                observed_data_bar = raw_data[data_start_idx:data_end_idx]
            elseif length(raw_data) >= n_bins
                observed_data_bar = raw_data[1:n_bins]
            else
                observed_data_bar = vcat(raw_data, zeros(n_bins - length(raw_data)))
            end
        else
            println("Warning: Cannot find data for antineutrino quartile $(q)")
            observed_data_bar = zeros(n_bins)
        end
        
        quartile_data_bar["observed"] = observed_data_bar
        quartile_data_bar["energy_edges"] = energy_edges
        
        # Load MC components for antineutrino quartile (same logic as neutrino)
        quartile_mc_key_bar = "prediction_components_numu_rhc_Quartile$(q-1)"
        
        if haskey(mc_file, quartile_mc_key_bar)
            mc_dir_bar = mc_file[quartile_mc_key_bar]
            
            if isa(mc_dir_bar, UnROOT.ROOTDirectory)
                for component_name in keys(mc_dir_bar)
                    component_hist = mc_dir_bar[component_name]
                    
                    if haskey(component_hist, :fSumw2)
                        raw_mc_data = component_hist[:fSumw2]
                        if length(raw_mc_data) == data_length
                            quartile_data_bar[string(component_name)] = raw_mc_data[data_start_idx:data_end_idx]
                        else
                            quartile_data_bar[string(component_name)] = raw_mc_data[1:min(n_bins, length(raw_mc_data))]
                        end
                    elseif haskey(component_hist, "fSumw2")
                        raw_mc_data = component_hist["fSumw2"]
                        if length(raw_mc_data) == data_length
                            quartile_data_bar[string(component_name)] = raw_mc_data[data_start_idx:data_end_idx]
                        else
                            quartile_data_bar[string(component_name)] = raw_mc_data[1:min(n_bins, length(raw_mc_data))]
                        end
                    else
                        println("Warning: No data found for antineutrino MC component $(component_name) in quartile $(q)")
                    end
                end
            else
                # Single histogram case
                if haskey(mc_dir_bar, :fSumw2)
                    raw_mc_data = mc_dir_bar[:fSumw2]
                    if length(raw_mc_data) == data_length
                        quartile_data_bar[quartile_mc_key_bar] = raw_mc_data[data_start_idx:data_end_idx]
                    else
                        quartile_data_bar[quartile_mc_key_bar] = raw_mc_data[1:min(n_bins, length(raw_mc_data))]
                    end
                elseif haskey(mc_dir_bar, :fArray)
                    fN = get(mc_dir_bar, :fN, length(mc_dir_bar[:fArray]))
                    n_entries = isa(fN, Vector) ? fN[1] : fN
                    quartile_data_bar[quartile_mc_key_bar] = n_entries > 0 ? mc_dir_bar[:fArray][1:min(n_bins, n_entries)] : zeros(n_bins)
                end
            end
        else
            println("Warning: Antineutrino MC key $(quartile_mc_key_bar) not found")
        end
        
        push!(numubar_quartiles, quartile_data_bar)
        
        # Accumulate antineutrino totals
        for component in ["NoOscillations_Signal", "Oscillated_Signal", "NoOscillations_Total_beam_bkg", "Cosmic_bkg"]
            if haskey(quartile_data_bar, component)
                data_vec = quartile_data_bar[component]
                if length(data_vec) == n_bins
                    numubar_total[component] += data_vec
                else
                    println("   Warning: antineutrino $(component) has wrong length $(length(data_vec)), expected $(n_bins)")
                end
            end
        end
    end
    
    println("\n✅ Data loading complete!")
    println("Energy edges: ", energy_edges)
    println("Number of bins: ", n_bins)
    println("Neutrino total components: ", keys(numu_total))
    println("Antineutrino total components: ", keys(numubar_total))
    
    numu_data = (
        quartiles = numu_quartiles,
        total = numu_total,
        energy_edges = energy_edges
    )
    
    numubar_data = (
        quartiles = numubar_quartiles,
        total = numubar_total,
        energy_edges = energy_edges
    )
    
    return numu_data, numubar_data
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

function fast_predictions_new(signal, backgrounds, norm_factor; condense_to_bin3=false)
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
        # Neutrino quartile
        p_smeared = smearnorm(energy_grid, p_nu_survival, assets.numu_smearing[i], 4, 
                            assets.numu_e_scale, assets.numu_e_bias)
        
        # Rebin to detector energy bins
        quartile_data = assets.numu_data.quartiles[i]
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
    
    for segment in 1:3
        # Neutrino segment
        backgrounds_nu = [
            assets.nue_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nue_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nue_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        condense_mode = (segment == 3)
        prediction_nu = fast_predictions_new(signal_nue, backgrounds_nu, params["nova_norm"],
                                           condense_to_bin3=condense_mode)
        
        predictions["nue"]["segment$(segment)"] = prediction_nu
        
        # Antineutrino segment
        backgrounds_nubar = [
            assets.nuebar_data.mc_components["Wrong_sign_bkg$(segment)"],
            assets.nuebar_data.mc_components["Beam_nue_bkg$(segment)"],
            assets.nuebar_data.mc_components["Cosmic_bkg$(segment)"]
        ]
        
        prediction_nubar = fast_predictions_new(signal_nuebar, backgrounds_nubar, params["nova_norm"],
                                              condense_to_bin3=condense_mode)
        
        predictions["nuebar"]["segment$(segment)"] = prediction_nubar
    end
    
    return predictions
end

function get_forward_model(physics, assets)

    function forward_model(params)
        exp_events_numu = make_numu_predictions(params, physics, assets)
        exp_events_nue = make_nue_predictions(params, physics, assets)
        
       
        return (
            numu = Poisson.(exp_events_numu),
            nue = Poisson.(exp_events_nue)
        )
    end
    return forward_model
end



function get_plot(physics, assets)
    """Create plotting function for NOvA data and predictions"""
    
    function plot(params)
        f = Figure(resolution=(1200, 800))
        
        # Get predictions
        numu_predictions = make_numu_predictions(params, physics, assets)
        nue_predictions = make_nue_predictions(params, physics, assets)
        
        # Plot muon neutrino disappearance by quartile
        for i in 1:4
            ax = Axis(f[1, i], title="νμ Quartile $(i)")
            
            observed = assets.numu_data.quartiles[i]["observed"]
            predicted = numu_predictions["numu"][i]
            energy_centers = (assets.numu_data.energy_edges[1:end-1] .+ 
                            assets.numu_data.energy_edges[2:end]) ./ 2
            
            scatter!(ax, energy_centers, observed, color=:black, label="Observed")
            lines!(ax, energy_centers, predicted, color=:red, label="Predicted")
            
            ax.xlabel = "Energy (GeV)"
            ax.ylabel = "Events"
            axislegend(ax)
        end
        
        # Plot electron neutrino appearance
        ax_nue = Axis(f[2, 1:2], title="νμ → νe Appearance")
        
        # Combine all segments
        all_observed_nue = vcat(assets.nue_data.observed.segment1,
                               assets.nue_data.observed.segment2,
                               assets.nue_data.observed.segment3)
        all_predicted_nue = vcat(nue_predictions["nue"]["segment1"],
                                nue_predictions["nue"]["segment2"],
                                nue_predictions["nue"]["segment3"])
        
        energy_bins = 1:length(all_observed_nue)
        scatter!(ax_nue, energy_bins, all_observed_nue, color=:black, label="Observed")
        lines!(ax_nue, energy_bins, all_predicted_nue, color=:blue, label="Predicted")
        
        ax_nue.xlabel = "Bin Number"
        ax_nue.ylabel = "Events"
        axislegend(ax_nue)
        
        # Plot antineutrino appearance
        ax_nuebar = Axis(f[2, 3:4], title="ν̄μ → ν̄e Appearance")
        
        all_observed_nuebar = vcat(assets.nuebar_data.observed.segment1,
                                  assets.nuebar_data.observed.segment2,
                                  assets.nuebar_data.observed.segment3)
        all_predicted_nuebar = vcat(nue_predictions["nuebar"]["segment1"],
                                   nue_predictions["nuebar"]["segment2"],
                                   nue_predictions["nuebar"]["segment3"])
        
        scatter!(ax_nuebar, energy_bins, all_observed_nuebar, color=:black, label="Observed")
        lines!(ax_nuebar, energy_bins, all_predicted_nuebar, color=:green, label="Predicted")
        
        ax_nuebar.xlabel = "Bin Number"
        ax_nuebar.ylabel = "Events"
        axislegend(ax_nuebar)
        
        return f
    end
    
    return plot
end

end  # module Nova

