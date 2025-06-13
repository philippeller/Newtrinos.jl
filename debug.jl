
using UnROOT




# Comprehensive energy edge extraction explorer
function explore_energy_edges(data_file)
    println("=== COMPREHENSIVE ROOT ENERGY EDGE EXPLORATION ===\n")
    
    # 1. Check top-level keys for anything edge-related
    println("1. TOP-LEVEL KEYS CONTAINING 'edge', 'bin', 'energy':")
    for key in keys(data_file)
        key_str = string(key)
        if occursin(r"edge|bin|energy|axis"i, key_str)
            println("   Found: ", key_str)
            try
                obj = data_file[key]
                println("     Type: ", typeof(obj))
                if isa(obj, Vector) && length(obj) > 0
                    println("     Length: ", length(obj))
                    println("     Sample: ", obj[1:min(5, length(obj))])
                end
            catch e
                println("     Error accessing: ", e)
            end
        end
    end
    
    # 2. Examine a specific histogram in detail
    println("\n2. DETAILED HISTOGRAM AXIS ANALYSIS:")
    hist_key = nothing
    for key in keys(data_file)
        if occursin("neutrino_mode_numu_quartile1", string(key))
            hist_key = key
            break
        end
    end
    
    if hist_key !== nothing
        hist = data_file[hist_key]
        println("Analyzing histogram: ", hist_key)
        
        # Look for X-axis related fields
        x_axis_keys = filter(k -> occursin(r"fXaxis|xaxis"i, string(k)), keys(hist))
        println("\nX-axis related keys:")
        for key in x_axis_keys
            try
                value = hist[key]
                println("   ", key, " = ", value, " (type: ", typeof(value), ")")
            catch e
                println("   ", key, " = ERROR: ", e)
            end
        end
        
        # Specifically look for bin edges
        println("\nBin edge candidates:")
        for key_name in ["fXaxis_fXbins", :fXaxis_fXbins, "fXaxis_fBins", :fXaxis_fBins]
            if haskey(hist, key_name)
                try
                    bins = hist[key_name]
                    println("   ", key_name, ": length=", length(bins), " values=", bins)
                catch e
                    println("   ", key_name, ": ERROR: ", e)
                end
            end
        end
        
        # Check if we can construct from min/max/nbins
        println("\nAxis construction parameters:")
        try
            nbins = get(hist, :fXaxis_fNbins, get(hist, "fXaxis_fNbins", "NOT_FOUND"))
            xmin = get(hist, :fXaxis_fXmin, get(hist, "fXaxis_fXmin", "NOT_FOUND"))
            xmax = get(hist, :fXaxis_fXmax, get(hist, "fXaxis_fXmax", "NOT_FOUND"))
            println("   fXaxis_fNbins: ", nbins)
            println("   fXaxis_fXmin: ", xmin)
            println("   fXaxis_fXmax: ", xmax)
            
            if all(x -> x != "NOT_FOUND", [nbins, xmin, xmax])
                edges = range(xmin, xmax, length=nbins+1)
                println("   Constructed edges: ", collect(edges))
                return collect(edges)
            end
        catch e
            println("   Error in construction: ", e)
        end
    end
    
    # 3. Look for edge information in other histogram types
    println("\n3. CHECKING OTHER HISTOGRAM TYPES:")
    for key in keys(data_file)
        key_str = string(key)
        if occursin("histogram", key_str) || occursin("axis", key_str) || occursin("edge", key_str)
            println("Checking: ", key_str)
            try
                obj = data_file[key]
                if haskey(obj, :fXaxis_fXbins) || haskey(obj, "fXaxis_fXbins")
                    bins = get(obj, :fXaxis_fXbins, get(obj, "fXaxis_fXbins", []))
                    if length(bins) > 1
                        println("   Found edges in ", key_str, ": ", bins)
                        return bins
                    end
                end
            catch e
                println("   Error: ", e)
            end
        end
    end
    
    # 4. Check subdirectories for edge information
    println("\n4. CHECKING SUBDIRECTORIES:")
    for key in keys(data_file)
        try
            obj = data_file[key]
            if isa(obj, UnROOT.ROOTDirectory)
                println("Directory: ", key)
                for subkey in keys(obj)
                    subkey_str = string(subkey)
                    if occursin(r"edge|bin|energy|axis"i, subkey_str)
                        println("   ", subkey_str, " -> ", typeof(obj[subkey]))
                        try
                            subobj = obj[subkey]
                            if isa(subobj, Vector) && length(subobj) > 1
                                println("     Values: ", subobj)
                                return subobj
                            end
                        catch e
                            println("     Error: ", e)
                        end
                    end
                end
            end
        catch e
            continue
        end
    end
    
    println("\n❌ Could not find energy edges in ROOT file")
    return nothing
end

# Run the exploration
println("Exploring your ROOT file for energy edges...")

data_file = ROOTFile("/home/sofialon/Newtrinos.jl/src/experiments/nova/NOvA_2020_data_histograms.root") # Replace with your actual ROOT file path
found_edges = explore_energy_edges(data_file)