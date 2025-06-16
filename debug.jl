
using UnROOT

# Debug why NUMU predictions are all zeros

function debug_numu_zero_predictions(params, physics, assets)
    println("🔍 DEBUGGING NUMU ZERO PREDICTIONS")
    println("="^50)
    
    # 1. Check if make_numu_predictions function exists and runs
    println("1. Testing make_numu_predictions function:")
    try
        result = make_numu_predictions(params, physics, assets)
        println("   ✅ Function runs successfully")
        println("   📊 Result type: $(typeof(result))")
        
        if isa(result, Dict)
            println("   📋 Result keys: $(keys(result))")
            
            # Check NUMU results
            if haskey(result, "numu")
                numu_data = result["numu"]
                println("   🔢 NUMU data type: $(typeof(numu_data))")
                
                if isa(numu_data, Vector)
                    println("   📏 NUMU quartiles: $(length(numu_data))")
                    for i in 1:min(4, length(numu_data))
                        if isa(numu_data[i], Vector)
                            println("     Quartile $i: length=$(length(numu_data[i])), sum=$(sum(numu_data[i]))")
                            if length(numu_data[i]) > 0
                                println("       First few values: $(numu_data[i][1:min(3, length(numu_data[i]))])")
                            end
                        else
                            println("     Quartile $i: $(numu_data[i])")
                        end
                    end
                end
            else
                println("   ❌ No 'numu' key found in results")
            end
            
            # Check NUMUBAR results  
            if haskey(result, "numubar")
                numubar_data = result["numubar"]
                println("   🔢 NUMUBAR data type: $(typeof(numubar_data))")
                
                if isa(numubar_data, Vector)
                    println("   📏 NUMUBAR quartiles: $(length(numubar_data))")
                    for i in 1:min(4, length(numubar_data))
                        if isa(numubar_data[i], Vector)
                            println("     Quartile $i: length=$(length(numubar_data[i])), sum=$(sum(numubar_data[i]))")
                        end
                    end
                end
            end
        else
            println("   ⚠️ Result is not a Dict: $result")
        end
        
    catch e
        println("   ❌ Function failed with error:")
        println("     $e")
        return
    end
    
    println("\n2. Checking MC histogram data:")
    # Check if MC histograms exist and have data
    if haskey(assets.numu_data.total, "Oscillated_Total_pred")
        mc_data = assets.numu_data.total["Oscillated_Total_pred"]
        println("   ✅ NUMU MC 'Oscillated_Total_pred' found")
        println("   📏 Length: $(length(mc_data))")
        println("   📊 Sum: $(sum(mc_data))")
        if length(mc_data) > 0
            println("   🔢 First few values: $(mc_data[1:min(5, length(mc_data))])")
            println("   📈 Min: $(minimum(mc_data)), Max: $(maximum(mc_data))")
        end
    else
        println("   ❌ NUMU MC 'Oscillated_Total_pred' not found")
        println("   📋 Available keys: $(keys(assets.numu_data.total))")
    end
    
    if haskey(assets.numubar_data.total, "Oscillated_Total_pred")
        mc_data = assets.numubar_data.total["Oscillated_Total_pred"]
        println("   ✅ NUMUBAR MC 'Oscillated_Total_pred' found")
        println("   📏 Length: $(length(mc_data))")
        println("   📊 Sum: $(sum(mc_data))")
    else
        println("   ❌ NUMUBAR MC 'Oscillated_Total_pred' not found")
        println("   📋 Available keys: $(keys(assets.numubar_data.total))")
    end
    
    println("\n3. Checking input parameters:")
    println("   📋 Parameter values:")
    for (key, val) in pairs(params)
        println("     $key: $val")
    end
    
    println("\n4. Checking observed data for comparison:")
    for i in 1:4
        if haskey(assets.numu_data.quartiles[i], "observed")
            observed = assets.numu_data.quartiles[i]["observed"]
            println("   📊 NUMU Quartile $i observed: sum=$(sum(observed))")
        end
    end
    
    println("\n5. Manual oscillation test:")
    # Test basic oscillation calculation
    try
        # Get some test parameters
        dm31 = params.Δm²₃₁
        theta23 = params.θ₂₃
        L = assets.L
        
        # Test energy (middle of typical neutrino range)
        test_E = 2.0  # GeV
        
        # Basic muon neutrino survival probability
        # P(νμ → νμ) ≈ 1 - sin²(2θ₂₃) * sin²(1.27 * Δm²₃₁ * L / E)
        arg = 1.27 * dm31 * L / test_E
        P_survival = 1 - sin(2*theta23)^2 * sin(arg)^2
        
        println("   🧮 Manual oscillation test:")
        println("     Δm²₃₁ = $dm31 eV²")
        println("     θ₂₃ = $theta23 rad")
        println("     L = $L km")
        println("     Test energy = $test_E GeV")
        println("     Oscillation argument = $arg")
        println("     Survival probability = $P_survival")
        
        if P_survival ≈ 1.0
            println("   ⚠️ Survival probability ≈ 1 → No oscillation effect!")
        elseif P_survival ≤ 0
            println("   ⚠️ Survival probability ≤ 0 → Unphysical!")
        else
            println("   ✅ Survival probability looks reasonable")
        end
        
    catch e
        println("   ❌ Manual oscillation test failed: $e")
    end
    
    println("\n" * "="^50)
    println("🎯 LIKELY CAUSES OF ZERO PREDICTIONS:")
    println("1. make_numu_predictions function returns zeros")
    println("2. MC histogram data is all zeros")
    println("3. Oscillation calculation has a bug")
    println("4. Parameter values are causing no oscillation")
    println("5. Wrong data structure being returned")
    println("="^50)
end

# Usage:
debug_numu_zero_predictions(params, physics, assets)
 