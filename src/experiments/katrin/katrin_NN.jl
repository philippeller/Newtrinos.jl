
module katrin

import ..Newtrinos

using LinearAlgebra, Statistics
using Distributions, StatsBase
using FileIO
using Base.Threads
using CSV
using DataFrames
using Interpolations



@kwdef struct Katrin <: Newtrinos.Experiment
      physics::NamedTuple
      params::NamedTuple
      priors::NamedTuple
      assets::NamedTuple
      forward_model::Function 
end

function configure(physics)
    physics = (;physics.osc)
    assets = get_assets(physics)

    return Katrin(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model_correct(physics, assets)
    )
end




function get_assets(physics; datadir = @__DIR__)
    @info "Loading Katrin data"

    # Read the CSV data
    data_df = CSV.read("/home/sofialon/Newtrinos.jl/src/experiments/katrin/posterior_m_nu.csv", DataFrame)
    
    # Create interpolation function
    #posterior_m_nu = interpolate((data_df[!, 1],), data_df[!, 2], Gridded(Linear()))
    #posterior_interp = extrapolate(posterior_m_nu, 0.0)  # Interpolation function
    
    # Use original DataFrame for observed data
    observed = (
        mass_values = data_df[:, 1],    # x-axis values from DataFrame
        counts = data_df[:, 2],         # y-axis values from DataFrame
    )
    
    posterior_sample = observed
    posterior_mean = mean(posterior_sample)
    assets = (

        posterior_sample = observed,
        mass_values = observed.mass_values,
        observed = 0.26,
       
    )
    return assets

    
end



# function to get m0 posterior from m_nu posterior in SM 

function get_posterior_SM(params)

        
    #load m_nu posterior data

    posterior_data_m_nu=CSV.read("/home/sofialon/Newtrinos.jl/src/experiments/katrin/posterior_m_nu.csv", DataFrame)

    #make the distribution continuous
    #posterior_m_nu=interpolate((posterior_data_m_nu[!,1],), posterior_data_m_nu[!,2], Gridded(Linear()))
    #posterior_m_nu = extrapolate(posterior_m_nu, 0.0)  # Extrapolate with 0.0 outside bounds


    m0_posterior = zeros(size(posterior_data_m_nu, 1), 2)

    for i in 1:size(posterior_data_m_nu, 1)
        m_nu_squared = posterior_data_m_nu[!, 1][i] 

        p = params
        U = Newtrinos.osc.get_PMNS(p)

        sumU = 0.0
        for j in 1:3 
            sumU += abs(U[1, j])^2
        end

        term1 = abs(U[1, 2])^2 * (p[:Δm²₂₁])
        term2 = abs(U[1, 3])^2 * (p[:Δm²₃₁])  
        m0_squared = (m_nu_squared - term1 - term2) / sumU

        m0_posterior[i, 1] = m0_squared
        jacobian = (sumU * sqrt(m0_squared)) / sqrt(m_nu_squared)

        m0_posterior[i, 2] =  posterior_data_m_nu[!, 2][i]  * jacobian

    end
    return m0_posterior

end    

# function to get m0 posterior from m_nu posterior in NND-NNM


function get_posterior_NN(params, cfg)

        
    #load m_nu posterior data

    posterior_data_m_nu=CSV.read("/home/sofialon/Newtrinos.jl/src/experiments/katrin/posterior_m_nu.csv", DataFrame)

    #make the distribution continuous
    #posterior_m_nu=interpolate((posterior_data_m_nu[!,1],), posterior_data_m_nu[!,2], Gridded(Linear()))
    #posterior_m_nu = extrapolate(posterior_m_nu, 0.0)  # Extrapolate with 0.0 outside bounds


    m0_posterior = zeros(size(posterior_data_m_nu, 1), 2)

    for k in 1:size(posterior_data_m_nu, 1)

        m_nu_squared = posterior_data_m_nu[!,1][k] 

        p= params
        N = round(Int,params[:N])

        U= Newtrinos.osc.get_PMNS(p)

        func= Newtrinos.osc.get_matrices(cfg)
        final, h, V = func(params)

        x_e = U[1,:]
        x_1 = V[1,:]

        
        delta_masses_NN = h

        delta_m_nu_sq = 0.0
        sumU = 0.0
        sumV= 0.0

        for i in 1:3
            sumU += abs(U[1,i])^2
        end

        for j in 1:N
            sumV += abs(V[1,j])^2
        end

        sum=params[:Δm²₃₁]*abs(x_e[3])^2*abs(x_1[3])^2 + params[:Δm²₂₁]*abs(x_e[2])^2*abs(x_1[2])^2

        for i in 1:3
            squared_x_e = abs(x_e[i])^2

            x_idx = 4 # Start at 4 for x_1
            delta_idx = 3+i # Start delta_masses_NN
            sum_int = 0.0
            for j in 1:(N-3)

            delta_mass = delta_masses_NN[delta_idx]
            integrand= squared_x_e * abs(x_1[x_idx])^2 * delta_mass
            sum_int += integrand

            x_idx += 1      # Increment by 1 for x_1
            delta_idx += 3  # Increment by 3 for delta_masses_NN (since you had 3*j)
            end

            delta_m_nu_sq += sum_int

        end

        m0_squared= (m_nu_squared-delta_m_nu_sq-sum) / (sumU*sumV)

        
       jacobian = (sumU * sumV * sqrt(abs(m0_squared))) / sqrt(m_nu_squared)

        if m0_squared < 0
            m0_squared = 0.0
        end

        m0_posterior[k, 1] = m0_squared
        m0_posterior[k, 2] = posterior_data_m_nu[!, 2][k] * jacobian


    end
    return m0_posterior

end    




function get_neutrinomass(cfg=NND)
    function NeutrinoMassNND(params::NamedTuple)

        U= Newtrinos.osc.get_PMNS(params)

        N = round(Int,params[:N])

        func=  Newtrinos.osc.get_matrices(cfg)

        final, h, V = func(params)
        
        x_e = U[1,:]
        x_1 = V[1,:]

        masses_SM_sq = Newtrinos.osc.get_abs_masses(params).^2

        delta_masses_NN = h
        if any(delta_masses_NN .> 1000)
            cancelled=delta_masses_NN[delta_masses_NN .> 1000]
            println("Delta masses exceed threshold: $cancelled > 1000")
            delta_masses_NN[delta_masses_NN .> 1000] .= 0
        end


        m_nu_sq = 0.0 #PROBLEMATIC SUM!
        sum = masses_SM_sq[1]*(abs(x_e[1])^2*abs(x_1[1])^2 +params[:Δm²₃₁]*abs(x_e[3])^2*abs(x_1[3])^2 + params[:Δm²₂₁]*abs(x_e[2])^2*abs(x_1[2])^2)
        
        for i in 1:3
            squared_x_e = abs(x_e[i])^2

            x_idx = 4 # Start at 4 for x_1
            delta_idx = 3+i # Start delta_masses_NN

            for j in 1:(N-3)

            mass = masses_SM_sq[1]+delta_masses_NN[delta_idx]
            integrand= squared_x_e * abs(x_1[x_idx])^2 * mass
            sum += integrand

            # Check if sum exceeds threshold after each addition
            if sum > 1000
                error("Neutrino mass calculation exceeded threshold: sum = $sum > 1000 " *
                        "at iteration i=$i, j=$j (N=$N)")
            end


            x_idx += 1      # Increment by 1 for x_1
            delta_idx += 3  # Increment by 3 for delta_masses_NN (since you had 3*j)
            end

        end
    # println("at parameters: N=$(params[:N]), m₀=$(params[:m₀]), r=$(params[:r])")   
    # println(sum)
     return sum

    end
    return NeutrinoMassNND
end



function get_neutrinomass_SM(cfg=ThreeFlavour())
    function NeutrinoMass_SM(params::NamedTuple)

        U=  Newtrinos.osc.get_PMNS(params)
        
        x_e = U[1,:]

        # Add new parameters
        new_params = merge(params, (m₀ = 0.1,))
        masses_SM_sq =  Newtrinos.osc.get_abs_masses(new_params).^2

        m_nu_sq = 0.0

        for i in 1:3
            squared_x_e = abs(x_e[i])^2*masses_SM_sq[i]

            m_nu_sq += squared_x_e

        end

     return m_nu_sq

    end
    return NeutrinoMass_SM
end






function get_forward_model_correct(physics, assets)
    function forward_model(params)
    
        cfg = Newtrinos.osc.NND()
        predicted_value =get_neutrinomass(cfg)(params) #get_neutrinomass_SM(cfg)(params) 
        #println("Predicted m_nu: ", predicted_value)
        return Normal(predicted_value, 0.3)
       

    end
    return forward_model
end



function create_katrin_likelihood_posteriors(experiments,params)
    katrin_exp = experiments.katrin
    posterior_sample = katrin_exp.assets.observed
    posterior_mean = mean(posterior_sample)
    posterior_std = std(posterior_sample)
    katrin_posterior = Normal(posterior_mean, posterior_std)
    
    predicted = get_posterior_SM(params)
    predicted = predicted[:,2]  # Extract the second column (counts)
    predicted_mean = mean(predicted)
    predicted_std = std(predicted)
    #predicted = Normal(predicted_mean, predicted_std)
    predicted_safe = max.(predicted, 1e-10)  # Avoid log(0) issues
    posterior_sample_safe = max.(posterior_sample, 1e-10)  # Avoid log(0) issues

    likelihood=-2*sum(log.(predicted_safe).- log.(posterior_sample_safe))

    return likelihood
end


end