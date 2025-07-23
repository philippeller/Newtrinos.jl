
module katrin

import ..Newtrinos

using LinearAlgebra, Statistics, Distributions, StatsBase
using FileIO
using Base.Threads
using CSV
using DataFrames
using Interpolations



@kwdef struct Katrin <: Newtrinos.Experiment
      physics::NamedTuple
      params::NamedTuple
      priors::NamedTuple
      #forward::Function
      # assets::NamedTuple
end

function configure(physics)
    physics = (;physics.osc)

    return Katrin(
        physics = physics,
        params = (;),
        priors = (;),
    )
end


# function to get m0 posterior from m_nu posterior in SM 

function get_posterior_SM(params)

        
    #load m_nu posterior data

    posterior_data_m_nu=CSV.read("/home/sofialon/Newtrinos.jl/src/experiments/katrin/posterior_m_nu.csv", DataFrame)

    #make the distribution continuous
    posterior_m_nu=interpolate((posterior_data_m_nu[!,1],), posterior_data_m_nu[!,2], Gridded(Linear()))
    posterior_m_nu = extrapolate(posterior_m_nu, 0.0)  # Extrapolate with 0.0 outside bounds


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
    posterior_m_nu=interpolate((posterior_data_m_nu[!,1],), posterior_data_m_nu[!,2], Gridded(Linear()))
    posterior_m_nu = extrapolate(posterior_m_nu, 0.0)  # Extrapolate with 0.0 outside bounds


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


end