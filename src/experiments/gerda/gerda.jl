
module gerda

import ..Newtrinos

using LinearAlgebra, Statistics
using Distributions, StatsBase
using FileIO
using Base.Threads
using CSV
using DataFrames
using Interpolations



@kwdef struct Gerda <: Newtrinos.Experiment
      physics::NamedTuple
      params::NamedTuple
      priors::NamedTuple
      assets::NamedTuple
      forward_model::Function 
end

function configure(physics)
    physics = (;physics.osc)
    assets = get_assets(physics)

    return Gerda(
        physics = physics,
        params = (;),
        priors = (;),
        assets = assets,
        forward_model = get_forward_model_correct(physics, assets)
    )
end




function get_assets(physics; datadir = @__DIR__)
    @info "Loading Gerda data"

    # MAYBE LOADING THE POSTERIOR OF T1/2 (?)

    assets = (

        observed = 0.9*1e26, #0.9* 1e26  ,
       
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

        
        delta_masses_NN_original = h

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
        
        #eliminate masses that exceed the threshold
        delta_masses_NN= delta_masses_NN_original
         
        if any(delta_masses_NN_original .> 1e6)   #exclude the masses that exceed the treshold
            # Find all indices where masses exceed threshold
            indices_above_threshold = findall(delta_masses_NN_original .> 1e6)
            #println("Indices of masses exceeding threshold: ", indices_above_threshold)
            
            #println("Delta masses exceed threshold: $cancelled > 1e6")
           

            delta_masses_NN = delta_masses_NN_original[delta_masses_NN_original .<= 1e6] #keep only the ones inside the threshold
            N=round(Int,length(delta_masses_NN)/3) #reduce the N value accordingly

            #unitarity of the new matrix

            A_square = V[1:N, 1:N]
            x_1=x_1[1:N]
            # Make it unitary  (write normalization term)
            #Q, R = qr(A_square)
            U, S, V = svd(A_square)
            U_clean = U*V'
            V_unitary = U_clean

            # Verify
            @assert isapprox(V_unitary' * V_unitary, I)
            xcol=V_unitary[:,1]
            x_1=V_unitary[1,:]
            sum_norm = Base.sum(abs.(x_1).^2)
            sum_norm_col=Base.sum(abs.(xcol).^2)
            @assert isapprox(sum_norm, sum_norm_col)

        end

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




function get_neutrinomass(cfg=NNM)
    function NeutrinoMassNNM(params::NamedTuple)

        U= Newtrinos.osc.get_PMNS(params)

        N = round(Int,params[:N])

        func=  Newtrinos.osc.get_matrices(cfg)

        final, h, V = func(params)
        
        x_e = U[1,:]
        x_1 = V[1,:]

        masses_SM_sq = Newtrinos.osc.get_abs_masses(params).^2

        delta_masses_NN = h

        masses_NN_original = masses_SM_sq[1].+delta_masses_NN
        masses_NN_original[1] = masses_SM_sq[1]
        masses_NN_original[2] = masses_SM_sq[2]
        masses_NN_original[3] = masses_SM_sq[3]

        masses_NN = masses_NN_original
        
        #=
        if any(masses_NN_original .> 1e6)   #exclude the masses that exceed the treshold
            # Find all indices where masses exceed threshold
            indices_above_threshold = findall(masses_NN_original .> 1e6)
            #println("Indices of masses exceeding threshold: ", indices_above_threshold)
            
            #println("Delta masses exceed threshold: $cancelled > 1e6")
           

            masses_NN = masses_NN_original[masses_NN_original .<= 1e6] #keep only the ones inside the threshold
            N=round(Int,length(masses_NN)/3) #reduce the N value accordingly

            #unitarity of the new matrix

            A_square = V[1:N, 1:N]
            x_1=x_1[1:N]
            # Make it unitary  (write normalization term)
            #Q, R = qr(A_square)
            U, S, V = svd(A_square)
            U_clean = U*V'
            V_unitary = U_clean

            # Verify
            @assert isapprox(V_unitary' * V_unitary, I)
            xcol=V_unitary[:,1]
            x_1=V_unitary[1,:]
            sum_norm = Base.sum(abs.(x_1).^2)
            sum_norm_col=Base.sum(abs.(xcol).^2)
            @assert isapprox(sum_norm, sum_norm_col)

        end=#

        # Calculate the neutrino mass sum for the SM only
        sum = abs((x_e[1]*x_1[1])^2* sqrt(masses_SM_sq[1]))+
              abs((x_e[2]*x_1[1])^2* sqrt(masses_SM_sq[2]))+
              abs((x_e[3]*x_1[1])^2* sqrt(masses_SM_sq[3]))
       

        # Calculate the neutrino mass sum for the other sectors      
        for i in 1:3
            

            x_idx = 4 # Start at 4 for x_1
            delta_idx = 3+i # Start delta_masses_NN

            for j in 1:(N-3)

                mass = sqrt(masses_NN[delta_idx])
                integrand= abs((x_e[i]*x_1[x_idx])^2 * mass)
                sum += integrand

                x_idx += 1      # Increment by 1 for x_1
                delta_idx += 3  # Increment by 3 for delta_masses_NN (since you had 3*j)
             
            end

        end
   

        return sum
     
    end
    return NeutrinoMassNNM
end



function get_neutrinomass_SM(cfg=ThreeFlavour())
    function NeutrinoMass_SM(params::NamedTuple)

        U=  Newtrinos.osc.get_PMNS(params)
        
        x_e = U[1,:]

        # Add new parameter
        new_params = merge(params, (m₀ = 0.1,))
        masses_SM_sq =  Newtrinos.osc.get_abs_masses(new_params)

        m_nu_sq = 0.0

        for i in 1:3
            squared_x_e = abs(x_e[i]^2*masses_SM_sq[i])

            m_nu_sq += squared_x_e

        end

     return m_nu_sq

    end
    return NeutrinoMass_SM
end



function get_halftime(cfg= Newtrinos.osc.NNM())
    function halftime(params::NamedTuple)

     
     mass=get_neutrinomass(cfg)(params)
     
     Gg=3.37*(1e-15) #2.363*( 1e-15) #yr^-1
     g_a=1.27#1.25
     M_sq=(5.551)^2
     m_e=0.511*(1e6)

    


     T_inv=(Gg*((g_a)^4)*M_sq*(mass)^2)/(m_e)^2
     Thalf=1/T_inv
     
     println(Thalf)



     return Thalf

    end
    return halftime
end







function get_forward_model_correct(physics, assets)
    function forward_model(params)
    
        cfg = Newtrinos.osc.NNM()
        #predicted_value =get_neutrinomass(cfg)(params) #get_neutrinomass_SM(cfg)(params) 
        fun=get_halftime()
        predicted_value_T=fun(params)
        sigma=1.1 * 1e26 #1.1* 1e26 
        lower_bound=predicted_value_T-sigma
        upper_bound=1e46 #predicted_value_T+3*sigma
        #println("Predicted m_nu: ", predicted_value_T)
       
        return Uniform(lower_bound, upper_bound)#Normal(predicted_value_T, sigma)
       

    end
    return forward_model
end



end