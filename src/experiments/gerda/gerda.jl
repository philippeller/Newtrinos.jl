
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

    assets = (

        observed = 1e28, #0.9*1e26, #gerda 
        # 1*1e28, #legend 1000
        
    )
    return assets

    
end


 
function get_neutrinomas_old(cfg=NNM)
    function NeutrinoMassNNM_old(params::NamedTuple)

        U= Newtrinos.osc.get_PMNS(params)

        N = round(Int,params[:N])

        func=  Newtrinos.osc.get_matrices(cfg)

        final, h, V = func(params)
        
        x_e = U[1,:]
        x_1 = V[1,:]

        masses_SM_sq = Newtrinos.osc.get_abs_masses(params).^2

        delta_masses_NN = h

        masses_NN_original = masses_SM_sq[1].+delta_masses_NN
        #masses_NN_original[1] = masses_SM_sq[1]
        #masses_NN_original[2] = masses_SM_sq[2]
        #masses_NN_original[3] = masses_SM_sq[3]

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



function get_neutrinomass(cfg=NNM)
    function NeutrinoMassNNM(params::NamedTuple)

        

        U= Newtrinos.osc.get_PMNS(params)

        N = round(Int,params[:N])

        func=  Newtrinos.osc.get_matrices(cfg)

        final, h, eigen, V_e, V_m, V_t = func(params)
        #final, h= func(params)
        masses_NN= eigen

        x_e = U[1,:]
        x_1_e = V_e[1,1: N]
        x_1_m = V_m[1,1: N]
        x_1_t = V_t[1,1: N]

        mass_e = eigen[1:3:end]      
        mass_m = eigen[2:3:end]   
        mass_t = eigen[3:3:end]      
        N_e = length(mass_e)
        N_m = length(mass_m)
        N_t = length(mass_t)

        if any(mass_e .> 1e12) 
            mass_e = mass_e[mass_e .<= 1e12]
            N_e = length(mass_e)
            x_1_e = V_e[1, 1:N_e]
        end

        if any(mass_m .> 1e12) 
            mass_m = mass_m[mass_m .<= 1e12]
            N_m = length(mass_m)
            x_1_m = V_m[1, 1:N_m]
        end

        if any(mass_t .> 1e12) 
            mass_t = mass_t[mass_t .<= 1e12]
            N_t = length(mass_t)
            x_1_t = V_t[1, 1:N_t]
        end

        N = [N_e, N_m, N_t]
        masses_NN = [mass_e, mass_m, mass_t]

        X=[x_1_e, x_1_m, x_1_t]
        sum=Float64(0.0)
        
        for i in 1:3
            
            
            for j in 1:N[i]

                mass = masses_NN[i][j]
                integrand= abs((X[i][j].*(x_e[i])))^2 * sqrt(mass)
                sum += integrand
            end

        end
   

        return sum
     
    end
    return NeutrinoMassNNM
end

function mixing_angles(params::NamedTuple,cfg=NNM)

    U = Newtrinos.osc.get_PMNS(params)
    N = round(Int, params[:N])

    func = Newtrinos.osc.get_matrices(cfg)
    #final, h, V, eigen= func(params)
    final, h, eigen, V_e, V_m, V_t = func(params)


    x_e = U[1, :]
    x_1_e = V_e[1, 1:N]
    x_1_m = V_m[1, 1:N]
    x_1_t = V_t[1, 1:N]

    angles_e=abs.(x_e[1]*x_1_e)
    angles_m=abs.(x_e[2]*x_1_m)
    angles_t=abs.(x_e[3]*x_1_t)


    mass_e = eigen[1:3:end]      
    mass_m = eigen[2:3:end]   
    mass_t = eigen[3:3:end]      

    return mass_e, mass_m, mass_t ,angles_e, angles_m, angles_t

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
     
     #println(Thalf)



     return Thalf

    end
    return halftime
end



function comparing_times(physics,experiments, params)


    cfg = Newtrinos.osc.NNM()
    predicted_value =get_halftime(cfg)(params)
    observed= experiments.gerda.assets.observed
    dist_observed= Normal(observed, 0.01*1e26)
    twosigma_level= quantile(dist_observed, 0.9772)

    return predicted_value, twosigma_level
end    




function get_forward_model_correct(physics, assets)
    function forward_model(params)
    
        cfg = Newtrinos.osc.NNM()
        observed = 1e28#legend 200
        #predicted_value =get_neutrinomass(cfg)(params) #get_neutrinomass_SM(cfg)(params) 
        fun=get_halftime()
        predicted_value_T=fun(params)

        if predicted_value_T >= observed 
           predicted_value_T=observed
        end   
        sigma= 0.01*1e28#0.1*1e28 #
        #println("Predicted m_nu: ", predicted_value_T)
       
        return Normal(predicted_value_T, sigma)
       

    end
    return forward_model
end



end