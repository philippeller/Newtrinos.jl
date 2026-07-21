
module katrin

import ..Newtrinos
using LinearAlgebra, Statistics
using Distributions, StatsBase
using FileIO
using Base.Threads
using CSV
using BAT
using DataFrames


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

       
    assets = (

       
        observed = -0.14,
       
    )
    return assets

    
end



function get_neutrinomass_all_modes(cfg=NND)
    function NeutrinoMassNN(params::NamedTuple)
        U = Newtrinos.osc.get_PMNS(params)
  
        N = round(Int, params[:N])

        func = Newtrinos.osc.get_matrices(cfg)
        final, h, eigen, V_e, V_m, V_t = func(params)

        x_e = U[1, :]
        x_1_e = V_e[1, 1:N]
        x_1_m = V_m[1, 1:N]
        x_1_t = V_t[1, 1:N]
     


        mass_e = eigen[1:3:end]      
        mass_m = eigen[2:3:end]   
        mass_t = eigen[3:3:end]      
        N_e = length(mass_e)
        N_m = length(mass_m)
        N_t = length(mass_t)
    
       cut=(18.6*1e3)^2
       if any(mass_e .> cut) 
                   
            mask = mass_e .<= cut

            mass_e = mass_e[mask]
            V_e    = V_e[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_e; dims=1))
            V_e .= V_e ./ reshape(norms, 1, :)


            N_e = length(mass_e)
            x_1_e = V_e[1, 1:N_e]

        end

        if any(mass_m .> cut) 
        
            mask = mass_m .<= cut

            mass_m = mass_m[mask]
            V_m    = V_m[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_m; dims=1))
            V_m .= V_m ./ reshape(norms, 1, :)


            N_m = length(mass_m)
            x_1_m = V_m[1, 1:N_m]

        end

        if any(mass_t .> cut) 
          
            mask = mass_t .<= cut

            mass_t = mass_t[mask]
            V_t    = V_t[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_t; dims=1))
            V_t .= V_t ./ reshape(norms, 1, :)

            N_t = length(mass_t)
            x_1_t = V_t[1, 1:N_t]

        end

     
        N = [N_e, N_m, N_t]
        masses_NN = [mass_e, mass_m, mass_t]

        X = [x_1_e, x_1_m, x_1_t]
        sum = Float64(0.0)

        for i in 1:3
            squared_x_e = abs(x_e[i])^2
        
            for j in 1:N[i]
                mass = masses_NN[i][j]
                integrand = squared_x_e * abs(X[i][j])^2 * mass
                sum += integrand
            end
        end
        
       
        return sum
            
    end
    return NeutrinoMassNN
end




function get_neutrinomass(cfg=NND)
    function NeutrinoMassNN(params::NamedTuple)
        U = Newtrinos.osc.get_PMNS(params)
  
        N = round(Int, params[:N])

        func = Newtrinos.osc.get_matrices(cfg)
        final, h, eigen, V_e, V_m, V_t = func(params)

        x_e = U[1, :]
        x_1_e = V_e[1, 1:N]
        x_1_m = V_m[1, 1:N]
        x_1_t = V_t[1, 1:N]
     
        mass_e = eigen[1:3:end]      
        mass_m = eigen[2:3:end]   
        mass_t = eigen[3:3:end]      
        N_e = length(mass_e)
        N_m = length(mass_m)
        N_t = length(mass_t)
    
       cut= (1)^2
       if any(mass_e .> cut) 
         
            mask = mass_e .<= cut

            mass_e = mass_e[mask]
            V_e    = V_e[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_e; dims=1))
            V_e .= V_e ./ reshape(norms, 1, :)


            N_e = length(mass_e)
            x_1_e = V_e[1, 1:N_e]

        end

        if any(mass_m .> cut) 
           
            
            mask = mass_m .<= cut

            mass_m = mass_m[mask]
            V_m    = V_m[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_m; dims=1))
            V_m .= V_m ./ reshape(norms, 1, :)


            N_m = length(mass_m)
            x_1_m = V_m[1, 1:N_m]

        end

        if any(mass_t .> cut) 
     
           
            mask = mass_t .<= cut

            mass_t = mass_t[mask]
            V_t    = V_t[mask, mask]
            norms = sqrt.(Base.sum(abs2, V_t; dims=1))
            V_t .= V_t ./ reshape(norms, 1, :)

            N_t = length(mass_t)
            x_1_t = V_t[1, 1:N_t]

        end

 
     
        N = [N_e, N_m, N_t]
        masses_NN = [mass_e, mass_m, mass_t]

        X = [x_1_e, x_1_m, x_1_t]
        sum = Float64(0.0)

        for i in 1:3
            squared_x_e = abs(x_e[i])^2
        
            for j in 1:N[i]
                mass = masses_NN[i][j]
                integrand = squared_x_e * abs(X[i][j])^2 *mass /(1-(1/N^2))
                sum += integrand
            end
        end
        
       
        return sum
            
    end
    return NeutrinoMassNN
end


function mixing_angles(params::NamedTuple,cfg=NND)

    U = Newtrinos.osc.get_PMNS(params)
    N = round(Int, params[:N])

    func = Newtrinos.osc.get_matrices(cfg)
 
    final, h, eigen, V_e, V_m, V_t = func(params)


    x_e = U[1, :]
    x_1_e = V_e[1, 1:N]
    x_1_m = V_m[1, 1:N]
    x_1_t = V_t[1, 1:N]

    angles_e=abs.(x_e[1])*abs.(x_1_e)
    angles_m=abs.(x_e[2])*abs.(x_1_m)
    angles_t=abs.(x_e[3])*abs.(x_1_t)


    mass_e = eigen[1:3:end]      
    mass_m = eigen[2:3:end]   
    mass_t = eigen[3:3:end]      

    return mass_e, mass_m, mass_t ,angles_e, angles_m, angles_t

end    


function get_neutrinomass_SM(cfg=ThreeFlavour())
    function NeutrinoMass_SM(params::NamedTuple)

        U=  Newtrinos.osc.get_PMNS(params)
        
        x_e = U[1,:]

        # Add new parameter
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
        cfg = physics.osc.cfg.flavour
        predicted_value = get_neutrinomass(cfg)(params) 
       
        return Normal(predicted_value, 0.13) 
    end
    return forward_model
end


end