using Pkg
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using Plots
using FileIO
using BAT, IntervalSets
using DelimitedFiles
using DensityInterface
using ValueShapes
using HDF5
using Neurthino
using Base.Threads
using QuadGK
using Tables
using TypedTables
using Statistics
using Distributions


include("neurthino.jl")

function heaviside(t)
    0.5 * (sign(t) + 1)
 end
 

prior = NamedTupleDist(
    theta12 = 0.63,
    theta13 = 0.152,
    theta23 = 0.84,
    Deltasq_m21 = 7.74e-5,
    Deltasq_m31 = 2.53e-3,
    delta_CP = 0.82 * pi,
    m0 = Uniform(0.0001, 1.99),
    H = 1,
    ADD_radius =Uniform(0.1, 10),
    ca_1 = Uniform(0.0001, 40),#2.23,
    ca_2 = Uniform(0.0001, 40),#0.63,
    ca_3 = Uniform(0.0001, 40)#-0.019
)

Q= 18572

signalvecLED= Vector{Float64}(undef,0)
signalvecSM= Vector{Float64}(undef,0)
signalvecDD= Vector{Float64}(undef,0)
ratio= Vector{Float64}(undef,0)
Qvec= Vector{Float64}(undef,0)


function integrandSM(K, pr)
    masses = get_abs_masses(pr)
    massesvec = Vector{Float64}(undef,0)
    v = Vector{Float64}(undef,0)
    thetafunc = Vector{Float64}(undef,0)
    SM= get_SM(pr)
    for i in 1:length(masses)
        m = (masses[i])^2
        push!(massesvec, m)
     end


    for i in 1:length(SM[1][1,:])
        a = (abs2(SM[1][1,i]))
        push!(v, a)
     end
     for i in 1:length(masses)
        b = heaviside(Q-K- masses[i])
        push!(thetafunc, b)
    end
    kurie = 0
    for i in 1:length(SM[1][1,:])
      
        kurie += v[i]*sqrt(complex((Q-K)^2 -massesvec[i]))*thetafunc[i]
     

    end

    return  sqrt(K)*(Q-K)*kurie
end

function integrandSM_neu(K, pr)
    prior3 = deepcopy(pr)
    radius = pr["ADD_radius"]
    prior3["m0"] = contur(radius)
    SM_neu = get_SM(prior3)
    masses = get_abs_masses(prior3)
    massesvec = Vector{Float64}(undef,0)
    v = Vector{Float64}(undef,0)
    thetafunc = Vector{Float64}(undef,0)
    
    for i in 1:length(masses)
        m = (masses[i])^2
        push!(massesvec, m)
     end


    for i in 1:length(SM_neu[1][1,:])
        a = (abs(SM_neu[1][1,i]))^2
        push!(v, a)
     end
     for i in 1:length(masses)
        b = heaviside(Q-K- masses[i])
        push!(thetafunc, b)
    end
    kurie = 0
    for i in 1:length(SM_neu[1][1,:])
      
        kurie += v[i]*sqrt(complex((Q-K)^2 -massesvec[i]))*thetafunc[i]
     

    end

    return  sqrt(K)*(Q-K)*kurie
end




function integrandLED(K, pr) 
    M = get_ADD(pr)
    v = Vector{Float64}(undef,0)
    thetafunc = Vector{Float64}(undef,0)
    for i in 1:length(M[1][1,:])
    a = (abs(M[1][1,i]))^2
    push!(v, a)
    end

    for i in 1:length(M[2])
        b = heaviside(Q-K- sqrt(complex(M[2][i])))
        push!(thetafunc, b)
    end
  
   
    kurie = 0
    for i in 1:length(M[1][1,:])
      
        kurie += v[i]*sqrt(complex((Q-K)^2 -M[2][i]))*thetafunc[i]
     

    end

    return  sqrt(K)*(Q-K)*kurie
end

function integrandLEDbulkmass(K, pr) 
    Mwithbulk = get_ADD_bulkmasses(pr)
    v = Vector{Float64}(undef,0)
    thetafunc = Vector{Float64}(undef,0)
    for i in 1:length(Mwithbulk[1][1,:])
    a = (abs(Mwithbulk[1][1,i]))^2
    push!(v, a)
    end

    for i in 1:length(Mwithbulk[2])
        b = heaviside(Q-K- abs(sqrt(complex(Mwithbulk[2][i]))))
        push!(thetafunc, b)
    end

    kurie = 0
    for i in 1:length(Mwithbulk[1][1,:])
      
        kurie += v[i]*sqrt(complex((Q-K)^2 -abs(Mwithbulk[2][i])))*thetafunc[i]
     

    end

    return  sqrt(K)*(Q-K)*kurie
end


function contur(x)

    return (0.4* heaviside(-x+0.010956878413974792) + heaviside(x-0.010956878413974792) * 0.042* 1*x^(-0.50) ) #10^(0)  + (heaviside(x-0.022956878413974792) *  
end


function integrandLED_finding(K, pr) 
    prior2 = deepcopy(pr)
    radius = pr["ADD_radius"]
    prior2["m0"] = contur(radius)

    v = Vector{Float64}(undef,0)
    thetafunc = Vector{Float64}(undef,0)
    M2 = get_ADD(prior2)
    for i in 1:length(M2[1][1,:])
    a = (abs(M2[1][1,i]))^2
    push!(v, a)
    end

    for i in 1:length(M2[2])
        b = heaviside(Q-K- sqrt(complex(M2[2][i])))
        push!(thetafunc, b)
    end
  
   
    
  return  sqrt(K)*(Q-K)*dot(v,((sqrt.(complex((Q-K)^2 .- M2[2])))).* thetafunc)
end






function testfunction(par)

    Signal_LED_unnormalized = quadgk(x ->integrandLED_finding(x, par), Q-40, Q,  rtol=1e-1)

    Signal_DarkDim =  quadgk(x ->integrandLEDbulkmass(x, par), Q-40, Q, rtol=1e-1)

    Signal_SM_unnormalized_neu = quadgk(x ->integrandSM_neu(x, par), Q-40, Q, rtol=1e-1)

    A = abs(1-(Signal_LED_unnormalized[1]/Signal_SM_unnormalized_neu[1]))
    B = abs(1-(Signal_DarkDim[1]/Signal_SM_unnormalized_neu[1]))

      if B > A

          return logpdf(Normal(100),0.0001)
      else return logpdf(Normal(100),100)
      end
  end 



likelihood = let Test = testfunction
  

    logfuncdensity(function (params)

   
    par = Dict(string(k) => v for (k,v) in pairs(params))

    ll_value = Test(par) 

        return ll_value
    end)
end 


posterior = PosteriorDensity(likelihood, prior)
sampling_algorithm = MCMCSampling(mcalg = MetropolisHastings(), nchains = 4, nsteps = 10000)
sampling_output  = bat_sample(posterior, sampling_algorithm)
samples  = sampling_output.result

plot(
    samples,
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50
)