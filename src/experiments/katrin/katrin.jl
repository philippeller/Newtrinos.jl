#using Random

module katrin

import ..Newtrinos

using LinearAlgebra, Statistics, Distributions, StatsBase
#using Plots
#using BAT, IntervalSets
#using DelimitedFiles
#using QuadGK
#using DensityInterface
#using ValueShapes
using HDF5
using FileIO
using Base.Threads



@kwdef struct Katrin <: Newtrinos.Experiment
      physics::NamedTuple
      params::NamedTuple
      priors::NamedTuple
     # assets::NamedTuple
end

function configure(physics)
    physics = (;physics.osc)

    return Katrin(
        physics = physics,
        params = (;),
        priors = (;)
    )
end

#=function NeutrinoMass(p::NamedTuple{(:sinsqrt2theta13, :Delta_m_31, :Species, :massfactor, :m_e, :theta23)})::Float64

  th13 = 0.5* asin(sqrt(p.sinsqrt2theta13))

  N = p.Species
  

  U1 = [1 0 0; 0 cos(p.theta23) sin(p.theta23); 0 -sin(p.theta23) cos(p.theta23)]
  U2 = [cos(th13) 0 sin(th13) * exp(-1im * deltacp); 0 1 0; -sin(th13) * exp(1im * deltacp) 0 cos(th13)]
  U3 = [cos(theta12) sin(theta12) 0; -sin(theta12) cos(theta12) 0; 0 0 1]
  U = U1 * U2 * U3
  x_e = U[1,:]
  x_mu = U[2,:]
  x_tau = U[3,:]


  
  xMS_e = [x_e[1]*sqrt((N-1)/N) x_e[2]*sqrt((N-1)/N) x_e[3]*sqrt((N-1)/N) x_e[1]*sqrt(1/N) x_e[2]*sqrt(1/N) x_e[3]*sqrt(1/N)]

  m_mu = sqrt(Delta_m_21+ (p.m_e)^2)
  m_tau= sqrt(p.Delta_m_31+ (p.m_e)^2)

  m_nu_sqrt = ((p.m_e)^2)* (N+(p.massfactor)^2 - 1)/(N) + Delta_m_21 * (xMS_e[2])^2 + p.Delta_m_31 * (xMS_e[3])^2 + ((p.massfactor)^2)*Delta_m_21*(xMS_e[5])^2 + ((p.massfactor)^2)*p.Delta_m_31*(xMS_e[6])^2

  return m_nu_sqrt

end

mkatrinsq= NeutrinoMass(params)

function nllh(params, osc_prob)
ll_value = -logpdf(TruncatedNormal(0.08, 0.295, 0.0, Inf), mkatrinsq)
return ll_value
end

function NeutrinoMassNND(params::NamedTuple,cfg::NND)
 
  U= get_PMNS(params)

  N = params[:N]
  r= params[:r]

  final, h, V = get_matrices(NND)
  x_e = U[1,:]
  x_1 = V[1,:]

  masses_SM_sq = get_abs_masses(params)^2

  delta_masses_NN = h

  for i in 1:3
    squared_x_e = x_e[i]^2

    for j in 1:N

      mass = masses_SM_sq[i]+delta_masses_NN[j]
      integrand= squared_x_e * x_1[j]^2 * mass
      sum += integrand
    end

    m_nu_sq += sum

  end

  return m_nu_sq

end

mkatrinsq= NeutrinoMass(params)


function NeutrinoMassGeneral(p::NamedTuple{(:sinsqrt2theta13, :Delta_m_31, :Species, :massfactor, :m_e, :theta23)}, U_mix)::Float64


  
  xMS_e = U_mix[1,:]

  m_nu_sq = (p.m_e)^2 *  xMS_e[1]^2 + (Delta_m_21 + (p.m_e)^2)* xMS_e[2]^2 + (p.Delta_m_31 + (p.m_e)^2)* xMS_e[3]^2 + (p.massfactor)^2 * (p.m_e)^2 * xMS_e[4]^2 + (p.massfactor)^2 *(Delta_m_21 + (p.m_e)^2)* xMS_e[5]^2+ (p.massfactor)^2 *(p.Delta_m_31 + (p.m_e)^2)* xMS_e[6]^2
  
  return m_nu_sq

end


function heaviside(t)
    0.5 * (sign(t) + 1)
 end=#

 

end