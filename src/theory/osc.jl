module osc
using LinearAlgebra
using StaticArrays

export ftype
export osc_kernel
export get_PMNS
export get_abs_masses

export standard
export sterile
export ADD
export Darkdim
export Darkdim_L

#using CUDA
const ftype = Float64

# TODO: implement neutrino vs. antineutrino


# Oscillation Kernel Simple
function osc_kernel(U::AbstractMatrix{<:Number}, H::AbstractVector, e::Real, l::Real)
    phase_factors = exp.(2.5338653580781976 * 1im * (l / e) .* H)
    p = U * Diagonal(phase_factors) * U'
    abs2.(p)
end

# Oscillation Kernel
function osc_kernel_smoothed(U::AbstractMatrix{<:Number}, H::AbstractVector, e::Real, l::Real; cutoff=Inf, damping=0, add=true)

    #cut off inaccessible states
    mask = sqrt.(abs.(H)) .< cutoff;

    if any(.!mask)
        H = H[mask];
        U_rest = U[:, .!mask]
        U = U[:, mask];
    else
        U_rest = 0
    end

    #H_sub = H .- minimum(H)
    
    phase_factors = 1im * (1 / e) .* H
    D = exp.(2.5338653580781976 * phase_factors .* l)
    decay = exp.(-2 * abs2.(H * damping))
    
    p = abs2.(U * Diagonal(D .* decay) * U') 
    
    if add
        p = p .+ abs2.(U_rest) * abs2.(U_rest)' .+ abs2.(U) * Diagonal((1 .- decay)) * abs2.(U)'
    end

    p
end

function osc_kernel(U::AbstractMatrix, H::AbstractMatrix, e::Real, l::Real)
    p = U * exp(2.5338653580781976 * 1im * (l/e) * H) * U'
    abs2.(p)
end

function get_PMNS(params)
    #T = float(typeof(params.θ₂₃))
    T = ftype

    U1 = SMatrix{3,3}(one(T), zero(T), zero(T), zero(T), cos(params.θ₂₃), -sin(params.θ₂₃), zero(T), sin(params.θ₂₃), cos(params.θ₂₃))
    U2 = SMatrix{3,3}(cos(params.θ₁₃), zero(T), -sin(params.θ₁₃)*exp(1im*params.δCP), zero(T), one(T), zero(T), sin(params.θ₁₃)*exp(-1im*params.δCP), zero(T), cos(params.θ₁₃))
    U3 = SMatrix{3,3}(cos(params.θ₁₂), -sin(params.θ₁₂), zero(T), sin(params.θ₁₂), cos(params.θ₁₂), zero(T), zero(T), zero(T), one(T))
    U = U1 * U2 * U3
end

function get_abs_masses(params)
    if params.H > 0
        m1 = params.m₀
        m2 = sqrt(params.Δm²₂₁ + params.m₀^2)
        m3 = sqrt(params.Δm²₃₁ + params.m₀^2)
    elseif params.H < 0
        m1 = sqrt(params.Δm²₃₁ + params.m₀^2)
        m2 = sqrt(params.Δm²₂₁ + params.Δm²₃₁ + params.m₀^2)
        m3 = params.m₀
    else
        error("Error: Please enter only either 1 for normal or -1 for inverted hierarchy.")
    end
    return m1, m2, m3
end

module standard
    using LinearAlgebra
    using DataStructures
    using StaticArrays
    using Distributions
    using ..osc

    function get_matrices(params)
        U = get_PMNS(params)
        H = Diagonal(SVector(zero(ftype),params.Δm²₂₁,params.Δm²₃₁));
        return U, H
    end
    
    # Oscillation over arrays of Energy (E) and Lnegth (L)
    function osc_prob(E, L, params; anti=false, use_cuda=false)
        U, H = get_matrices(params);
        Uc = anti ? conj.(U) : U
        #p = stack(map(e -> stack(map(l -> osc_kernel(U, diag(H), e, l), L)), E))
        #p = stack(map(x -> osc_kernel(U, diag(H), x[1], x[2]), Iterators.product(E, L)))
    
        if use_cuda
            p = stack(Array(broadcast((e, l) -> osc_kernel(Uc, diag(H), e, l), cu(E), cu(L'))))
        else
            p = stack(broadcast((e, l) -> osc_kernel(Uc, diag(H), e, l), E, L'))
        end
        Float64.(permutedims(p, (3, 4, 1, 2)))
    end
    
    params = OrderedDict()
    params[:θ₁₂] = ftype(asin(sqrt(0.307)))
    params[:θ₁₃] = ftype(asin(sqrt(0.021)))
    params[:θ₂₃] = ftype(asin(sqrt(0.57)))
    params[:δCP] = ftype(1.)
    params[:H] = 1
    params[:Δm²₂₁] = ftype(7.53e-5)
    if params[:H] > 0
        params[:Δm²₃₁] = ftype(2.4e-3 + params[:Δm²₂₁])
    else
        params[:Δm²₃₁] = ftype(-2.4e-3)
    end
    
    priors = OrderedDict()
    priors[:θ₁₂] = Uniform(atan(sqrt(0.2)), atan(sqrt(1)))
    priors[:θ₁₃] = Uniform(ftype(0.1), ftype(0.2))
    priors[:θ₂₃] = Uniform(ftype(pi/4 *2/3), ftype(pi/4 *4/3))
    priors[:δCP] = Uniform(ftype(0), ftype(2*π))
    priors[:H] = params[:H]
    priors[:Δm²₂₁] = Uniform(ftype(6.5e-5), ftype(9e-5))
    if params[:H] > 0
        priors[:Δm²₃₁] = Uniform(ftype(2e-3), ftype(3e-3))
    else
        priors[:Δm²₃₁] = Uniform(ftype(-3e-3), ftype(-2e-3))
    end
end

module sterile
    using Distributions
    using ..osc
    using LinearAlgebra

    function get_matrices(params)
    
        H = [0 ,params.Δm²₂₁,params.Δm²₃₁,params.Δm²₄₁]
     
        R14 = [cos(params.θ₁₄) 0 0 sin(params.θ₁₄); 0 1 0 0; 0 0 1 0; -sin(params.θ₁₄) 0 0 cos(params.θ₁₄)]
        R24 = [1 0 0 0; 0 cos(params.θ₂₄) 0 sin(params.θ₂₄); 0 0 1 0; 0 -sin(params.θ₂₄) 0 cos(params.θ₂₄)]
        R34 = [1 0 0 0; 0 1 0 0; 0 0 cos(params.θ₃₄) sin(params.θ₃₄); 0 0 -sin(params.θ₃₄) cos(params.θ₃₄)]
        
        U = get_PMNS(params)
        
        U_sterile = R34 * R24 * R14 * hcat(vcat(U, [0 0 0]), [0 0 0 1]')
        
        return U_sterile, H
    end

    params = copy(standard.params)
    params[:Δm²₄₁] = 1
    params[:θ₁₄] = 0.1
    params[:θ₂₄] = 0.1
    params[:θ₃₄] = 0.1

    # Missing
    priors = nothing
end

module ADD
    using Distributions
    using ..osc
    using LinearAlgebra
    using Zygote

    function get_matrices(params)
        N_KK = 5
        
        # um to eV
        umev = 5.067730716156395
        PMNS = get_PMNS(params)
    
        m1, m2, m3 = get_abs_masses(params)
    
        # MD is the Dirac mass matrix that appears in the Lagrangian.
        MD = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
        aM1 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
        aM2 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
    
        # init buffers
        for i in 1:3*(N_KK+1)
            for j in 1:3*(N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
    
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.ADD_radius * MD[i, j] * umev
            end
        end
        
        for n in 1:N_KK
            for i in 1:3
                for j in 1:3
                    aM1[3*n + i, j] = sqrt(2) * params.ADD_radius * MD[i, j] * umev
                end
            end
        end
    
        for i in 1:N_KK
            aM2[3*i + 1, 3*i + 1] = i
            aM2[3*i + 2, 3*i + 2] = i
            aM2[3*i + 3, 3*i + 3] = i
        end
    
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
    
        h, U = eigen(aaMM)
        h = h / (params.ADD_radius^2 * umev^2.)
        return U, h
    end

    # Oscillation over arrays of Energy (E) and Lnegth (L)
    function osc_prob(E, L, params; anti=false)
        U, H = get_matrices(params);
        Uc = anti ? conj.(U) : U
        p = stack(broadcast((e, l) -> osc_kernel(Uc, H, e, l), E, L'))
    
        Float64.(permutedims(p, (3, 4, 1, 2)))
    end

    params = copy(standard.params)
    params[:m₀] = ftype(0.01)
    params[:ADD_radius] = ftype(1e-2)

    priors = copy(standard.priors)
    priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
    priors[:ADD_radius] = LogUniform(ftype(1e-3),ftype(1))

end

module Darkdim
    using Distributions
    using ..osc
    using LinearAlgebra
    using Zygote

    function get_matrices(params)
        N_KK = 5
        
        # um to eV
        umev = 5.067730716156395
        PMNS = get_PMNS(params)
    
        m1, m2, m3 = get_abs_masses(params)
    
        m1_MD = m1 * sqrt((exp(2 * π * params.ca1) - 1) / (2 * π * params.ca1))
        m2_MD = m2 * sqrt((exp(2 * π * params.ca2) - 1) / (2 * π * params.ca2))
        m3_MD = m3 * sqrt((exp(2 * π * params.ca3) - 1) / (2 * π * params.ca3))
        
        #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.
        
        # Compute MDc00
        MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
        # Initialize aM1 matrix
        aM1 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
        aM2 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
        # init buffers
        for i in 1:3*(N_KK+1)
            for j in 1:3*(N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
        
        # Fill in the aM1 matrix for the first term
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.Darkdim_radius * MDc00[i, j] * umev
            end
        end
    
        # Update aM1 matrix for the second term
        for n in 1:N_KK
            MDcoff = PMNS * Diagonal([
                m1_MD * sqrt(n^2 / (n^2 + params.ca1^2)),
                m2_MD * sqrt(n^2 / (n^2 + params.ca2^2)),
                m3_MD * sqrt(n^2 / (n^2 + params.ca3^2))
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM1[3 * n + i, j] = sqrt(2) * params.Darkdim_radius * MDcoff[i, j] * umev
                end
            end
        end
    
        # Fill in the aM2 matrix
        for n in 1:N_KK
            aMD2 = PMNS * Diagonal([
                sqrt(n^2 + params.ca1^2),
                sqrt(n^2 + params.ca2^2),
                sqrt(n^2 + params.ca3^2)
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM2[3 * n + i, 3 * n + j] = aMD2[i, j]
                end
            end
        end
    
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
    
        h, U = eigen(aaMM)
        h = h / (params.Darkdim_radius^2 * umev^2)
    
        return U, h
    end


    # Oscillation over arrays of Energy (E) and Lnegth (L)
    function osc_prob(E, L, params; anti=false)
        U, H = get_matrices(params);
        Uc = anti ? conj.(U) : U
        p = stack(broadcast((e, l) -> osc_kernel_smoothed(Uc, H, e, l, cutoff=0.5), E, L'))
    
        Float64.(permutedims(p, (3, 4, 1, 2)))
    end

    params = copy(standard.params)
    params[:m₀] = ftype(0.01)
    params[:ca1] = ftype(1e-4)
    params[:ca2] = ftype(1e-4)
    params[:ca3] = ftype(1e-4)
    params[:Darkdim_radius] = ftype(1e-2)
    
    priors = copy(standard.priors)
    priors[:m₀] = LogUniform(ftype(1e-3),ftype(1))
    priors[:ca1] = LogUniform(ftype(1e-5), ftype(10))
    priors[:ca2] = LogUniform(ftype(1e-5), ftype(10))
    priors[:ca3] = LogUniform(ftype(1e-5), ftype(10))
    priors[:Darkdim_radius] = LogUniform(ftype(1e-3),ftype(1))

end

module Darkdim_L
    using Distributions
    using ..osc
    using LinearAlgebra
    using Zygote

    function get_matrices(params)
    
        N_KK = 5
        MP = 2.435e18 # GeV
        M5 = 1e6 # GeV
        vev = 174e9 # eV
        lambda_list = [params.λ₁, params.λ₂, params.λ₃]
        m1_MD, m2_MD, m3_MD = (vev * M5 / MP) .* lambda_list
    
    
        m1 = m1_MD * (sqrt(2 * π * params.ca1 / (exp(2 * π * params.ca1) - 1)))
        m2 = m2_MD * (sqrt(2 * π * params.ca2 / (exp(2 * π * params.ca2) - 1)))
        m3 = m3_MD * (sqrt(2 * π * params.ca3 / (exp(2 * π * params.ca3) - 1)))
        
        # um to eV
        umev = 5.067730716156395
        PMNS = get_PMNS(params)    
        
        #MD is the Dirac mass matrix that appears in the Lagrangian. Note the difference with ADD through the multiplication by c.
        
        # Compute MDc00
        MDc00 = PMNS * Diagonal([m1, m2, m3]) * adjoint(PMNS)
    
        # Initialize aM1 matrix
        aM1 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
        aM2 = Zygote.Buffer(PMNS, 3*(N_KK+1), 3*(N_KK+1))
        # init buffers
        for i in 1:3*(N_KK+1)
            for j in 1:3*(N_KK+1)
                aM1[i,j] = 0.
                aM2[i,j] = 0.
            end
        end
        
        # Fill in the aM1 matrix for the first term
        for i in 1:3
            for j in 1:3
                aM1[i, j] = params.Darkdim_radius * MDc00[i, j] * umev
            end
        end
    
        # Update aM1 matrix for the second term
        for n in 1:N_KK
            MDcoff = PMNS * Diagonal([
                m1_MD * sqrt(n^2 / (n^2 + params.ca1^2)),
                m2_MD * sqrt(n^2 / (n^2 + params.ca2^2)),
                m3_MD * sqrt(n^2 / (n^2 + params.ca3^2))
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM1[3 * n + i, j] = sqrt(2) * params.Darkdim_radius * MDcoff[i, j] * umev
                end
            end
        end
    
        # Fill in the aM2 matrix
        for n in 1:N_KK
            aMD2 = PMNS * Diagonal([
                sqrt(n^2 + params.ca1^2),
                sqrt(n^2 + params.ca2^2),
                sqrt(n^2 + params.ca3^2)
            ]) * adjoint(PMNS)
            for i in 1:3
                for j in 1:3
                    aM2[3 * n + i, 3 * n + j] = aMD2[i, j]
                end
            end
        end
    
        aM = copy(aM1) + copy(aM2)
        aaMM = Hermitian(conj(transpose(aM)) * aM)
    
        h, U = eigen(aaMM)
        h = h / (params.Darkdim_radius^2 * umev^2) 
        return U, h
    end 
    
    # Oscillation over arrays of Energy (E) and Lnegth (L)
    function osc_prob(E, L, params; anti=false)
        U, H = get_matrices(params);
        Uc = anti ? conj.(U) : U
    
        p = stack(broadcast((e, l) -> osc_kernel_smoothed(Uc, H, e, l, damping=2.), E, L'))
    
        Float64.(permutedims(p, (3, 4, 1, 2)))
    end

    params = copy(standard.params)
    pop!(params, :Δm²₂₁)
    pop!(params, :Δm²₃₁)
    pop!(params, :H)
    
    # params[:Darkdim_radius] = ftype(1.9/5.067730716156395)
    # params[:ca1] = ftype(0.43)
    # params[:ca2] = ftype(1.)
    # params[:ca3] = ftype(0.41)
    # params[:λ₁] = ftype(0.42)
    # params[:λ₂] = ftype(2.4)
    # params[:λ₃] = ftype(1.7)
    
    #params[:θ₁₂] = ftype(0.5289)
    #params[:θ₁₃] = ftype(0.15080)
    #params[:θ₂₃] = ftype(0.93079)
    #params[:δCP] = ftype(0.683)
    
    # Machado P1:
    # params[:Darkdim_radius] = ftype(1.9/5.067730716156395)
    # params[:ca1] = ftype(4.24)
    # params[:ca2] = ftype(1.19)
    # params[:ca3] = ftype(-0.037)
    # params[:λ₁] = ftype(0.42)
    # params[:λ₂] = ftype(2.0)
    # params[:λ₃] = ftype(0.66)
    
    # my P1
    params[:Darkdim_radius] = 2.50529
    params[:ca1] = 3.99996
    params[:ca2] = 1.98377
    params[:ca3] = -9.46609
    params[:δCP] = 3.25601
    params[:θ₁₂] = 0.608085
    params[:θ₁₃] = 0.143864
    params[:θ₂₃] = 0.670785
    params[:λ₁] = 0.221826
    params[:λ₂] = 0.000558143
    params[:λ₃] = 0.0902954

    priors = copy(standard.priors)
    pop!(priors, :Δm²₂₁)
    pop!(priors, :Δm²₃₁)
    pop!(priors, :H)
    priors[:Darkdim_radius] = LogUniform(ftype(4),ftype(6))
    
    priors[:ca1] = Uniform(ftype(1e-7), ftype(10)) #LogUniform(ftype(1e-5), ftype(100))
    priors[:ca2] = Uniform(ftype(1e-7), ftype(10))#LogUniform(ftype(1e-5), ftype(100))
    priors[:ca3] = Uniform(-ftype(10), -ftype(1e-7))#LogUniform(ftype(1e-5), ftype(100))
    
    priors[:λ₁] = Uniform(ftype(0), ftype(5))
    priors[:λ₂] = Uniform(ftype(0), ftype(5))
    priors[:λ₃] = Uniform(ftype(0), ftype(5))

end


end
