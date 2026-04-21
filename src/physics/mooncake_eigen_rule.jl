using LinearAlgebra
using Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, primal, tangent, NoRData, rrule!!
using ..Newtrinos

# Wrapper that takes a plain matrix and returns (eigenvalues, eigenvectors).
# Mooncake can't differentiate through LAPACK's eigen, so we mark this as a
# primitive and provide a hand-written reverse-mode rule.
function _eigen_hermitian(A::Matrix{T}) where T<:Union{ComplexF64, Float64}
    E = eigen(Hermitian(A))
    return E.values, E.vectors
end

@is_primitive MinimalCtx Tuple{typeof(_eigen_hermitian), Matrix{ComplexF64}}
@is_primitive MinimalCtx Tuple{typeof(_eigen_hermitian), Matrix{Float64}}

function Mooncake.rrule!!(
    ::CoDual{typeof(_eigen_hermitian)},
    A_cd::CoDual{Matrix{T}}
) where T
    A = primal(A_cd)
    dA = tangent(A_cd)
    E = eigen(Hermitian(A))
    λ = E.values
    U = E.vectors
    n = length(λ)

    # Shadow arrays for output gradients (Mooncake accumulates into these)
    dλ = zero(λ)
    dU = zero(U)

    function eigen_pb!!(::NoRData)
        # F[i,j] = 1/(λ[j] - λ[i]) for i≠j, 0 for i==j
        F = zeros(eltype(U), n, n)
        for j in 1:n, i in 1:n
            i != j && (F[i, j] = inv(λ[j] - λ[i]))
        end

        # Standard Hermitian eigen reverse formula:
        #   dH = U * (F .* (U' * dU) + Diagonal(dλ)) * U'
        tmp = U' * dU
        dH = U * (F .* tmp + Diagonal(real.(dλ))) * U'

        # Propagate Hermitian gradient to plain matrix (symmetrize)
        for j in 1:n, i in 1:n
            dA[i, j] += (dH[i, j] + conj(dH[j, i])) / 2
        end

        # Reset output shadows for potential reuse
        fill!(dλ, zero(eltype(dλ)))
        fill!(dU, zero(eltype(dU)))
        return NoRData(), NoRData()
    end
    return CoDual((λ, U), (dλ, dU)), eigen_pb!!
end

# Route DefaultEigen through _eigen_hermitian for non-static matrices,
# so Mooncake can use the custom rule above.
function Newtrinos.osc.decompose(H::Hermitian{T, Matrix{T}}, ::Newtrinos.osc.DefaultEigen) where T
    λ, U = _eigen_hermitian(parent(H))
    Eigen(λ, U)
end
