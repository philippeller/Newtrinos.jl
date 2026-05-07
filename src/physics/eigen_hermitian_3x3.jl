using StaticArrays
using LinearAlgebra

"""
    fast_eigen(C::Hermitian{T, <:SMatrix{3,3,T}}; sortby=identity) -> Eigen

Analytic eigendecomposition of a 3×3 Hermitian matrix using closed-form expressions.

Implements the algorithm of Deledalle et al. (2017),
[hal-01501221](https://hal.archives-ouvertes.fr/hal-01501221/document).
This avoids the iterative LAPACK `eigen` call and is significantly faster for the
small matrices that arise in three-flavour neutrino oscillations with matter effects.

Called by [`BargerEigen`](@ref) via [`decompose`](@ref).

**Eigenvalue algorithm:** Computes the three roots of the characteristic polynomial
via the Cardano trigonometric method. With ``x_1`` and ``x_2`` defined from the matrix
elements, the discriminant ``\\Delta = 4x_1^3 - x_2^2 \\geq 0`` and the roots are

```math
\\lambda_k = \\frac{a+b+c}{3} + \\frac{2\\sqrt{x_1}}{3}
\\cos\\!\\left(\\frac{\\phi + 2\\pi(k-1)}{3}\\right), \\quad k = 1,2,3
```

where ``\\phi = \\operatorname{atan}(\\sqrt{\\Delta},\\, x_2)``.

**Eigenvector algorithm:** Each eigenvector is obtained as the cross product of two rows
of ``(C - \\lambda_k I)``. Three row-pair combinations are tried in order to handle
degenerate or near-degenerate cases where a cross product vanishes.

Dispatches to specialized helpers [`_fast_eigen_efzero`](@ref),
[`_fast_eigen_edzero`](@ref), [`_fast_eigen_dfzero`](@ref) when pairs of
off-diagonal elements are zero (reducing to 2×2 subproblems).

The matrix elements are labelled as follows (`a`, `b`, `c` are real diagonal entries;
`d`, `e`, `f` are complex lower-triangle entries):

```math
C = \\begin{pmatrix}
      a       & d^*     & f^* \\\\
      d       & b       & e^* \\\\
      f       & e       & c
    \\end{pmatrix}
```

# Arguments
- `C::Hermitian{T, <:SMatrix{3,3,T}}`: the 3×3 Hermitian matrix to decompose.
- `sortby::F = identity`: ordering function applied to eigenvalues (e.g. `abs`, `real`).

# Returns
An `Eigen` object with `values::SVector{3}` and `vectors::SMatrix{3,3}`.
"""
function fast_eigen(
        C::Hermitian{T, <:SMatrix{3,3,T}};
        sortby::F = identity,
    ) where {T, F}
    R = real(T)
    
    a = real(C[1, 1])
    b = real(C[2, 2])
    c = real(C[3, 3])
    d = C[2, 1]
    e = C[3, 2]
    f = C[3, 1]
    
    d2 = abs2(d)
    e2 = abs2(e)
    f2 = abs2(f)

    # edge cases that may break fast_eigen: 1) e=f=0, 2) e=d=0, 3) d=f=0 
    if iszero(e) && iszero(f)
        return _fast_eigen_efzero(C; sortby)
    elseif iszero(e) && iszero(d)
        return _fast_eigen_edzero(C; sortby)
    elseif iszero(f) && iszero(d)
        return _fast_eigen_dfzero(C; sortby)
    end

    
    # For numerical stability, we rewrite x1 in a way that ensures its
    # positiveness.
    # Concretely, we use:
    #   a^2 + b^2 + c^2 - ab - ac - bc =
    #           [(a - b)^2 + (b - c)^2 + (a - c)^2] / 2
    #
    # Note: it's easy to show that x1 == 0 iif the matrix C is proportional to
    # the identity matrix. In that case, x2 must also be 0.
    x1 = R(0.5) * (abs2(a - b) + abs2(b - c) + abs2(a - c)) +
        3 * (d2 + e2 + f2)

    x2 = let
        A = 2a - b - c
        B = 2b - a - c
        C = 2c - a - b
        (
            - A * B * C
            + 9 * (C * d2 + B * f2 + A * e2)
            - 54 * real(conj(d) * conj(e) * f)
        )
    end

    # This seems to be more numerically stable than Δ = 4 * x1^3 - x2^2
    Δ = let
        y1 = 2 * sqrt(x1^3)
        (y1 + x2) * (y1 - x2)
    end

    if Δ < 0
        # In this case, sqrt(-Δ) should be really small wrt x2
        @assert -Δ / x2^2 < sqrt(eps(R))
        Δ = zero(Δ)
    end

    φ = atan(sqrt(Δ), x2)

    λs = let
        A = (a + b + c) / 3
        B = 2 * sqrt(x1) / 3
        SVector(
            A - B * cos(φ / 3),
            A + B * cos((φ - π) / 3),
            A + B * cos((φ + π) / 3),
        )
    end

    us = map(λs) do λ
        α = d * (c - λ) - conj(e) * f
        β = f * (b - λ) - d * e
        u = SVector(β * (λ - c) - α * e, α * f, β * f)
        
        # provide fall backs for edge cases where kernel rows are parallel 
        # and therefore we get a null eigenvector from the row cross product
        # if this happens then the fallbacks select another row combination for the calculation
        if all(iszero, u)
            # rows 2&3 cross product was zero; try rows 1&3
            u = SVector(
                conj(d) * (c - λ) - e * conj(f),
                abs2(f) - (a - λ) * (c - λ),
                (a - λ) * e - conj(d) * f,
            )
        end
        if all(iszero, u)
            # still zero; try rows 1&2 (handles all degenerate single-eigenvalue cases)
            u = SVector(
                conj(d) * conj(e) - (b - λ) * conj(f),
                conj(f) * d - (a - λ) * conj(e),
                (a - λ) * (b - λ) - abs2(d),
            )
        end
        # normalize the eigenvector
        normalize(u)
    end

    vs = hcat(us...)

    _sorted_eigen(sortby, λs, vs)
end

"""
    _fast_eigen_efzero(C; sortby=identity) -> Eigen

Edge-case eigendecomposition when `C[3,1] == C[3,2] == 0` (off-diagonal elements `e=f=0`).

Row 3 decouples: ``\\lambda_3 = C_{33}`` with eigenvector ``(0, 0, 1)``. The remaining
two eigenvalues and vectors are obtained by calling Julia's `eigen` on the 2×2 upper-left
submatrix ``C_{1:2,1:2}``, then embedding the result back into 3×3.
"""
function _fast_eigen_efzero(
        C::Hermitian{T, <:SMatrix{3,3,T}};
        sortby::F = identity,
    ) where {T, F}
    Csub = let A = parent(C)
        Asub = SA[A[1, 1] A[1, 2]; A[2, 1] A[2, 2]]
        Hermitian(Asub)
    end
    Esub = eigen(Csub)  # 2×2 Hermitian eigendecomposition
    values = SVector(Esub.values..., C[3, 3])
    vs = Esub.vectors
    vectors = @SMatrix [
        vs[1, 1] vs[1, 2] 0;
        vs[2, 1] vs[2, 2] 0;
        0        0        1;
    ]
    _sorted_eigen(sortby, values, vectors)
end

"""
    _fast_eigen_edzero(C; sortby=identity) -> Eigen

Edge-case eigendecomposition when `C[1,2] == C[2,3] == 0` (off-diagonal elements `e=d=0`).

Row 2 decouples: ``\\lambda_2 = C_{22}`` with eigenvector ``(0, 1, 0)``. The remaining
two eigenvalues are found from the 2×2 submatrix of rows/columns 1 and 3.
"""
function _fast_eigen_edzero(
        C::Hermitian{T, <:SMatrix{3,3,T}};
        sortby::F = identity,
    ) where {T, F}
    Csub = let A = parent(C)
        Asub = SA[A[1, 1] A[1, 3]; A[3, 1] A[3, 3]]
        Hermitian(Asub)
    end
    Esub = eigen(Csub)  # 2×2 Hermitian eigendecomposition
    values = SVector(Esub.values[1], C[2, 2], Esub.values[2])
    vs = Esub.vectors
    vectors = @SMatrix [
        vs[1, 1] 0 vs[1, 2];
        0 1 0;
        vs[2, 1] 0 vs[2, 2];
    ]
    _sorted_eigen(sortby, values, vectors)
end

"""
    _fast_eigen_dfzero(C; sortby=identity) -> Eigen

Edge-case eigendecomposition when `C[1,2] == C[1,3] == 0` (off-diagonal elements `d=f=0`).

Row 1 decouples: ``\\lambda_1 = C_{11}`` with eigenvector ``(1, 0, 0)``. The remaining
two eigenvalues are found from the 2×2 lower-right submatrix ``C_{2:3,2:3}``.
"""
function _fast_eigen_dfzero(
        C::Hermitian{T, <:SMatrix{3,3,T}};
        sortby::F = identity,
    ) where {T, F}
    Csub = let A = parent(C)
        Asub = SA[A[2, 2] A[2, 3]; A[3, 2] A[3, 3]]
        Hermitian(Asub)
    end
    Esub = eigen(Csub)
    values = SVector(C[1, 1], Esub.values[1], Esub.values[2] )
    vs = Esub.vectors
    vectors = @SMatrix [
        1 0 0;
        0 vs[1, 1] vs[1, 2];
        0 vs[2, 1] vs[2, 2];
    ]
    _sorted_eigen(sortby, values, vectors)
end

"""
    _sorted_eigen(::Nothing, vals, vecs) -> Eigen

Passthrough overload that wraps eigenvalues and eigenvectors into an `Eigen` object
without reordering. Called when `sortby = nothing`.
"""
_sorted_eigen(::Nothing, args...) = Eigen(args...)

"""
    _sorted_eigen(by::F, vals, vecs) -> Eigen

Sort eigenvalues by the key function `by` and permute eigenvector columns accordingly.

# Arguments
- `by::F`: ordering key function (e.g. `identity`, `abs`, `real`).
- `vals`: eigenvalue vector.
- `vecs`: eigenvector matrix (columns correspond to eigenvalues).

# Returns
An `Eigen` object with eigenvalues sorted by `by` and eigenvectors permuted to match.
"""
@inline function _sorted_eigen(by::F, vals, vecs) where {F}
    p = sortperm(vals; by)
    λs = vals[p]
    vs = vecs[:, p]
    Eigen(λs, vs)
end

"""
    _sorted_eigen(by::Fun, F::Eigen) -> Eigen

Convenience overload that unpacks an `Eigen` object and delegates to
`_sorted_eigen(by, F.values, F.vectors)`.
"""
@inline _sorted_eigen(by::Fun, F::Eigen) where {Fun} =
    _sorted_eigen(by, F.values, F.vectors)
