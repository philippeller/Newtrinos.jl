using Test
using LinearAlgebra
using StaticArrays
using Newtrinos 

#define test cases with known eigenvalues for testing fast_eigen function
# Case 0: identity matrix -> eigenvalues are all 1
Cid = Hermitian(SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
λid_expected = [1.0, 1.0, 1.0]

# Case 1: diagonal -> eigenvalues are just the diagonal entries 1, 2, 3
C1 = Hermitian(SMatrix{3,3}([3.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0]))
λ1_expected = [1.0, 2.0, 3.0]

# Case 2: complex Hermitian, no null entries -> eigenvalues = 1,2,5
C2 = Hermitian(SMatrix{3,3}([3.0 1.0-1.0im 1.0im; 1.0+1.0im 3.0 -1.0; 1.0im -1.0 2.0]))
λ2_expected =[1.0, 2.0, 5.0]

# Case 3: real Hermitian, no null entries -> eigenvalues = 1,5,6
C3 = Hermitian(SMatrix{3,3}([2.0 1.0 -1.0; 1.0 5.0 2.0; -1.0 2.0 5.0]))
λ3_expected =[1.0, 4.0, 7.0]

#Case 4: e=f=0 -> Submatrix [[3,1+im],[1-im,4]]: trace=7, det=10 → λ=(7±3)/2 = 2, 5  → eigenvalues 2, 5, 6
C4 = Hermitian(SMatrix{3,3}([3.0 1.0+im 0.0; 1.0-im 4.0 0.0; 0.0 0.0 6.0]))
λ4_expected = [2.0, 5.0, 6.0]

# Case 5: d=e=0
C5 = Hermitian(SMatrix{3,3}([5.0 0.0 2.0; 0.0 7.0 0.0; 2.0 0.0 5.0]))
λ5_expected = [3.0, 7.0, 7.0]

# Case 6: d=f=0
C6 = Hermitian(SMatrix{3,3}([1.0 0.0 0.0; 0.0 4.0 2.0; 0.0 2.0 4.0]))
λ6_expected = [1.0, 2.0, 6.0]

all_cases= [(Cid, λid_expected), (C1, λ1_expected), (C2, λ2_expected), (C3, λ3_expected), (C4, λ4_expected), (C5, λ5_expected), (C6, λ6_expected)]

@testset "fast_eigen 3x3 Hermitian" begin

    @testset "general properties" begin
        for C in [Cid, C1, C2, C3, C4, C5, C6]
            F = Newtrinos.fast_eigen(C, sortby=real)
            V, λ = F.vectors, F.values

            @test all(isreal, λ) # real Eigenvalues 
            @test V' * V ≈ I atol=1e-6 # Eigenvectors are orthonormal
            @test V * Diagonal(real.(λ)) * V' ≈ Matrix(C) atol=1e-12 # Reconstruction: V diag(λ) V' == C
            @test issorted(real.(λ)) # Default sort is ascending
        end
    end
    
    @testset "eigenvalue values" begin
        for (C, λ_expected) in all_cases
            F = Newtrinos.fast_eigen(C, sortby=real)
            @test real.(F.values) ≈ λ_expected atol=1e-10
        end
    end 

    @testset "efzero case" begin
        Fgeneral = Newtrinos.fast_eigen(C4, sortby=real)
        Fefzero  = Newtrinos._fast_eigen_efzero(C4, sortby=real)
        # Both paths must agree on eigenvalues and reconstruction
        @test real.(Fefzero.values) ≈ real.(Fgeneral.values) atol=1e-12
        @test Fefzero.vectors ≈ Fgeneral.vectors atol=1e-12
    end

    @testset "edzero case" begin
        Fgeneral = Newtrinos.fast_eigen(C5, sortby=real)
        Fedzero  = Newtrinos._fast_eigen_edzero(C5, sortby=real)
        # Both paths must agree on eigenvalues and reconstruction
        @test real.(Fedzero.values) ≈ real.(Fgeneral.values) atol=1e-12
        @test Fedzero.vectors ≈ Fgeneral.vectors atol=1e-12
    end

    @testset "dfzero case" begin
        Fgeneral = Newtrinos.fast_eigen(C6, sortby=real)
        Fdfzero  = Newtrinos._fast_eigen_dfzero(C6, sortby=real)
        # Both paths must agree on eigenvalues and reconstruction
        @test real.(Fdfzero.values) ≈ real.(Fgeneral.values) atol=1e-12
        @test Fdfzero.vectors ≈ Fgeneral.vectors atol=1e-12
    end

    @testset "Sorting Logic" begin
        # Create an additional simple diagonal matrix with known eigenvalues 
        A = Hermitian(SMatrix{3,3,Float64}(2.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 10.0))
        
        # 1. Sort by identity (follows the order in the SVector λs)
        E_def = Newtrinos.fast_eigen(A, sortby=identity)
        @test issorted(E_def.values)

        # 2. Sort by absolute value (Magnitude)
        E_abs = Newtrinos.fast_eigen(A, sortby=abs)
        @test E_abs.values == SVector(2.0, -5.0, 10.0)
        # Verify eigenvectors moved with values: Ax = λx
        for i in 1:3
            @test A * E_abs.vectors[:, i] ≈ E_abs.values[i] * E_abs.vectors[:, i]
        end

        # 3. Descending order
        E_rev = Newtrinos.fast_eigen(A, sortby=x -> -x)
        @test E_rev.values == SVector(10.0, 2.0, -5.0)
    end

end