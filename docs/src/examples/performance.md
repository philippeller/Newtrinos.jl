# Performance

## Hot Path: Oscillation Probabilities

`osc_prob` is the performance-critical function. For 3-flavour vacuum oscillations, it uses `SMatrix`/`SVector` from StaticArrays.jl for zero-allocation computation. Matter effects require eigendecomposition which allocates (~19 allocs/call).

## ForwardDiff Chunk Size

ForwardDiff processes parameters in chunks of 12 by default. With N parameters, gradient computation costs `ceil(N/12)` forward passes. For example, 21 parameters (typical for DeepCore) requires 2 passes.

## Response Matrix Contractions

Super-Kamiokande uses `contract_R` with pre-flattened Float64 matrices for BLAS-accelerated matrix-vector multiply, avoiding Dual number broadcast over large arrays.

## Benchmarking

Run the likelihood benchmark:

```bash
julia -t 4 --project benchmark/bench_likelihood.jl --experiments deepcore dayabay
```

This benchmarks both likelihood evaluation and gradient computation, comparing ForwardDiff and PolyesterForwardDiff (threaded chunked differentiation).

Run the oscillation benchmark:

```bash
julia --project benchmark/bench_osc.jl
```

## Tips

- Use `julia -t N` to enable threading for scan/profile tasks
- For large grids, use `--workers N` for distributed parallelism
- Profile caching (`cache_dir`) prevents redundant MLE computations when re-running
- PolyesterForwardDiff provides ~1.5x speedup over ForwardDiff when parameters > chunk size (12)
