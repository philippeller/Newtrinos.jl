# Automatic Differentiation

All code in Newtrinos.jl is fully differentiable via AutoDiff, using `ForwardDiff.jl` under the hood. This means you can compute exact gradients of the joint log-likelihood with respect to all physics and nuisance parameters — which is essential for gradient-based optimizers and samplers.

## Configuration

We must first configure the experiments (with default physics):

```julia
using Newtrinos, DensityInterface

experiments = (
    deepcore = Newtrinos.deepcore.configure(),
    dayabay = Newtrinos.dayabay.configure(),
    kamland = Newtrinos.kamland.configure(),
    minos = Newtrinos.minos.configure(),
    orca = Newtrinos.orca.configure(),
);
```

This is enough to generate a joint likelihood and collect the nominal parameters for everything:

```julia
likelihood = Newtrinos.generate_likelihood(experiments)
params = Newtrinos.get_params(experiments)
priors = Newtrinos.get_params(experiments)
```

## Computing gradients

In a neutrino fit the gradient is useful for:

- **Gradient-based optimizers** (e.g. L-BFGS) – finding the best-fit point much faster than grid scans
- **HMC/NUTS samplers** – needed for efficient Bayesian posterior sampling via BAT.jl
- **Sensitivity studies** – large $|\partial \text{llh} / \partial \theta |$ means the data is highly sensitive to that parameter 

We can calculate gradients directly using ForwardDiff 

```julia
using ForwardDiff

# Wrap the likelihood as a plain function of parameters
f(params) = logdensityof(likelihood, params)

# Evaluate the gradient at nominal parameter values
grad = ForwardDiff.gradient(f, params)
```

The result is a `NamedTuple` with the same fields as `p`, giving you ∂llh/∂θ for every parameter:

```julia
grad.δCP     # ∂llh/∂δCP
grad.θ₂₃     # ∂llh/∂θ₂₃
grad.Δm²₃₁  # ∂llh/∂Δm²₃₁
```

The gradient calculation can be automatically activated in likelihood scans by setting `gradient_map=true`. Then, at each grid point the ForwardDiff.gradient is evaluated and the per-parameter gradient is included in the `NewtrinosResult.values`.

```julia
result=Newtrinos.scan(likelihood, priors, vars_to_scan, params, gradient_map=true)
```