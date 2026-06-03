# Custom Physics

By default, each experiment configures its own physics modules. You can override this to use custom oscillation models, inverted ordering, or BSM physics.

## Custom Oscillation Configuration

```julia
using Newtrinos

osc_cfg = Newtrinos.osc.OscillationConfig(
    flavour     = Newtrinos.osc.ThreeFlavour(ordering=:IO),
    propagation = Newtrinos.osc.Basic(),
    states      = Newtrinos.osc.All(),
    interaction = Newtrinos.osc.SI(),
)

osc = Newtrinos.osc.configure(osc_cfg)
```

## Assembling a Physics Bundle

Atmospheric experiments need oscillations, fluxes, Earth density, and cross-sections:

```julia
atm_flux     = Newtrinos.atm_flux.configure()
earth_layers = Newtrinos.earth_layers.configure()
xsec         = Newtrinos.xsec.configure()

physics = (; osc, atm_flux, earth_layers, xsec)
```

## Passing Physics to Experiments

```julia
experiments = (
    deepcore = Newtrinos.deepcore.configure(physics),
    dayabay  = Newtrinos.dayabay.configure(physics),
)
```

Each experiment extracts whatever physics modules it needs. Reactor experiments (Daya Bay, KamLAND) only use `osc`; atmospheric experiments use all four.

## Available Flavour Models

- `ThreeFlavour(ordering=:NO)` / `ThreeFlavour(ordering=:IO)` — standard 3-flavour
- `Sterile()` — 3+1 sterile neutrino model
- `ADD()` — large extra dimensions
- `Darkdim_tower()`, `Darkdim_lightest()` — dark dimension models

## Available Interaction Models

- `Vacuum()` — vacuum oscillations only
- `SI()` — standard matter effects (MSW)
- `NSI()` — non-standard interactions

## Available Propagation Models

- `Basic()` — standard propagation
- `Decoherent()` — quantum decoherence
- `Damping()` — damped oscillations
