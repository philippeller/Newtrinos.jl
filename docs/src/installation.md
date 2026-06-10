# Installation

## 1. Install Julia

Newtrinos.jl is written in the [Julia programming language](https://julialang.org/).
Download and install it from the [official downloads page](https://julialang.org/downloads/).

---

## 2. Install Newtrinos.jl

Newtrinos.jl is not yet registered in the Julia General registry, so it must be
installed directly from GitHub.

Open a Julia session by running `julia` in your terminal (or launching the Julia
application), then run:

```julia
using Pkg
Pkg.add(url="https://github.com/philippeller/Newtrinos.jl.git")
```

Julia will download the package and all its dependencies automatically.
This may take a few minutes the first time.

!!! note
    You only need to do this once. After installation the package is available
    in any Julia session on your machine.

---

## 3. Load the Package

After installation, load Newtrinos.jl at the start of any Julia session:

```julia
using Newtrinos
```

---

## Development Installation (optional)

If you want to modify the source code or contribute to Newtrinos.jl, clone the
repository and activate it as a local project.

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/philippeller/Newtrinos.jl.git
cd Newtrinos.jl
```

**Step 2 — Install dependencies:**

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

**Step 3 — Activate the local project in Julia:**

```julia
using Pkg
Pkg.activate("/path/to/Newtrinos.jl")   # replace with your actual clone path
using Newtrinos
```
