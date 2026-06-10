# Neutrino Oscillation Physics

For analysing neutrino oscillations it is certainly helpful to know about the physics of what is happening at neutrino oscillations. So this section is dedicated to develop basic knowledge of neutrino oscillation theory. It's important to note that this is just a brief overview and has no claim to completeness. It's still recommended to look in your favourite neutrino physics book for more detailed explanations. After mastering the basics, in the last section we briefly introduce the different BSM neutrino physics models that are implemented into the package.

## Flavor-Mass Mixing
Neutrino oscillations arise because the **flavor eigenstates** $|\nu_\alpha\rangle$ ($\alpha = e, \mu, \tau$) produced and detected via weak interactions are **not identical** to the **mass eigenstates** $|\nu_i\rangle$ ($i = 1, 2, 3$) that propagate through space. The two bases are related by a unitary mixing matrix. 

A neutrino produced in flavor state $\alpha$ is a coherent superposition of mass eigenstates:

$$
\boxed{|\nu_\alpha\rangle = \sum_{i=1}^{3} U_{\alpha i}\,|\nu_i\rangle},
$$

where $U$ is the **Pontecorvo-Maki-Nakagawa-Sakata (PMNS)** matrix. Conversely:

$$
|\nu_i\rangle = \sum_{\alpha = e,\mu,\tau} U_{\alpha i}^*\,|\nu_\alpha\rangle,
$$

For antineutrinos, $U \to U^*$:
$$
|\bar{\nu}_\alpha\rangle = \sum_i U_{\alpha i}^*\,|\bar{\nu}_i\rangle.
$$

## PMNS Matrix

The PMNS matrix relates flavor and mass eigenstates via:

$$\begin{pmatrix} \nu_e \\ \nu_\mu \\ \nu_\tau \end{pmatrix} = U_{\text{PMNS}} \begin{pmatrix} \nu_1 \\ \nu_2 \\ \nu_3 \end{pmatrix}$$, 

it is parametrized as 
$$
U_{\text{PMNS}} = \underbrace{\begin{pmatrix} 1 & 0 & 0 \\ 0 & c_{23} & s_{23} \\ 0 & -s_{23} & c_{23} \end{pmatrix}}_{R_{23}} \underbrace{\begin{pmatrix} c_{13} & 0 & s_{13}e^{-i\delta} \\ 0 & 1 & 0 \\ -s_{13}e^{i\delta} & 0 & c_{13} \end{pmatrix}}_{R_{13}(\delta)} \underbrace{\begin{pmatrix} c_{12} & s_{12} & 0 \\ -s_{12} & c_{12} & 0 \\ 0 & 0 & 1 \end{pmatrix}}_{R_{12}} \cdot P_M, 
$$ where $c_{ij} = \cos(\theta_{ij}), s_{ij} = \sin(\theta_{ij}),$ and $P_M = \text{diag}(1, e^{i\alpha/2}, e^{i\beta/2})$ contains the **Majorana phases**.

So the PMNS matrix is parametrized by three mixing angles $\theta_{ij}$, the CP violating phase $\delta$, and the two Majorana phases $\alpha, \beta$.

## Two-Flavour Oscillations
For simplicity we will start with the two flavour case and then go on with three flavours.

Consider only two flavors ($\nu_\alpha, \nu_\beta$) and two mass eigenstates ($\nu_1, \nu_2$) related by a single mixing angle $\theta$:

$$
\begin{pmatrix} \nu_\alpha \\ \nu_\beta \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \nu_1 \\ \nu_2 \end{pmatrix}.
$$

At $t = 0$, a $\nu_\alpha$ is produced:

$$
|\nu_\alpha(0)\rangle = \cos\theta\,|\nu_1\rangle + \sin\theta\,|\nu_2\rangle.
$$

As the neutrino propagates, each mass eigenstate propagates differently as a plane wave:

$$
|\nu_i(t)\rangle = e^{-iE_i t}\,|\nu_i\rangle.
$$

Since $m_\nu \ll E_\nu$ its common to use the **ultra-relativistic limit**:

$$
E_i = \sqrt{p^2 + m_i^2} \approx p + \frac{m_i^2}{2p} \approx E + \frac{m_i^2}{2E},
$$

where we identify $p \approx E$ for ultra-relativistic neutrinos (We work in natural units $\hbar = c = 1$).

The flavour transition amplitude is then given by 

$$
\langle\nu_\beta(t)|\nu_\alpha\rangle = -\sin\theta\cos\theta\,e^{-iE_1 t} + \sin\theta\cos\theta\,e^{-iE_2 t} = \sin\theta\cos\theta\left(e^{-iE_2 t} - e^{-iE_1 t}\right).
$$ 

This yields the **transition probability**

$$
P(\nu_\alpha \to \nu_\beta) = |\langle\nu_\beta(t)|\nu_\alpha\rangle|^2,
$$

$$
\boxed{P(\nu_\alpha \to \nu_\beta) = \sin^2(2\theta)\,\sin^2\!\left(\frac{\Delta m^2 L}{4E}\right)},
$$

where $\Delta m^2 = m_2^2 - m_1^2$ and $L$ is the source-detector distance (using $v_\nu \approx c \rightarrow t \approx L$).

The survival probability is: 
$$
P(\nu_\alpha \to \nu_\alpha) = 1 - \sin^2(2\theta)\,\sin^2\!\left(\frac{\Delta m^2 L}{4E}\right).
$$

The oscillation length is the distance over which the oscillation phase advances by $\pi$:

$$
L_{\text{osc}} = \frac{4\pi E}{\Delta m^2}
$$

In **practical (SI-like) units**:

$$
\boxed{L_{\text{osc}} = 2.48\,\frac{E\,[\text{GeV}]}{\Delta m^2\,[\text{eV}^2]}\ \text{km}}
$$

Typical Oscillation Lengths are: 
    - **Solar neutrinos** ($\Delta m^2_{21} \sim 7.5 \times 10^{-5}$ eV², $E \sim 1$ MeV): $L_{\text{osc}} \sim 33$ km
    - **Atmospheric neutrinos** ($\Delta m^2_{32} \sim 2.5 \times 10^{-3}$ eV², $E \sim 1$ GeV): $L_{\text{osc}} \sim 1000$ km
    - **Reactor neutrinos** ($E \sim 3$ MeV, $\Delta m^2_{32}$): $L_{\text{osc}} \sim 3$ km → Daya Bay baseline

Restoring $\hbar$ and $c$ we get the practical transition probability with SI Units:

$$
P(\nu_\alpha \to \nu_\beta) = \sin^2(2\theta)\,\sin^2\!\left(1.27\,\frac{\Delta m^2\,[\text{eV}^2]\cdot L\,[\text{km}]}{E\,[\text{GeV}]}\right)
$$

The factor $1.27$ comes from $\frac{1}{4}\frac{(\hbar c)}{(\text{GeV}\cdot\text{km}/\text{eV}^2)}$.

## Three-Flavour Oscillations

A neutrino produced as flavor $\alpha$ has a probability of being detected as flavor $\beta$ after propagating a distance $L$:

$$
P(\nu_\alpha \to \nu_\beta) = \left|\sum_i U_{\alpha i}^* U_{\beta i} e^{-im_i^2 L/(2E)}\right|^2
$$

$$
 = \delta_{\alpha\beta} - 4\sum_{i>j}\text{Re}(U_{\alpha i}^*U_{\beta i}U_{\alpha j}U_{\beta j}^*)\sin^2\!\left(\frac{\Delta m^2_{ij}L}{4E}\right)
+ 2\sum_{i>j}\text{Im}(U_{\alpha i}^*U_{\beta i}U_{\alpha j}U_{\beta j}^*)\sin\!\left(\frac{\Delta m^2_{ij}L}{2E}\right)
$$

In the three-flavour framework it is possible to get CP violating processes due to the CP phase $\delta$. Generally CP violation can only be observed in comparison of particle $\leftrightarrow$ anti-particle. For $\nu \to \bar{\nu}$, only the last term changes sign (since $U \to U^*$), giving:

$$
P(\nu_\alpha \to \nu_\beta) \neq P(\bar{\nu}_\alpha \to \bar{\nu}_\beta) \quad \text{if } \delta_{CP} \neq 0, \pi
$$

The CP asymmetry is:

$$
A_{CP}^{\alpha\beta} \propto P(\nu_\alpha \to \nu_\beta) - P(\bar{\nu}_\alpha \to \bar{\nu}_\beta) \propto J_{CP}\sin\!\left(\frac{\Delta m^2_{21}L}{4E}\right)\sin\!\left(\frac{\Delta m^2_{31}L}{4E}\right)\sin\!\left(\frac{\Delta m^2_{32}L}{4E}\right)
$$
where $J_{CP} = \frac{1}{8}\sin 2\theta_{12}\sin 2\theta_{13}\sin 2\theta_{23}\cos\theta_{13}\sin\delta_{CP}$ is the leptonic Jarlskog invariant. 

With this, it can also be seen that for $\alpha = \beta$ we get that $U_{\alpha i}^*U_{\beta i}U_{\alpha j}U_{\beta j}^* = |U_{\alpha i}|^2 |U_{\alpha j}|^2$ is real. Thus, we get $P(\nu_\alpha \to \nu_\beta) = P(\bar{\nu}_\alpha \to \bar{\nu}_\beta)$ and $A_{CP}^{\alpha\alpha}=0$. This means that CP violation in neutrino oscillations can only be measured with appearance experiments ($\alpha \neq \beta$).

## Neutrino Mass ordering
Since neutrino oscillations are only sensitive to the mass difference $\Delta m_{ij}^2$ we cannot determine the hierarchy of the mass eigenstates purely from oscillation theory. From the MSW effect we know that $\Delta m_{21}^2 = m_2^2 -m_1^2 > 0$. But we still can't determine where to put $m_3$. So analyses distinguish two possible orderings: 
- Normal ordering (NO): $m_1<m_2<m_3$, i.e. $\Delta m_{31}^2 > 0$
- Inverted ordering (IO): $m_3<m_1<m_2$, i.e. $\Delta m_{31}^2 < 0$

## Wave packet treatment
The plane-wave derivation above has conceptual issues:
1. Plane waves are infinitely delocalized — when does the neutrino "arrive"?
2. Different mass eigenstates have slightly different velocities: $v_i = p/E_i \approx 1 - m_i^2/2E^2$.
3. Over long distances, wave packets of different mass eigenstates **separate**, destroying coherence.

**Solution**: Model each mass eigenstate as a Gaussian wave packet with spatial width $\sigma_x$:

$$
|\nu_i(x,t)\rangle \propto \int dp\, \exp\!\left(-\frac{(p-p_0)^2}{4\sigma_p^2}\right)\exp\!\left(i(px - E_i(p)t)\right),
$$

where $\sigma_p \sim 1/(2\sigma_x)$ by the uncertainty principle. 

Now each packet has a group velocity $v_i \approx 1-\frac{m_i^2}{2E^2}$ yielding a velocity difference 

$$
\Delta v_{ij} = \frac{\Delta m^2_{ij}}{2E^2}.
$$

The wave packets of $\nu_i$ and $\nu_j$ overlap as long as their separation $\Delta v_{ij} \cdot t$ is smaller than the packet width $\sigma_x$. The **coherence length** is:

$$
\boxed{L_{\text{coh}} = \frac{4\sqrt{2}\,E^2\,\sigma_x}{\Delta m^2}}
$$

When $L \gg L_{\text{coh}}$, the wave packets no longer overlap and the interference (oscillation) term averages to zero. The oscillation probability becomes:

$$
P(\nu_\alpha \to \nu_\beta) \xrightarrow{L \gg L_{\text{coh}}} \sum_i |U_{\alpha i}|^2 |U_{\beta i}|^2
$$

This is the **incoherent (averaged) limit**.

Including wave packet effects, the probability becomes:

$$
P(\nu_\alpha \to \nu_\beta) = \sum_i |U_{\alpha i}|^2|U_{\beta i}|^2 + 2\,\text{Re}\sum_{i>j} U_{\alpha i}^*U_{\beta i}U_{\alpha j}U_{\beta j}^*\, e^{-i\Delta m^2_{ij}L/2E}\, \exp\!\left(-\frac{L^2}{L_{\text{coh}}^2}\right)
$$

The Gaussian damping factor $\exp(-L^2/L_{\text{coh}}^2)$ suppresses the oscillatory terms at large $L$.

## Neutrino Standard Model Interactions

In the Standard Model, neutrinos interact only via the weak force: **charged-current (CC)** interactions mediated by the $W^\pm$ boson couple $\nu_\alpha$ to the corresponding charged lepton $\ell_\alpha$, while **neutral-current (NC)** interactions mediated by the $Z^0$ boson couple all active flavors universally. For oscillation physics, only coherent elastic forward scattering is relevant — it imprints a phase on the propagating neutrino without changing its momentum, effectively adding a flavor-dependent potential to the Hamiltonian. This is the origin of matter effects.

## Matter Effects

In the presence of a medium, the neutrino effective Hamiltonian receives an additional contribution from coherent forward scattering on the ambient electrons and nucleons. The result is that the oscillation parameters governing propagation through matter differ from their vacuum values, and resonant flavor conversion becomes possible.

### Standard MSW Effect

The dominant contribution comes from CC scattering of $\nu_e$ on electrons (only $\nu_e$ participates in CC scattering at low energies). Adding this to the vacuum Hamiltonian gives:

$$
H_{\text{eff}} = H_{\text{vac}} + V,\qquad V = \sqrt{2}\,G_F\,n_e\,\text{diag}(1,\,0,\,0),
$$

where $n_e$ is the local electron number density and $G_F$ is the Fermi constant.

For two flavors, an **MSW resonance** occurs when the matter potential exactly cancels the vacuum splitting, maximizing the effective mixing angle in matter regardless of the vacuum value of $\theta$:

$$
\Delta m^2\cos 2\theta = 2\sqrt{2}\,G_F\,n_e\,E.
$$

At the resonance the transition probability reaches unity even for small vacuum mixing. This mechanism explains the large flavor conversion of solar $\nu_e$ on their way out of the Sun (Mikheyev–Smirnov–Wolfenstein effect).

### Non-Standard Interactions

**Non-Standard Interactions (NSI)** extend the matter potential with BSM couplings $\varepsilon_{\alpha\beta}$ between neutrinos and matter fermions $f$:

$$
V_{\text{NSI}} = \sqrt{2}\,G_F\,n_e\begin{pmatrix} \varepsilon_{ee} & \varepsilon_{e\mu} & \varepsilon_{e\tau} \\ \varepsilon_{e\mu}^* & \varepsilon_{\mu\mu} & \varepsilon_{\mu\tau} \\ \varepsilon_{e\tau}^* & \varepsilon_{\mu\tau}^* & \varepsilon_{\tau\tau} \end{pmatrix}.
$$

The Hermitian $\varepsilon$ matrix adds both diagonal (flavor-universal shifting) and off-diagonal (flavor-changing) terms to the effective Hamiltonian. NSI can mimic or mask standard oscillation signals, shift the apparent values of mixing angles and mass splittings, and generate new matter resonances.

## Neutrino Sources

Different production mechanisms yield neutrinos across many orders of magnitude in energy and baseline, each probing a distinct combination of oscillation parameters.

### Atmospheric Neutrinos

**Atmospheric neutrinos** are produced when cosmic rays (mostly protons and helium nuclei) collide with nuclei in the upper atmosphere, initiating hadronic showers containing charged pions and kaons. Their decays yield $\nu_\mu$, $\bar\nu_\mu$, and $\nu_e$ fluxes with an approximate 2:1 ratio at low energies.

- **Energy range**: $\sim$100 MeV to $\sim$100 TeV
- **Baseline**: 10 km (downward-going) to $\sim$13,000 km (upward-going, traversing the full Earth)
- **Primary sensitivity**: $\theta_{23}$, $\Delta m^2_{31}$; matter effects for upward-going tracks
- **Package experiments**: `deepcore`, `ic_upgrade`, `super_k`, `orca`

The large range of baselines and the Earth's density profile make atmospheric neutrinos uniquely sensitive to both vacuum oscillations and the MSW effect simultaneously.

### Solar Neutrinos

**Solar neutrinos** are produced in the thermonuclear fusion reactions of the pp chain and the CNO cycle in the Sun's core.

- **Energy range**: $\sim$100 keV to $\sim$15 MeV (pp to $^8$B flux)
- **Baseline**: $\sim 1.5 \times 10^8$ km (Sun–Earth distance)
- **Primary sensitivity**: $\theta_{12}$, $\Delta m^2_{21}$; adiabatic MSW resonance inside the Sun

The SNO and Super-Kamiokande experiments established flavor conversion of solar $\nu_e$ via the MSW effect. Solar neutrinos are not currently implemented in Newtrinos.jl.

### Reactor Neutrinos

**Reactor antineutrinos** ($\bar\nu_e$) arise from $\beta$-decay of neutron-rich fission products in nuclear power plants.

- **Energy range**: $\sim$1–10 MeV
- **Baselines**: $\sim$1 km (near detectors) to $\sim$200 km (far detectors)
- **Primary sensitivity**: $\theta_{13}$ (near–far comparison, e.g. Daya Bay); $\theta_{12}$, $\Delta m^2_{21}$ at long baselines (KamLAND)
- **Package experiments**: `dayabay`, `kamland`, `juno`, `tao`

Because $\bar\nu_e$ disappearance is a pure vacuum process at these energies, reactor experiments provide clean measurements of $\theta_{13}$ and $\Delta m^2_{21}$ without matter-effect complications.

### Accelerator Neutrinos

**Accelerator neutrino beams** are produced from decays of pions and kaons created by a proton beam on a fixed target, yielding a predominantly $\nu_\mu$ (or $\bar\nu_\mu$) flux.

- **Energy range**: $\sim$100 MeV to $\sim$10 GeV
- **Baselines**: 100 km to 1300 km (long-baseline experiments)
- **Primary sensitivity**: $\theta_{23}$, $\Delta m^2_{31}$; $\theta_{13}$ and $\delta_{CP}$ via $\nu_e$ appearance
- **Package experiments**: `minos`

The controlled beam profile and near/far detector combinations allow precise extraction of oscillation parameters and are the primary tool for $\delta_{CP}$ measurements.

### Astrophysical Neutrinos

**High-energy astrophysical neutrinos** are produced in hadronic processes at cosmic accelerators such as active galactic nuclei (AGN), gamma-ray bursts (GRB), and starburst galaxies, via $pp$ and $p\gamma$ interactions.

- **Energy range**: TeV–EeV
- **Baselines**: cosmological (Mpc–Gpc)
- **Primary sensitivity**: flavor ratios at Earth; ultra-high-energy probes of new physics (pseudo-Dirac mass splittings, Lorentz violation, neutrino decay)

At these baselines oscillations are fully averaged, so only the incoherent flavor ratio $\sum_i |U_{\alpha i}|^2 |U_{\beta i}|^2$ is observable. Astrophysical neutrinos are not currently implemented in Newtrinos.jl.

## Advanced BSM Models

Beyond the standard three-flavor oscillation framework, Newtrinos.jl implements several BSM extensions selectable via `OscillationConfig`. Each model introduces new physics parameters that appear in `params` and `priors` alongside the standard oscillation parameters.

**Sterile neutrinos** (`Newtrinos.osc.Sterile`) extend the three-flavor framework with a fourth (or more) mass eigenstate that has no weak interactions. The PMNS matrix is extended to a $4 \times 4$ unitary matrix, introducing three new mixing angles and additional mass splittings $\Delta m^2_{41}$. Active flavors mix into the sterile state, suppressing their oscillation probabilities and producing "disappearance" signatures at short baselines.

**Large Extra Dimensions** (`Newtrinos.osc.ADD`) build on the Arkani-Hamed–Dimopoulos–Dvali framework: if right-handed neutrinos propagate into $n$ compact extra dimensions of radius $R$, a tower of Kaluza–Klein (KK) mass states appears. The oscillation probability becomes a sum over KK modes weighted by mixing coefficients determined by $R$ and the fundamental mass scale, modifying the standard $L/E$ pattern at short baselines.

**Dark Dimensions** (`Newtrinos.osc.Darkdim_*`) are variants of the LED scenario where the extra-dimensional bulk is populated by a dark sector. Several sub-models are implemented (`Darkdim_radius`, `Darkdim_mass`, etc.), each parametrizing the dark-dimension radius and the bulk-to-brane coupling differently. Like ADD, the signature is a deviation from standard oscillations due to active–KK mixing.

**Quantum Decoherence** (`Newtrinos.osc.Decoherent`) models the loss of quantum coherence between mass eigenstates during propagation — for example due to wave-packet separation (see Wave Packet Treatment above) or interactions with a stochastic environment. The oscillation term is multiplied by a damping factor $e^{-\Gamma_{ij} L}$ where $\Gamma_{ij}$ is a decoherence rate. At $\Gamma \to 0$ the standard result is recovered; at large $L$ the probability approaches the incoherent average.

**Damping** (`Newtrinos.osc.Damping`) is a phenomenological model that applies energy-dependent exponential suppression to all off-diagonal oscillation terms, $\exp(-(E/\Lambda)^n)$, without specifying the underlying mechanism. It serves as a model-independent probe of new physics that suppresses coherence, including neutrino absorption, neutrino decay, and exotic interactions.

For **Non-Standard Interactions** in matter, see [Non-Standard Interactions](#non-standard-interactions) above.

