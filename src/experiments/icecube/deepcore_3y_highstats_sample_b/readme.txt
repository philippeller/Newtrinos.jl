Data release of the IceCube/DeepCore 3y high statistics oscillation analysis.
This sample is referred to as 'Analysis B' in [1], and is a very close variant of what was used in [2].
Please refer to [1] for a detailed explanation of the sample and analysis.

Note: the instructions contained here are to be applied to Sample B only. This is particularly important
for the detector uncertainties, which have slightly different implementations between the two samples.
For sample A there is a separate, different readme file.


Files:
------
There are seven files containing comma separated values (CSV):
- neutrino_mc.csv : event-by-event information for simulated neutrinos
- muons.csv : the per-bin information from the data driven muon background
- data.csv : the count of actual data events per analysis bin
- hyperplanes_*.csv : the corrections to be applied to the neutrino mc expected count per bin accounting for detector uncertainties


Conventions:
------------
- the cosine of zenith values (`coszen`) are used to encode directional information. The convention
  is as follows: -1 corresponds to straight "upgoing" (= Earth crossing, = from the North) trajectories,
  +1 to straight "downgoing" (= coming directly from sky above, = from the South) trajectories and 0 is horizontal


Binning:
--------
The analysis binning to be used is:
- reco_energy : bin edges = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413] in GeV
- reco_coszen : bin edges = [-1., -0.75, -0.5 , -0.25,  0., 0.25, 0.5, 0.75, 1.]
- pid : 0 = cascades, 1 = tracks

neutrino_mc.csv:
----------------
This file contains simulated neutrinos with one event per row with the following information:
- true_energy: simulated energy (GeV) of neutrino
- true_coszen: simulated cosine(zenith) of neutrino
- reco_energy: reconstructed energy (GeV) in analysis binning, values are chosen so that events will fall into the correct bins
- reco_coszen: reconstructed cosine(zenith) in analysis binning, values are chosen so that events will fall into the correct bins
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- weight: weight per event, this shall be multiplied by the atmospheric flux in (m^-2 s^-1) and the livetime in (s)
- pdg: PDG code of the neutrino interacting in the detector
- type: interaction type where 0 = any neutral current (NC), 1 = charged current quasi elastic (CC_QE), 2 = charged current resonant (CC_RES), 3 = charged current deep inelastic (CC_DIS)


muons.csv:
----------
This file contains the counts and the absolute uncertainties for the data driven muon background:
- reco_energy: reconstructed energy (GeV) in analysis binning
- reco_coszen: reconstructed cosine(zenith) in analysis binning
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- count: nominal count
- abs_uncert: absolute uncertainty on count, accounting for statistical and shape uncertainty


data.csv:
--------
This file contains the counts of the observed data:
- reco_energy: reconstructed energy (GeV) in analysis binning
- reco_coszen: reconstructed cosine(zenith) in analysis binning
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- count: count of data events


hyperplanes_*.csv:
------------------
these files contain the information needed to apply the corrections from detector uncertainties.
Four different files are following the same structure. The different files are to be used on the neutrino MC, 
with `hyperplanes_all_nc.csv` applied to any neutral current event (type == 0), `hyperplanes_nue_cc.csv` to the neutrinos that 
have pdg == +12 or -12 (nue and nuebar) and type > 0 (CC), and so forth.

The files contain the usual reco_energy, reco_coszen, pid (same as in muons.csv and data.csv) and 6 additional values:
- offset : constant factor
- ice_absorption : slope accounting for differences in ice optical absorption, nominal parameter values is 1, the uncertainty +/- 0.1 and the valid range +/- 0.15
- ice_scattering : slope accounting for differences in ice optical scattering, nominal parameter values is 1, the uncertainty +/- 0.1 and the valid range +/- 0.15
- opt_eff_overall : slope accounting for overall optical efficiency of DOMs, nominal parameter value is 1, the uncertainty +/- 0.1 and the valid range is +/- 0.2
- opt_eff_lateral : slope accounting for lateral optical efficiency of DOMs, nominal parameter value is 0, the uncertainty +/- 1 and the valid range is +/- 2
- opt_eff_headon : slope accounting for head-on optical efficiency of DOMs, nominal parameter value is 0, the uncertainty unknown (flat) and the valid range is [-5.0, +2.0]

To arrive at the correction factor per bin, the offset plus each slope multiplied by a given parameter value will result in a multiplicative factor to be applied to the event count in each analysis bin, calculated as:

`f = df['offset']
    + df['ice_absorption'] * 100 * (value - 1.)
    + df['ice_scattering'] * 100 * (value - 1.)
    + df['opt_eff_overall'] * value
    + df['opt_eff_lateral'] * ((value * 10) + 25.)
    + df['opt_eff_headon'] * value` 
    
where `df` holds the values from the CSV file.

The factor `f` is then applied multiplicatively to the final event rates for the given bin.

So for example with all parameters at nominal values this factor `f` would be: 
`f = df['offset']
    + df['ice_absorption'] * 0. 
    + df['ice_scattering'] * 0. 
    + df['opt_eff_overall'] * 1. 
    + df['opt_eff_lateral'] * 25. 
    + df['opt_eff_headon'] * 0.` 
    
Or accordingly with all parameters at the (CC + NC) bestfit values (see Table 2 in reference [1]) this factor `f` would be: 
`f = df['offset']
    + df['ice_absorption'] * 2.1 
    + df['ice_scattering'] * -2.6 
    + df['opt_eff_overall'] * 1.05 
    + df['opt_eff_lateral'] * 22.5 
    + df['opt_eff_headon'] * -1.15` 

References:
-----------
[1] M. G. Aartsen et al. Measurement of Atmospheric Tau Neutrino Appearance with IceCube DeepCore. Phys. Rev., D99(3):032007, 2019. arXiv:1901.05366, doi:10.1103/PhysRevD.99.032007
[2] M. G. Aartsen et al. Measurement of Atmospheric Neutrino Oscillations at 6-56 GeV with IceCube DeepCore. Phys. Rev. Lett., 120(7):071801, 2018. arXiv:1707.07081, doi:10.1103/PhysRevLett.120.071801
