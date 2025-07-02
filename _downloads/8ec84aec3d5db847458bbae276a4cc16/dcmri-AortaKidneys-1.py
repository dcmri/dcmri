import numpy as np
import pydmr
import dcmri as dc
#
# Read the dataset:
#
datafile = dc.fetch('minipig_renal_fibrosis')
data = pydmr.read(datafile, 'nest')
rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']
#
# Create an array of time points:
#
time = pars['TS'] * np.arange(len(rois['Aorta']))
#
# Initialize the tissue:
#
aorta_kidneys = dc.AortaKidneys(
    sequence='SSI',
    heartlung='chain',
    organs='comp',
    agent="gadoterate",
    dt=0.25,
    field_strength=pars['B0'],
    weight=pars['weight'],
    dose=pars['dose'],
    rate=pars['rate'],
    R10a=1/dc.T1(pars['B0'], 'blood'),
    R10_lk=1/dc.T1(pars['B0'], 'kidney'),
    R10_rk=1/dc.T1(pars['B0'], 'kidney'),
    vol_lk=85,
    vol_rk=85,
    TR=pars['TR'],
    FA=pars['FA'],
    TS=pars['TS'],
    CO=60,
    t0=15,
)
#
# Define time and signal data
#
t = (time, time, time)
signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])
#
# Train the system to the data:
#
aorta_kidneys.train(t, signal)
#
# Plot the reconstructed signals and concentrations:
#
aorta_kidneys.plot(t, signal)
#
# Print the model parameters:
#
aorta_kidneys.print_params(round_to=4)
# Expected:
## --------------------------------
## Free parameters with their stdev
## --------------------------------
## Bolus arrival time (BAT): 16.7422 (0.2853) sec
## Inflow time (TF): 0.2801 (0.0133) sec
## Cardiac output (CO): 72.762 (12.4426) mL/sec
## Heart-lung mean transit time (Thl): 16.2249 (0.3069) sec
## Organs blood mean transit time (To): 14.3793 (1.2492) sec
## Body extraction fraction (Eb): 0.0751 (0.0071)
## Heart-lung dispersion (Dhl): 0.0795 (0.0041)
## Renal plasma flow (RPF): 3.3489 (0.7204) mL/sec
## Differential renal function (DRF): 0.9085 (0.0212)
## Differential renal plasma flow (DRPF): 0.812 (0.0169)
## Left kidney arterial mean transit time (Ta_lk): 0.6509 (0.2228) sec
## Left kidney plasma volume (vp_lk): 0.099 (0.0186) mL/cm3
## Left kidney tubular mean transit time (Tt_lk): 46.9705 (3.3684) sec
## Right kidney arterial mean transit time (Ta_rk): 1.4206 (0.2023) sec
## Right kidney plasma volume (vp_rk): 0.1294 (0.0175) mL/cm3
## Right kidney tubular mean transit time (Tt_rk): 4497.8301 (39890.3818) sec
## Aorta signal scaling factor (S0a): 4912.776 (254.2363) a.u.
## ----------------------------
## Fixed and derived parameters
## ----------------------------
## Filtration fraction (FF): 0.0812
## Glomerular Filtration Rate (GFR): 0.2719 mL/sec
## Left kidney plasma flow (RPF_lk): 2.7194 mL/sec
## Right kidney plasma flow (RPF_rk): 0.6295 mL/sec
## Left kidney glomerular filtration rate (GFR_lk): 0.247 mL/sec
## Right kidney glomerular filtration rate (GFR_rk): 0.0249 mL/sec
## Left kidney plasma flow (Fp_lk): 0.032 mL/sec/cm3
## Left kidney plasma mean transit time (Tp_lk): 2.838 sec
## Left kidney vascular mean transit time (Tv_lk): 3.0958 sec
## Left kidney tubular flow (Ft_lk): 0.0029 mL/sec/cm3
## Left kidney filtration fraction (FF_lk): 0.0908
## Left kidney extraction fraction (E_lk): 0.0833
## Right kidney plasma flow (Fp_rk): 0.0074 mL/sec/cm3
## Right kidney plasma mean transit time (Tp_rk): 16.8121 sec
## Right kidney vascular mean transit time (Tv_rk): 17.4762 sec
## Right kidney tubular flow (Ft_rk): 0.0003 mL/sec/cm3
## Right kidney filtration fraction (FF_rk): 0.0395
## Right kidney extraction fraction (E_rk): 0.038
