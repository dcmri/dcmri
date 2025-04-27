import numpy as np
import pydmr
import dcmri as dc
#
# Read the dataset:
#
datafile = dc.fetch('minipig_renal_fibrosis')
data = pydmr.read(datafile, 'nest')
rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']
time = pars['TS'] * np.arange(len(rois['LeftKidney']))
#
# Generate an AIF at high temporal resolution (250 msec):
#
dt = 0.25
t = np.arange(0, np.amax(time) + dt, dt)
ca = dc.aif_tristan(
   t,
   agent="gadoterate",
   dose=pars['dose'],
   rate=pars['rate'],
   weight=pars['weight'],
   CO=60,
   BAT=time[np.argmax(rois['Aorta'])] - 20,
)
#
# Initialize the tissue:
#
kidney = dc.Kidney(
   ca=ca,
   dt=dt,
   kinetics='HF',
   field_strength=pars['B0'],
   agent="gadoterate",
   t0=pars['TS'] * pars['n0'],
   TS=pars['TS'],
   TR=pars['TR'],
   FA=pars['FA'],
   R10a=1/dc.T1(pars['B0'], 'blood'),
   R10=1/dc.T1(pars['B0'], 'kidney'),
)
#
# Train the kidney on the data:
#
kidney.set_free(Ta=[0,30])
kidney.train(time, rois['LeftKidney'])
#
# Plot the reconstructed signals and concentrations:
#
kidney.plot(time, rois['LeftKidney'])
#
# Print the model parameters:
#
kidney.print_params(round_to=4)
# Expected:
## --------------------------------
## Free parameters with their stdev
## --------------------------------
## Arterial mean transit time (Ta): 13.8658 (0.1643) sec
## Plasma volume (vp): 0.0856 (0.003) mL/cm3
## Tubular flow (Ft): 0.0024 (0.0001) mL/sec/cm3
## Tubular mean transit time (Tt): 116.296 (7.6526) sec
