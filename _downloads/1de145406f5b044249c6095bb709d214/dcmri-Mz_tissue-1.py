import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# Define constants and model parameters:
#
R10, r1 = 1, 5000
seq = {'model': 'SS', 'FA':15, 'TR': 0.001, 'B1corr':1}
pars = {
    'sequence':seq, 'kinetics':'2CX', 'water_exchange':'NN',
    'H':0.045, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005}
inflow = {'R10a': 0.7, 'B1corr_a':1}
#
# Generate arterial blood concentrations:
#
t = np.arange(0, 300, 1.5)
ca = dc.aif_parker(t, BAT=20)/(1-0.45)
#
# Calculate the signal with and without inflow:
#
Mf = dc.Mz_tissue(ca, R10, r1, t=t, inflow=inflow, **pars)
Mn = dc.Mz_tissue(ca, R10, r1, t=t, **pars)
#
# Compare them in a plot:
#
plt.figure()
plt.plot(t/60, np.sum(Mn, axis=0), label='No inflow', linewidth=3)
plt.plot(t/60, np.sum(Mf, axis=0), label='Inflow')
plt.xlabel('Time (min)')
plt.ylabel('Magnetization (A/cm)')
plt.legend()
plt.show()
