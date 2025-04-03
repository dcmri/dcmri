import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# Generate a population-average input function:
#
t = np.arange(0, 300, 1.5)
ca = dc.aif_parker(t, BAT=20)
#
# Define some parameters and generate plasma and tubular tissue concentrations with a 2-compartment filtration model:
#
Fp, Tp, Ft, Tt = 0.05, 10, 0.01, 120
C = dc.conc_kidney(ca, Fp, Tp, Ft, Tt, t=t, sum=False, kinetics='2CF')
#
# Plot all concentrations:
#
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Kidney concentrations')
ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Tubuli')
ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Whole kidney')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Tissue concentration (mM)')
ax.legend()
plt.show()
#
# Use generate plasma and tubular tissue concentrations using the free nephron model for comparison. We assume 4 transit time bins with the following boundaries (in units of seconds):
#
TT = [0, 15, 30, 60, 120]
#
# with longest transit times most likely (note the frequences to not have to add up to 1):
#
h = [1, 2, 3, 4]
C = dc.conc_kidney(ca, Fp, Tp, Ft, h, t=t, sum=False, kinetics='FN', TT=TT)
#
# Plot all concentrations:
#
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Kidney concentrations')
ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Tubuli')
ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Whole kidney')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Tissue concentration (mM)')
ax.legend()
plt.show()
