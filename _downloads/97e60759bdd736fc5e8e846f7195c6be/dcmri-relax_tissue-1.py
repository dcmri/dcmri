import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# Generate a population-average input function:
#
t = np.arange(0, 300, 1.5)
ca = dc.aif_parker(t, BAT=20)
#
# Define constants and model parameters:
#
R10, r1 = 1/dc.T1(), dc.relaxivity()
pf = {'H':0.5, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005}
pn = {'H':0.5, 'vb':0.1, 'vi':0.3, 'Fb':0.01, 'PS':0.005}
#
# Calculate tissue relaxation rates without water exchange,
# and also in the fast exchange limit for comparison:
#
R1f, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='FF', **pf)
R1n, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='NN', **pn)
#
# Plot the relaxation rates in the three compartments, and compare
# against the fast exchange result:
#
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
#
# Plot restricted water exchange in the left panel:
#
ax0.set_title('Restricted water exchange')
ax0.plot(t/60, R1n[0,:], linestyle='-',
         linewidth=2.0, color='darkred', label='Blood')
ax0.plot(t/60, R1n[1,:], linestyle='-',
         linewidth=2.0, color='darkblue', label='Interstitium')
ax0.plot(t/60, R1n[2,:], linestyle='-',
         linewidth=2.0, color='grey', label='Cells')
ax0.set_xlabel('Time (min)')
ax0.set_ylabel('Compartment relaxation rate (1/sec)')
ax0.legend()
#
# Plot fast water exchange in the right panel:
#
ax1.set_title('Fast water exchange')
ax1.plot(t/60, R1f, linestyle='-',
         linewidth=2.0, color='black', label='Tissue')
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Tissue relaxation rate (1/sec)')
ax1.legend()
plt.show()
