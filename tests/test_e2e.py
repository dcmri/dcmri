import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc

#
# Generate a population-average input function:
#
t = np.arange(0, 300, 1.5)
ca = dc.aif_parker(t, BAT=20)
#
# Use the function to generate cortex and medulla tissue concentrations:
#
C = dc.kidney_conc_pf(ca, 0.05, 10, 0.01, [1,1,1,1,1,1], t, sum=False)
#
# Plot all concentrations:
#
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Kidney concentrations')
ax.plot(t/60, 1000*C[0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkgreen', label='Medulla')
ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-.', linewidth=3.0, color='darkviolet', label='Whole kidney')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Tissue concentration (mM)')
ax.legend()
plt.show()