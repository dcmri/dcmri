import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# Generate a population-average input function:
#
t = np.arange(0, 30*60, 1.5)
ca = dc.aif_parker(t, BAT=20)
#
# Generate plasma and tubular tissue
# concentrations with a non-stationary model:
#
C = dc.conc_liver(ca, t, sum=False,
    H = 0.45, ve = 0.2, Te = 30, De = 0.5,
    khe = [0.003, 0.01], Th = [180, 600],
)
#
# Plot all concentrations:
#
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Kidney concentrations')
ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0,
        color='darkred', label='Extracellular')
ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0,
        color='darkblue', label='Hepatocytes')
ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0,
        color='grey', label='Whole liver')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Tissue concentration (mM)')
ax.legend()
plt.show()
