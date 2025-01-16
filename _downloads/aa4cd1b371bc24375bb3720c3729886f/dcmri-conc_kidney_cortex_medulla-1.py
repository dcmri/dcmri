import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# Generate a population-average input function:
#
t = np.arange(0, 300, 1.5)
ca = dc.aif_parker(t, BAT=20)
#
# Use the function to generate total cortex and medulla tissue concentrations:
#
Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd = 0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30
Cc, Cm = dc.conc_kidney_cortex_medulla(ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, t=t, kinetics='7C')
#
# Plot all concentrations:
#
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.set_title('Kidney concentrations')
ax.plot(t/60, 1000*Cc, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
ax.plot(t/60, 1000*Cm, linestyle='-', linewidth=3.0, color='darkgreen', label='Medulla')
ax.plot(t/60, 1000*(Cc+Cm), linestyle='-', linewidth=3.0, color='darkgrey', label='Whole kidney')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Tissue concentration (mM)')
ax.legend()
plt.show()
