import matplotlib.pyplot as plt
import dcmri as dc
#
# Simulate a synthetic blood flow image:
#
im = dc.shepp_logan('Fb')
#
# Plot the result in units of mL/min/100mL:
#
fig, ax = plt.subplots(figsize=(5, 5), ncols=1)
pos = ax.imshow(6000*im, cmap='gray', vmin=0.0, vmax=80)
fig.colorbar(pos, ax=ax, label='blood flow (mL/min/100mL)')
plt.show()
