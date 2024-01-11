"""
==============
The Parker AIF
==============

Use `~dcmri.aif_parker` to generate a Parker AIF with different settings. 
"""

# %%
# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import dcmri

# %%
# Generate synthetic AIF and plot the result.

# Define time points in units of seconds.
# In this case we use a time resolution of 0.5 sec 
# and a total duration of 6 minutes:
t = np.arange(0, 6*60, 0.5)

# Create an AIF with arrival time of 0:
ca = dcmri.aif_parker(t)

# Plot the AIF over the full range
plt.plot(t, ca*1000, 'r-')
plt.plot(t, 0*t, 'k-')
plt.xlabel('Time (sec)')
plt.ylabel('Plasma concentration (mM)')
plt.show()

# %%
# The bolus arrival time (BAT) defaults to 0s. What happens if we change it? Let's try, by changing it in steps of 30s:

# Create AIfs with different BAT's:
ca1 = dcmri.aif_parker(t, BAT=0)
ca2 = dcmri.aif_parker(t, BAT=30)
ca3 = dcmri.aif_parker(t, BAT=60)
ca4 = dcmri.aif_parker(t, BAT=90)

# Show them all on the same plot:
plt.plot(t, ca1*1000, 'b-', label='BAT = 0s')
plt.plot(t, ca2*1000, 'r-', label='BAT = 30s')
plt.plot(t, ca3*1000, 'g-', label='BAT = 60s')
plt.plot(t, ca4*1000, 'm-', label='BAT = 90s')
plt.xlabel('Time (sec)')
plt.ylabel('Plasma concentration (mM)')
plt.legend()
plt.show()


# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
