import matplotlib.pyplot as plt
import dcmri as dc
#
# Generate the MR Shepp-Logan phantom in low resolution:
#
im = dc.shepp_logan('PD', 'T1', 'T2', n=64)
#
# Plot the result:
#
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 5), ncols=3)
ax1.imshow(im['PD'], cmap='gray')
ax2.imshow(im['T1'], cmap='gray')
ax3.imshow(im['T2'], cmap='gray')
plt.show()
