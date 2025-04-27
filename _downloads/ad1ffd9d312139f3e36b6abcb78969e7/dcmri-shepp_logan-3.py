import matplotlib.pyplot as plt
import dcmri as dc
#
# Generate the MR Shepp-Logan phantom masks:
#
im = dc.shepp_logan(n=128)
#
# Plot all masks:
#
fig, ax = plt.subplots(figsize=(8, 8), ncols=4, nrows=4)
ax[0,0].imshow(im['background'], cmap='gray')
ax[0,1].imshow(im['scalp'], cmap='gray')
ax[0,2].imshow(im['bone'], cmap='gray')
ax[0,3].imshow(im['CSF skull'], cmap='gray')
ax[1,0].imshow(im['CSF left'], cmap='gray')
ax[1,1].imshow(im['CSF right'], cmap='gray')
ax[1,2].imshow(im['gray matter'], cmap='gray')
ax[1,3].imshow(im['tumor 1'], cmap='gray')
ax[2,0].imshow(im['tumor 2'], cmap='gray')
ax[2,1].imshow(im['tumor 3'], cmap='gray')
ax[2,2].imshow(im['tumor 4'], cmap='gray')
ax[2,3].imshow(im['tumor 5'], cmap='gray')
ax[3,0].imshow(im['tumor 6'], cmap='gray')
ax[3,1].imshow(im['sagittal sinus'], cmap='gray')
ax[3,2].imshow(im['anterior artery'], cmap='gray')
for i in range(4):
    for j in range(4):
        ax[i,j].set_yticklabels([])
        ax[i,j].set_xticklabels([])
plt.show()
