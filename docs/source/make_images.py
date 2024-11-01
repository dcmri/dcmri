import os
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import dcmri as dc


def fake_brain():

    time, signal, aif, gt = dc.fake_brain(n=64, tacq=90, CNR=100)

    fig, ax = plt.subplots()
    ims = []
    for i in range(time.size):
        im = ax.imshow(signal[:,:,i], cmap='magma', animated=True, 
                       vmin=0, vmax=30)
        ims.append([im])
    ani = ArtistAnimation(
        fig, ims, interval=50, blit=True,
        repeat_delay=0)
    ani.save("docs/source/_static/animations/fake_brain.gif")
    plt.close(fig)


def getting_started():

    time, aif, roi, _ = dc.fake_tissue(CNR=50)
    tissue = dc.Tissue(aif=aif, t=time)
    tissue.train(time, roi)
    tissue.plot(time, roi, fname='docs/source/user_guide/tissue.png')
    tissue.print_params(round_to=3)

    tissue = dc.Tissue(aif=aif, t=time, water_exchange='FR')
    tissue.train(time, roi).print_params(round_to=3)

    tissue = dc.Tissue(aif=aif, t=time, water_exchange='FR', vb=0.5)
    tissue.set_free(B1corr=[0,2])
    tissue.train(time, roi).print_params(round_to=3)


def getting_started_images():

    n = 128
    time, signal, aif, _ = dc.fake_brain(n)
    shape = (n, n)

    image = dc.TissueArray(shape, aif=aif, t=time, verbose=1)
    image.train(time, signal)
    image.plot(time, signal, 
               fname='docs/source/user_guide/pixel.png')
    
    image = dc.TissueArray(shape, aif=aif, t=time, verbose=1, kinetics='2CU')
    image.train(time, signal)
    image.plot(time, signal, fname='docs/source/user_guide/pixel_2cu.png')


if __name__ == '__main__':

    # fake_brain()
    getting_started()
    # getting_started_images()