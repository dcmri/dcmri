import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import dcmri as dc


def fake_brain():

    time, signal, aif, gt = dc.fake_brain(n=64, tacq=90, CNR=100)

    fig, ax = plt.subplots()
    ims = []
    for i in range(time.size):
        im = ax.imshow(signal[:,:,i], cmap='magma', animated=True, vmin=0, vmax=30)
        ims.append([im])
    ani = ArtistAnimation(
        fig, ims, interval=50, blit=True,
        repeat_delay=0)
    ani.save("docs/source/_static/animations/fake_brain.gif")
    plt.close(fig)


if __name__ == '__main__':

    fake_brain()