import numpy as np


from dcmri import FluxTissueX



def test_flux_tissue():

    # flux_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = FluxTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0


if __name__ == '__main__':

    test_flux_tissue()

    print('All kinetics tests passed!!')