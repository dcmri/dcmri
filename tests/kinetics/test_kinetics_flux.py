import numpy as np


import dcmri as dc

def test_flux_block():
    dt = 1.5
    t = dt * np.arange(20)
    J = np.ones(20)
    J = dc.FluxBlock('comp')(J, t=t, T=20)
    assert round(J[-1]) == 1
    dc.FluxBlock()._param_names()


def test_flux_tissue():

    # flux_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.FluxTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0


if __name__ == '__main__':

    test_flux_block()
    test_flux_tissue()

    print('All kinetics tests passed!!')