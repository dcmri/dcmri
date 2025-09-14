import numpy as np

import dcmri as dc



def test_convmat():

    # Uniform time grid with increasing precision: compare against analytical convolution.

    Tf = 20
    Th = 30
    tmax = 30

    prec = [1e-1, 1e-3, 1e-5]
    for i, dt in enumerate([10,1,0.1]):
        t = np.arange(0,tmax,dt)
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        mat = dc.convmat(f)
        #g = dt*np.matmul(mat.T, h)
        g = dt * (mat @ h)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]
        mat = dc.convmat(f, order=1)
        g1 = dt * (mat @ h)
        assert np.linalg.norm(g-g0) < np.linalg.norm(g1-g0)


def test_invconvmat():

    Tf = 20
    tmax = 30
    dt = 0.1
    order = 1
    t = np.arange(0,tmax,dt)
    f = np.exp(-t/Tf)/Tf
    mat = dc.convmat(f, order=order)
    matinv = dc.invconvmat(f, order=order, tol=1e-12)
    id = mat @ matinv
    idexact = np.eye(len(t))
    assert np.linalg.norm(id-idexact)/np.linalg.norm(idexact) < 1e-9


def test_deconv():

    # Uniform time grid with increasing precision: compare against analytical convolution.

    Tf = 20
    Tg = 30
    tmax = 100
    dt = 0.1
    t = np.arange(0,tmax,dt)
    f = np.exp(-t/Tf)/Tf
    g = np.exp(-t/Tg)/Tg
    h = (dc.convmat(g) @ f) * dt
    frec = dc.deconv(h, g, dt)
    assert np.linalg.norm(f-frec)/np.linalg.norm(f) < 0.1

    F = np.tile(f[:, None], (1, 3))
    H = np.tile(h[:, None], (1, 3))
    Frec = dc.deconv(H, g, dt)
    assert np.linalg.norm(F-Frec)/np.linalg.norm(F) < 0.1


if __name__ == "__main__":

    print('Testing pk_inv..')

    test_convmat()
    test_invconvmat()
    test_deconv()

    print('All pk_inv tests passed!!')
