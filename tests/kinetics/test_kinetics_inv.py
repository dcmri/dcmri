import numpy as np

import dcmri as dc
from dcmri.kinetics.functions_inv import _params_2cfm


def test_linfit_2cfm():
    nx, nt = 3, 5
    imgs = np.arange(nx * nt).reshape((nx, nt))
    aif = np.ones(nt)
    time = np.arange(nt)
    fir, pars = dc.linfit_2cfm(imgs, aif, time)

    _params_2cfm([1, 1, 1, 1])
    _params_2cfm([1, 3, 1, 1])
    _params_2cfm([0, 1, 1, 1])
    _params_2cfm([0, 0, 1, 1])
    _params_2cfm([1, 0, 1, 1])


if __name__=='__main__':
    test_linfit_2cfm()