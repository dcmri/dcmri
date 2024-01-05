import numpy as np
from scipy.stats import norm
import dcmri.tools as tools
import dcmri.pk as pk

def test_res_trap():
    t = np.linspace(0, 100, 20)
    r = pk.res_trap(t)
    assert np.sum(r) == 20

def test_prop_trap():
    t = np.linspace(0, 100, 20)
    h = pk.prop_trap(t)
    assert np.linalg.norm(h) == 0

def test_conc_trap():
    t = np.linspace(0, 100, 20)
    J = np.ones(len(t))
    C = pk.conc_trap(J, t)
    assert C[-1] == 100
    C = pk.conc_trap(J, dt=t[1])
    assert C[-1] == 100
    C0 = tools.conv(pk.res_trap(t), J, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) == 0

def test_flux_trap():
    t = np.linspace(0, 100, 20)
    J = np.ones(len(t))
    Jo = pk.flux_trap(J, t)
    assert np.linalg.norm(Jo) == 0
    Jo = pk.flux_trap(J, dt=t[0])
    assert np.linalg.norm(Jo) == 0
    Jo0 = tools.conv(pk.prop_trap(t), J, t)
    assert np.linalg.norm(Jo-Jo0) == 0

if __name__ == "__main__":

    test_conc_trap()
    test_flux_trap()
    test_res_trap()
    test_prop_trap()

    print('All pk tests passed!!')