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

def test_res_pass():
    T = 30
    t = np.linspace(0, 100, 20)
    r = pk.res_pass(T, t)
    assert round(r[0] * t[1]/2) == T
    assert np.linalg.norm(r[1:]) == 0
    
def test_prop_pass():
    T = 30
    t = np.linspace(0, 100, 20)
    h = pk.prop_pass(t)
    assert round(h[0] * t[1]/2) == 1
    assert np.linalg.norm(h[1:]) == 0

def test_conc_pass():
    T = 30
    t = np.linspace(0, 100, 20)
    J = np.ones(len(t))
    C = pk.conc_pass(J, T, t)
    assert np.unique(C)[0] == T
    C = pk.conc_pass(J, T, dt=t[1])
    assert np.unique(C)[0] == T
    C0 = tools.conv(pk.res_pass(T,t), J, t)
    assert np.linalg.norm(C[1:]-C0[1:])/np.linalg.norm(C0[1:]) < 1e-12

def test_flux_pass():
    T = 30
    t = np.linspace(0, 100, 20)
    J = np.ones(len(t))
    Jo = pk.flux_pass(J, t)
    assert np.linalg.norm(J-Jo)/np.linalg.norm(J) < 1e-12
    Jo = pk.flux_pass(J, dt=t[0])
    assert np.linalg.norm(J-Jo)/np.linalg.norm(J) < 1e-12
    Jo0 = tools.conv(pk.prop_pass(T,t), J, t)
    assert np.linalg.norm(Jo[1:]-Jo0[1:])/np.linalg.norm(Jo[1:]) < 1e-12

if __name__ == "__main__":

    test_conc_trap()
    test_flux_trap()
    test_res_trap()
    test_prop_trap()

    test_conc_pass()
    test_flux_pass()
    test_res_pass()
    test_prop_pass()

    print('All pk tests passed!!')