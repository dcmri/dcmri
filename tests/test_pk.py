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
    Jo = pk.flux_trap(J)
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
    C = pk.conc_pass(J, T)
    assert np.unique(C)[0] == T
    C0 = tools.conv(pk.res_pass(T,t), J, t)
    assert np.linalg.norm(C[1:]-C0[1:])/np.linalg.norm(C0[1:]) < 1e-12

def test_flux_pass():
    T = 30
    t = np.linspace(0, 100, 20)
    J = np.ones(len(t))
    Jo = pk.flux_pass(J)
    assert np.linalg.norm(J-Jo)/np.linalg.norm(J) < 1e-12
    Jo0 = tools.conv(pk.prop_pass(t), J, t)
    assert np.linalg.norm(Jo[1:]-Jo0[1:])/np.linalg.norm(Jo[1:]) < 1e-12


def test_res_comp():
    T = 25
    t = np.linspace(0, 150, 20)
    r = pk.res_comp(T, t)
    assert np.round(np.trapz(r,t)) == T
    t = [0,5,15,30,60,90,150]
    r = pk.res_comp(T, t)
    assert (np.trapz(r,t)-T)**2/T**2 < 1e-2

def test_prop_comp():
    T = 25
    t = np.linspace(0, 150, 20)
    h = pk.prop_comp(T, t)
    assert np.round(np.trapz(h,t)) == 1
    t = [0,5,15,30,60,90,150]
    h = pk.prop_comp(T, t)
    assert (np.trapz(h,t)-1)**2 < 1e-2

def test_conc_comp():
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_comp(T,t), J, t)
    C = pk.conc_comp(J, T, t)
    assert np.linalg.norm(C[1:]-C0[1:])/np.linalg.norm(C0[1:]) < 1e-2
    C = pk.conc_comp(J, T, dt=t[1])
    assert np.linalg.norm(C[1:]-C0[1:])/np.linalg.norm(C0[1:]) < 1e-2
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_comp(T,t), J, t)
    C = pk.conc_comp(J, T, t)
    assert np.linalg.norm(C[1:]-C0[1:])/np.linalg.norm(C0[1:]) < 1e-1

def test_flux_comp():
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_comp(T,t), J, t)
    Jo = pk.flux_comp(J, T, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-2
    Jo = pk.flux_comp(J, T, dt=t[1])
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-2
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_comp(T,t), J, t)
    Jo = pk.flux_comp(J, T, t)
    assert np.linalg.norm(J0-Jo)/np.linalg.norm(J0) < 1e-1

def test_res_plug():
    T = 25
    t = np.linspace(0, 150, 20)
    r = pk.res_plug(T, t)
    assert (np.trapz(r,t)-T)**2/T**2 < 0.02
    t = [0,5,15,30,60,90,150]
    r = pk.res_plug(T, t)
    assert (np.trapz(r,t)-T)**2/T**2 < 0.06

def test_prop_plug():
    T = 25
    t = np.linspace(0, 150, 500)
    h = pk.prop_plug(T, t)
    assert np.abs((np.trapz(h,t)-1)) < 1e-12
    t = [0,5,15,30,60,90,150]
    h = pk.prop_plug(T, t)
    assert np.abs((np.trapz(h,t)-1)) < 1e-12

def test_conc_plug():
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_plug(T,t), J, t)
    C = pk.conc_plug(J, T, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_plug(J, T, dt=t[1])
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_plug(T,t), J, t)
    C = pk.conc_plug(J, T, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12

def test_flux_plug():
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_plug(T,t), J, t)
    Jo = pk.flux_plug(J, T, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 0.2
    Jo = pk.flux_plug(J, T, dt=t[1])
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 0.2
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_plug(T,t), J, t)
    Jo = pk.flux_plug(J, T, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 0.3

def test_prop_chain():
    T = 25
    D = 0.5
    t = np.linspace(0, 150, 500)
    h = pk.prop_chain(T, D, t)
    assert np.abs(np.trapz(h,t)-1) < 0.001
    assert np.abs(np.trapz(t*h,t)-T) < 0.02
    t = [0,5,15,30,60,90,150]
    h = pk.prop_chain(T, D, t)
    assert np.abs(np.trapz(h,t)-1) < 0.03
    assert np.abs(np.trapz(t*h,t)-T) < 0.5

def test_res_chain():
    T = 25
    D = 0.5
    t = np.linspace(0, 150, 500)
    h = pk.res_chain(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-6
    t = [0,5,15,30,60,90,150]
    h = pk.res_chain(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-3

def test_conc_chain():
    T = 25
    D = 0.5
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_chain(T,D,t), J, t)
    C = pk.conc_chain(J, T, D, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_chain(J, T, D, dt=t[1])
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_chain(T,D,t), J, t)
    C = pk.conc_chain(J, T, D, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12

def test_flux_chain():
    T = 25
    D = 0.5
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_chain(T,D,t), J, t)
    Jo = pk.flux_chain(J, T, D, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    Jo = pk.flux_chain(J, T, D, dt=t[1])
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_chain(T,D,t), J, t)
    Jo = pk.flux_chain(J, T, D, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12

def test_prop_step():
    T, D = 25, 0.5
    t = np.linspace(0, 150, 100)
    h = pk.prop_step(T, D, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    assert np.abs(np.trapz(t*h,t)-T) < 1e-12
    h = pk.prop_step(T, 1, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = pk.prop_step(120, 0.5, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = pk.prop_step(120, 0.5, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    
    t = [0,5,15,30,60,90,150]
    h = pk.prop_step(T, D, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12

def test_res_step():
    T, D = 25, 0.5
    t = np.linspace(0, 150, 500)
    h = pk.res_step(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-9
    t = [0,5,15,30,60,90,150]
    h = pk.res_step(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 0.5

def test_conc_step():
    T, D = 25, 0.5
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_step(T,D,t), J, t)
    C = pk.conc_step(J, T, D, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_step(J, T, D, dt=t[1])
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_step(T,D,t), J, t)
    C = pk.conc_step(J, T, D, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12

def test_flux_step():
    T, D = 25, 0.5
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_step(T,D,t), J, t)
    Jo = pk.flux_step(J, T, D, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    Jo = pk.flux_step(J, T, D, dt=t[1])
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_step(T,D,t), J, t)
    Jo = pk.flux_step(J, T, D, t)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12


def test_prop_free():
    t = np.linspace(0, 150, 100)
    h = pk.prop_free([1,1], t, TT=[30,60,90])
    T = 60
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    assert (np.trapz(t*h,t)-T)**2/T**2 < 1e-9
    h = pk.prop_free([2,2], t, TT=[30,60,90])
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    assert (np.trapz(t*h,t)-T)**2/T**2 < 1e-9
    h = pk.prop_free([2,5], t, TT=[30,60,90])
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = pk.prop_free([2,5], t, TT=[30,60,120])
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = pk.prop_free([2,5], t, TT=[30.5,60.5,120.5])
    assert np.abs(np.trapz(h,t)-1) < 1e-12

def test_res_free():
    t = np.linspace(0, 150, 100)
    h = pk.res_free([1,1], t, TT=[30,60,90])
    T = 60
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-9

def test_conc_free():
    H = [1,1]
    TT = [30,60,90]
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_free(H,t,TT), J, t)
    C = pk.conc_free(J, H, t, TT=TT)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_free(J, H, t, dt=t[1], TT=TT)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C0 = tools.conv(pk.res_free(H,t,TT=TT), J, t)
    C = pk.conc_free(J, H, t, TT=TT)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12

def test_flux_free():
    H = [1,1]
    TT = [30,60,90]
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_free(H,t,TT), J, t)
    Jo = pk.flux_free(J, H, t, TT=TT)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    Jo = pk.flux_free(J, H, t, dt=t[1], TT=TT)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    J0 = tools.conv(pk.prop_free(H,t,TT=TT), J, t)
    Jo = pk.flux_free(J, H, t, TT=TT)
    assert np.linalg.norm(Jo-J0)/np.linalg.norm(J0) < 1e-12

if __name__ == "__main__":

    test_conc_trap()
    test_flux_trap()
    test_res_trap()
    test_prop_trap()

    test_conc_pass()
    test_flux_pass()
    test_res_pass()
    test_prop_pass()

    test_conc_comp()
    test_flux_comp()
    test_res_comp()
    test_prop_comp()

    test_conc_plug()
    test_flux_plug()
    test_res_plug()
    test_prop_plug()

    test_conc_chain()
    test_flux_chain()
    test_prop_chain()
    test_res_chain()

    test_conc_step()
    test_flux_step()
    test_prop_step()
    test_res_step()

    test_conc_free()
    test_flux_free()
    test_prop_free()
    test_res_free()

    print('All pk tests passed!!')