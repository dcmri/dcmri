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
    r = pk.res_comp(np.inf, t)
    assert np.sum(np.unique(r))== 1
    r = pk.res_comp(0, t)
    assert np.sum(r) == 1

def test_prop_comp():
    T = 25
    t = np.linspace(0, 150, 20)
    h = pk.prop_comp(T, t)
    assert np.round(np.trapz(h,t)) == 1
    t = [0,5,15,30,60,90,150]
    h = pk.prop_comp(T, t)
    assert (np.trapz(h,t)-1)**2 < 1e-2
    h = pk.prop_comp(np.inf, t)
    assert np.linalg.norm(h)==0
    h = pk.prop_comp(0, t)
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
    C = pk.conc_comp(J, np.inf, t)
    C0 = pk.conc_trap(J,t)
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
    Jo = pk.flux_comp(J, np.inf, t)
    assert np.linalg.norm(Jo) == 0

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
    C = pk.conc_plug(J, np.inf, t)
    C0 = pk.conc_trap(J)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_plug(J, 0, t)
    assert np.linalg.norm(C) == 0

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
    Jo = pk.flux_plug(J, np.inf, t)
    assert np.linalg.norm(Jo) == 0
    Jo = pk.flux_plug(J, 0, t)
    assert np.linalg.norm(Jo-J) == 0

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
    try:
        h = pk.prop_chain(-1, D, t)
    except:
        assert True
    else:
        assert False
    try:
        h = pk.prop_chain(T, -1, t)
    except:
        assert True
    else:
        assert False
    try:
        h = pk.prop_chain(T, 2, t)
    except:
        assert True
    else:
        assert False
    h = pk.prop_chain(T, 0, t)
    h0 = pk.prop_plug(T,t)
    assert np.linalg.norm(h-h0)==0
    h = pk.prop_chain(T, 1, t)
    h0 = pk.prop_comp(T, t)
    assert np.linalg.norm(h-h0)==0


def test_res_chain():
    T = 25
    D = 0.5
    t = np.linspace(0, 150, 500)
    h = pk.res_chain(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-6
    t = [0,5,15,30,60,90,150]
    h = pk.res_chain(T, D, t)
    assert (np.trapz(h,t)-T)**2/T**2 < 1e-3
    h = pk.res_chain(T, 0, t)
    h0 = pk.res_plug(T,t)
    assert np.linalg.norm(h-h0)==0
    h = pk.res_chain(T, 1, t)
    h0 = pk.res_comp(T, t)
    assert np.linalg.norm(h-h0)==0

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
    h = pk.conc_chain(J, T, 0, t)
    h0 = pk.conc_plug(J, T,t)
    assert np.linalg.norm(h-h0)==0
    h = pk.conc_chain(J, T, 1, t)
    h0 = pk.conc_comp(J, T, t)
    assert np.linalg.norm(h-h0)==0

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
    h = pk.flux_chain(J, T, 0, t)
    h0 = pk.flux_plug(J, T,t)
    assert np.linalg.norm(h-h0)==0
    h = pk.flux_chain(J, T, 1, t)
    h0 = pk.flux_comp(J, T, t)
    assert np.linalg.norm(h-h0)==0

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
    try:
        h = pk.prop_step(-1, D, t)
    except:
        assert True
    else:
        assert False
    try:
        h = pk.prop_step(T, -1, t)
    except:
        assert True
    else:
        assert False
    try:
        h = pk.prop_step(T, 2, t)
    except:
        assert True
    else:
        assert False
    h = pk.prop_step(np.inf, D, t)
    assert np.linalg.norm(h)==0
    h = pk.prop_step(T, 0, t)
    h0 = pk.prop_plug(T, t)
    assert np.linalg.norm(h-h0)==0

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
    C = pk.conc_step(J, T, 0, t)
    C0 = pk.conc_plug(J, T, t)
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
    Jo = pk.flux_step(J, T, 0, t)
    J0 = pk.flux_plug(J, T, t)
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
    h = pk.prop_free([1,1], t)
    h0 = pk.prop_free([1,1], t, TTmax=np.amax(t))
    assert np.linalg.norm(h-h0)/np.linalg.norm(h0) < 1e-12
    try:
        h = pk.prop_free([1,1,1], t, TT=[30,60,90])
    except:
        assert True
    else:
        assert False


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


def test_K_ncomp():
    T1, T2 = 2, 7
    E11, E12, E21, E22 = 1, 3, 9, 2
    K = pk.K_ncomp([T1,T2], [[E11,E12],[E21,E22]])
    K0 = [[(E11+E21)/T1, -E12/T2],[-E21/T1, (E12+E22)/T2]]
    assert np.array_equal(K, K0)
    assert np.array_equal(np.matmul(K, [1,0]), [(E11+E21)/T1,-E21/T1])
    assert np.array_equal(np.matmul(K, [0,1]), [-E12/T2,(E12+E22)/T2])
    assert np.array_equal(np.matmul(K, [1,0]), K[:,0]) # col 0
    assert np.array_equal(np.matmul(K, [0,1]), K[:,1]) # col 1
    try:
        pk.K_ncomp([T1,T2], [[E11,-1],[E21,E22]])
    except:
        assert True
    else:
        assert False
    K = pk.K_ncomp([T1,T2], [[0,E12],[0,E22]])
    assert K[0,0] == 0

def test_J_ncomp():
    T = [1,2]
    E = [[1,0],[0,1]]
    C = [[1],
         [3]]
    J = pk.J_ncomp(np.array(C), T, E)
    assert J[0,0,:] == 1
    assert J[0,1,:] == 0
    assert J[1,0,:] == 0
    assert J[1,1,:] == 1.5

def test_conc_ncomp_prop():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0],[0,1]]
    C0 = pk.conc_comp(J[0,:], T[0], t)
    C1 = pk.conc_comp(J[1,:], T[1], t)
    prec = [1e-1, 1e-2, 1e-3]
    for i, dt_prop in enumerate([None, 0.1, 0.01]):
        C = pk.conc_ncomp_prop(J, T, E, t, dt_prop=dt_prop)
        assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < prec[i]
        assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < prec[i]

def test_conc_ncomp_diag():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0],[0,1]]
    C0 = pk.conc_comp(J[0,:], T[0], t)
    C1 = pk.conc_comp(J[1,:], T[1], t)
    C = pk.conc_ncomp_diag(J, T, E, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 1e-12

def test_conc_ncomp():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0],[0,1]]
    C0 = pk.conc_comp(J[0,:], T[0], t)
    C1 = pk.conc_comp(J[1,:], T[1], t)
    prec = [1e-1, 1e-2, 1e-3]
    for i, dt_prop in enumerate([None, 0.1, 0.01]):
        C = pk.conc_ncomp(J, T, E, t, dt_prop=dt_prop)
        assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < prec[i]
        assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < prec[i]
    # Compare both solvers for a coupled system
    E = [[1,0.5],[0.5,1]]
    C = pk.conc_ncomp(J, T, E, t, solver='prop', dt_prop=0.01)
    C0 = pk.conc_ncomp(J, T, E, t, solver='diag')
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3

def test_flux_ncomp():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0],[0,1]]
    J0 = pk.conc_comp(J[0,:], T[0], t) / T[0]
    J1 = pk.conc_comp(J[1,:], T[1], t) / T[1]
    J = pk.flux_ncomp(J, T, E, t, dt_prop=0.01)
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 1e-3
    assert np.linalg.norm(J[1,1,:]-J1)/np.linalg.norm(J1) < 1e-3
    assert np.linalg.norm(J[1,0,:]) == 0
    assert np.linalg.norm(J[0,1,:]) == 0

def test_res_ncomp():
    # Compare a decoupled system against analytical 1-comp model solutions.
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = [[1,0],[0,1]]
    R0 = pk.res_comp(T[0], t)
    R1 = pk.res_comp(T[1], t)
    R = pk.res_ncomp(T, E, t)
    assert np.linalg.norm(R[0,0,:]-R0)/np.linalg.norm(R0) < 1e-12
    assert np.linalg.norm(R[1,1,:]-R1)/np.linalg.norm(R1) < 1e-12
    assert np.linalg.norm(R[1,0,:]) == 0
    assert np.linalg.norm(R[0,1,:]) == 0
    # Compare against convolution on a coupled system.
    J = np.ones((2,len(t)))
    J[1,:] = 2
    E = [[1,0.5],[0.5,1]]
    C = pk.conc_ncomp(J, T, E, t, solver='prop', dt_prop=0.01)
    R = pk.res_ncomp(T, E, t)
    C0 = tools.conv(R[0,0,:], J[0,:], t) + tools.conv(R[1,0,:], J[1,:], t)
    C1 = tools.conv(R[0,1,:], J[0,:], t) + tools.conv(R[1,1,:], J[1,:], t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.1
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01

def test_prop_ncomp():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = [[1,0],[0,1]]
    H0 = pk.prop_comp(T[0], t) 
    H1 = pk.prop_comp(T[1], t)
    H = pk.prop_ncomp(T, E, t)
    assert np.linalg.norm(H[0,0,0,:]-H0)/np.linalg.norm(H0) < 1e-12
    assert np.linalg.norm(H[1,1,1,:]-H1)/np.linalg.norm(H1) < 1e-12
    assert np.linalg.norm(H[:,1,0,:]) == 0
    assert np.linalg.norm(H[:,0,1,:]) == 0
    # Compare against convolution on a coupled system
    J = np.ones((2,len(t)))
    J[1,:] = 2
    E = [[1,0.5],[0.5,1]]
    Jo = pk.flux_ncomp(J, T, E, t, solver='prop', dt_prop=0.01)
    H = pk.prop_ncomp(T, E, t)
    Jo0 = tools.conv(H[0,0,0,:], J[0,:], t) + tools.conv(H[1,0,0,:], J[1,:], t)
    Jo1 = tools.conv(H[0,1,1,:], J[0,:], t) + tools.conv(H[1,1,1,:], J[1,:], t)
    assert np.linalg.norm(Jo[0,0,:]-Jo0)/np.linalg.norm(Jo0) < 0.1
    assert np.linalg.norm(Jo[1,1,:]-Jo1)/np.linalg.norm(Jo1) < 0.01


def test_K_2comp():
    # Compare against numerical computation
    T = [6,12]
    E = [[1,0.5],[0.5,1]]
    # Analytical
    _, K, _ = pk.K_2comp(T,E)
    # Numerical
    Knum = pk.K_ncomp(T, E)
    Knum, _ = np.linalg.eig(Knum)
    assert np.linalg.norm(K-Knum)/np.linalg.norm(Knum) < 1e-12

def test_conc_2comp():
    # Compare a coupled system against numerical solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0.5],[0.5,1]]
    C0 = pk.conc_2comp(J, T, E, t) 
    C = pk.conc_ncomp(J, T, E, t, dt_prop=0.1)
    assert np.linalg.norm(C[0,:]-C0[0,:])/np.linalg.norm(C[0,:]) < 1e-2
    assert np.linalg.norm(C[1,:]-C0[1,:])/np.linalg.norm(C[1,:]) < 1e-2
    # Compare a decoupled system against individual solutions
    E = [[1,0],[0,1]]
    C = pk.conc_2comp(J, T, E, t)
    C0 = pk.conc_comp(J[0,:], T[0], t)
    C1 = pk.conc_comp(J[1,:], T[1], t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 1e-12
    try:
        pk.conc_2comp(J, [6,0], E, t)
    except:
        assert True
    else:
        assert False
    C = pk.conc_2comp([[1,1],[1,1]], [1,1], [[1,0],[0,1]])
    assert C[0,0]==0


def test_flux_2comp():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 2
    T = [6,12]
    E = [[1,0],[0,1]]
    J0 = pk.conc_comp(J[0,:], T[0], t) / T[0]
    J1 = pk.conc_comp(J[1,:], T[1], t) / T[1]
    J = pk.flux_2comp(J, T, E, t)
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 1e-3
    assert np.linalg.norm(J[1,1,:]-J1)/np.linalg.norm(J1) < 1e-3
    assert np.linalg.norm(J[1,0,:]) == 0
    assert np.linalg.norm(J[0,1,:]) == 0

def test_res_2comp():
    # Compare a decoupled system against analytical 1-comp model solutions.
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = [[1,0],[0,1]]
    R0 = pk.res_comp(T[0], t)
    R1 = pk.res_comp(T[1], t)
    R = pk.res_2comp(T, E, t)
    assert np.linalg.norm(R[0,0,:]-R0)/np.linalg.norm(R0) < 1e-12
    assert np.linalg.norm(R[1,1,:]-R1)/np.linalg.norm(R1) < 1e-12
    assert np.linalg.norm(R[1,0,:]) == 0
    assert np.linalg.norm(R[0,1,:]) == 0
    # Compare against convolution on a coupled system.
    J = np.ones((2,len(t)))
    J[1,:] = 2
    E = [[1,0.5],[0.5,1]]
    C = pk.conc_2comp(J, T, E, t)
    R = pk.res_2comp(T, E, t)
    C0 = tools.conv(R[0,0,:], J[0,:], t) + tools.conv(R[1,0,:], J[1,:], t)
    C1 = tools.conv(R[0,1,:], J[0,:], t) + tools.conv(R[1,1,:], J[1,:], t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.1
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01

def test_prop_2comp():
    # Compare a decoupled system against analytical 1-comp model solutions
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = [[1,0],[0,1]]
    H0 = pk.prop_comp(T[0], t) 
    H1 = pk.prop_comp(T[1], t)
    H = pk.prop_2comp(T, E, t)
    assert np.linalg.norm(H[0,0,0,:]-H0)/np.linalg.norm(H0) < 1e-12
    assert np.linalg.norm(H[1,1,1,:]-H1)/np.linalg.norm(H1) < 1e-12
    assert np.linalg.norm(H[:,1,0,:]) == 0
    assert np.linalg.norm(H[:,0,1,:]) == 0
    # Compare against convolution on a coupled system
    J = np.ones((2,len(t)))
    J[1,:] = 2
    E = [[1,0.5],[0.5,1]]
    Jo = pk.flux_2comp(J, T, E, t)
    H = pk.prop_2comp(T, E, t)
    Jo0 = tools.conv(H[0,0,0,:], J[0,:], t) + tools.conv(H[1,0,0,:], J[1,:], t)
    Jo1 = tools.conv(H[0,1,1,:], J[0,:], t) + tools.conv(H[1,1,1,:], J[1,:], t)
    assert np.linalg.norm(Jo[0,0,:]-Jo0)/np.linalg.norm(Jo0) < 0.1
    assert np.linalg.norm(Jo[1,1,:]-Jo1)/np.linalg.norm(Jo1) < 0.01


def test_conc_2cxm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    C0 = pk.conc_2cxm(J[0,:], T, E, t) 
    C = pk.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    try:
        pk.conc_2cxm(J, [0,12], Emat, t)
    except:
        assert True
    else:
        assert False
    C = pk.conc_2cxm([1,1], [1,1], 0.5)
    assert C[0,0]==0


def test_flux_2cxm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    Jo0 = pk.flux_2cxm(J[0,:], T, E, t) 
    Jo = pk.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3

def test_res_2cxm():
    # Compare against convolution on a coupled system.
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = 0.2
    J = np.ones(len(t))
    C = pk.conc_2cxm(J, T, E, t)
    R = pk.res_2cxm(T, E, t)
    C0 = tools.conv(R[0,:], J, t) 
    C1 = tools.conv(R[1,:], J, t) 
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.1
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.1

def test_prop_2cxm():
    # Compare against convolution on a coupled system
    t = np.linspace(0, 20, 100)
    T = [6,12]
    E = 0.2
    J = np.ones(len(t))
    Jo = pk.flux_2cxm(J, T, E, t)
    H = pk.prop_2cxm(T, E, t)
    Jo0 = tools.conv(H[0,0,:], J, t) 
    assert np.linalg.norm(Jo[0,0,:]-Jo0)/np.linalg.norm(Jo0) < 1e-3

def test_conc_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    C0 = pk.conc_2cfm(J[0,:], T, E, t) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    C = pk.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-2
    C = pk.conc_2cfm(J[0,:], T, 0, t)
    C0 = pk.conc_comp(J[0,:], 6, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
    C = pk.conc_2cfm(J[0,:], [0,12], 1, t)
    assert np.linalg.norm(C[0,:]) == 0

def test_flux_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Jo0 = pk.flux_2cfm(J[0,:], T, E, t) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    Jo = pk.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3

def test_res_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = 0.2
    R0 = pk.res_2cfm(T, E, t) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    R = pk.res_2comp(T, Emat, t)
    assert np.linalg.norm(R[0,:,:]-R0)/np.linalg.norm(R0) < 1e-2
    R = pk.res_2cfm(T, 0, t)
    R0 = pk.res_comp(6, t)
    assert np.linalg.norm(R[0,:]-R0)/np.linalg.norm(R0) < 1e-12
    R = pk.res_2cfm([0,12], 1, t)
    assert np.sum(R[0,:]) == 1

def test_prop_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    T = [6,12]
    E = 0.2
    H0 = pk.prop_2cfm(T, E, t) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    H = pk.prop_2comp(T, Emat, t)
    assert np.linalg.norm(H[0,:,:,:]-H0)/np.linalg.norm(H0) < 1e-2

def test_conc_nscomp():

    # Some inputs
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))

    # Test exceptions
    try:
        pk.conc_nscomp(J, T, t)
    except:
        assert True
    try:
        pk.conc_nscomp(J, [T,T], t)
    except:
        assert True
    try:
        pk.conc_nscomp(J, -T*np.ones(len(t)), t)
    except:
        assert True

    # Test functionality
    C = pk.conc_comp(J, T, t)
    C0 = pk.conc_nscomp(J, T*np.ones(len(t)), t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.1
    
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C = pk.conc_comp(J, T, t)
    C0 = pk.conc_nscomp(J, T*np.ones(len(t)), t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.1


def test_flux_nscomp():

    # Some inputs
    T = 25
    t = np.linspace(0, 150, 20)
    J = np.ones(len(t))

    # Test functionality
    C = pk.flux_comp(J, T, t)
    C0 = pk.flux_nscomp(J, T*np.ones(len(t)), t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.1
    
    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C = pk.flux_comp(J, T, t)
    C0 = pk.flux_nscomp(J, T*np.ones(len(t)), t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.1


def test_conc_mmcomp():
    T = 25
    t = np.linspace(0, 150, 500)
    J = np.ones(len(t))
    C = pk.conc_comp(J, T, t)
    Km = np.amax(C)/2
    Vmax = np.amax(C)/np.amax(T)
    C = pk.conc_mmcomp(J, Vmax, Km, t, solver='SM')
    C0 = pk.conc_mmcomp(J, Vmax, Km, t, solver='prop')
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-2

    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C = pk.conc_mmcomp(J, Vmax, Km, t, solver='SM')
    C0 = pk.conc_mmcomp(J, Vmax, Km, t, solver='prop')
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.3

    try:
        C = pk.conc_mmcomp(J, -Vmax, Km, t, solver='SM')
    except:
        True
    try:
        C = pk.conc_mmcomp(J, Vmax, -Km, t, solver='SM')
    except:
        True

def test_flux_mmcomp():
    T = 25
    t = np.linspace(0, 150, 500)
    J = np.ones(len(t))
    C = pk.conc_comp(J, T, t)
    Km = np.amax(C)/2
    Vmax = np.amax(C)/np.amax(T)
    C = pk.flux_mmcomp(J, Vmax, Km, t, solver='SM')
    C0 = pk.flux_mmcomp(J, Vmax, Km, t, solver='prop')
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 1e-2

    t = [0,5,15,30,60,90,150]
    J = np.ones(len(t))
    C = pk.flux_mmcomp(J, Vmax, Km, t, solver='SM')
    C0 = pk.flux_mmcomp(J, Vmax, Km, t, solver='prop')
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.3



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

    test_K_2comp()
    test_K_ncomp()
    test_J_ncomp()
    test_conc_ncomp_prop()
    test_conc_ncomp_diag()

    test_conc_ncomp()
    test_flux_ncomp()
    test_res_ncomp()
    test_prop_ncomp()

    test_conc_2comp()
    test_flux_2comp()
    test_res_2comp()
    test_prop_2comp()

    test_conc_2cxm()
    test_flux_2cxm()
    test_res_2cxm()
    test_prop_2cxm()

    test_conc_2cfm()
    test_flux_2cfm()
    test_res_2cfm()
    test_prop_2cfm()

    test_conc_nscomp()
    test_flux_nscomp()

    test_conc_mmcomp()
    test_flux_mmcomp()

    print('All pk tests passed!!')