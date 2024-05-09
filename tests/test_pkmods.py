import numpy as np
import dcmri as dc
from scipy.integrate import cumulative_trapezoid

def test_conc_1cum():
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_1cum(ca, Fp, t)
    C0 = Fp*cumulative_trapezoid(ca, t, initial=0)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test_flux_1cum():
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_1cum(ca, Fp, t)
    J0 = np.zeros(len(t))
    assert np.linalg.norm(J-J0) < 0.01

def test_conc_1cm():
    n = 10
    Ta = 10
    Fp = 2
    v = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_1cm(ca, Fp, v, t)
    C0 = Fp*dc.biexpconv(Ta, v/Fp, t)*v/Fp
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_1cm(ca, 0, v, t)
    assert np.linalg.norm(C) == 0

def test_flux_1cm():
    n = 10
    Ta = 10
    Fp = 2
    v = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_1cm(ca, Fp, v, t)
    J0 = Fp*dc.biexpconv(Ta, v/Fp, t)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_1cm(ca, 0, v, t)
    assert np.linalg.norm(J) == 0

def test_conc_tofts():
    n = 10
    Ta = 10
    Ktrans = 2
    kep = 5
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tofts(ca, Ktrans, kep, t)
    C0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)/kep
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tofts(ca, Ktrans, 0, t)
    C0 = dc.conc_trap(Ktrans*ca, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tofts(ca, Ktrans, 0, t, sum=False)
    assert np.linalg.norm(C[1,:]-C0)/np.linalg.norm(C0) < 0.01
    

def test_flux_tofts():
    n = 10
    Ta = 10
    Ktrans = 2
    kep = 5
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tofts(ca, Ktrans, kep, t)
    J0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tofts(ca, Ktrans, 0, t)
    assert np.linalg.norm(J[0,1,:]) < 0.01
    

def test_conc_patlak():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_patlak(ca, vp, Ktrans, t, sum=False)
    C0 = vp*ca
    C1 = Ktrans*dc.conc_trap(ca, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_patlak(ca, vp, Ktrans, t, sum=True)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test_flux_patlak():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_patlak(ca, vp, Ktrans, t)
    J0 = Ktrans*ca
    assert np.linalg.norm(J[1,0,:]-J0)/np.linalg.norm(J0) < 0.01

def test_conc_etofts():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    ve = 0.2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_etofts(ca, vp, Ktrans, ve, t, sum=False)
    C0 = vp*ca
    C1 = Ktrans*dc.biexpconv(Ta, ve/Ktrans, t)*ve/Ktrans
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_etofts(ca, vp, Ktrans, ve, t, sum=True)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_etofts(ca, vp, Ktrans, 0, t, sum=False)
    C0 = vp*ca
    C1 = Ktrans*cumulative_trapezoid(ca,t,initial=0)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01

def test_flux_etofts():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    kep = 5
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_etofts(ca, vp, Ktrans, kep, t)
    J0 = kep*Ktrans*dc.biexpconv(Ta, 1/kep, t)/kep
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_etofts(ca, vp, Ktrans, 0, t)
    assert np.linalg.norm(J[0,1,:]) == 0

def test_conc_2cum():
    n = 10
    Ta = 10
    vp = 0.3
    PS = 2
    Fp = 4
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_2cum(ca, Fp, vp, PS, t, sum=False)
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_2cum(ca, Fp, vp, PS, t, sum=True)
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_2cum(ca, Fp, 0, PS, t, sum=True)
    Ktrans = Fp*PS/(Fp+PS)
    C0 = dc.conc_1cum(ca, Ktrans, t)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_2cum(ca, 0, vp, 0, t, sum=True)
    assert np.linalg.norm(Cs)==0

def test_flux_2cum():
    n = 10
    Ta = 10
    vp = 0.3
    PS = 2
    Fp = 4
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_2cum(ca, Fp, vp, PS, t)
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_2cum(ca, Fp, 0, PS, t)
    assert np.linalg.norm(J[0,0,:]-Fp*ca)/np.linalg.norm(Fp*ca) < 0.01

def test_conc_2cxm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    ve = T[1]*PS
    vp = T[0]*(Fp+PS)
    C0 = dc.conc_2cxm(ca, Fp, vp, PS, ve, t, sum=False)
    C = dc.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = dc.conc_2cxm(ca, Fp, vp, PS, ve, t, sum=True)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_2cxm(ca, 0, vp, 0, ve, t, sum=True)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_2cxm(ca, 0, vp, 0, ve, t, sum=False)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_2cxm(ca, Fp, vp, 0, ve, t, sum=True)
    C0 = dc.conc_1cm(ca, Fp, vp, t)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = dc.conc_2cxm(ca, Fp, vp, 0, ve, t, sum=False)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    
def test_flux_2cxm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    ve = T[1]*PS
    vp = T[0]*(Fp+PS)
    Jo0 = dc.flux_2cxm(ca, Fp, vp, PS, ve, t) 
    Jo = dc.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3
    Jo0 = dc.flux_2cxm(ca, 0, vp, 0, ve, t) 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = dc.flux_2cxm(ca, Fp, vp, 0, ve, t) 
    Jo = dc.flux_1cm(ca, Fp, vp, t)
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3


def test_conc_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    Te = T[1]
    vp = T[0]*(Fp+PS)
    C0 = dc.conc_2cfm(ca, Fp, vp, PS, Te, t, sum=False) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    C = dc.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-2
    Cs = dc.conc_2cfm(ca, Fp, vp, PS, Te, t, sum=True)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_2cfm(ca, Fp, vp, 0, Te, t, sum=False)
    C0 = dc.conc_comp(J[0,:], vp/Fp, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
    C = dc.conc_2cfm(ca, Fp, 0, PS, Te, t, sum=False)
    assert np.linalg.norm(C[0,:]) == 0
    Cs = dc.conc_2cfm(ca, 0, vp, 0, Te, t, sum=True)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_2cfm(ca, 0, vp, 0, Te, t, sum=False)
    assert np.linalg.norm(Cs) == 0


def test_flux_2cfm():
    # Compare against general 2comp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    Te = T[1]
    vp = T[0]*(Fp+PS)
    Jo0 = dc.flux_2cfm(ca, Fp, vp, PS, Te, t) 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    Jo = dc.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3
    Jo0 = dc.flux_2cfm(ca, 0, vp, 0, Te, t) 
    assert np.linalg.norm(Jo0) == 0




if __name__ == "__main__":

    test_conc_1cum()
    test_flux_1cum()

    test_conc_1cm()
    test_flux_1cm()

    test_conc_tofts()
    test_flux_tofts()

    test_conc_patlak()
    test_flux_patlak()

    test_conc_etofts()
    test_flux_etofts()

    test_conc_2cum()
    test_flux_2cum()

    test_conc_2cxm()
    test_flux_2cxm()

    test_conc_2cfm()
    test_flux_2cfm()

    print('All pkmods tests passing!')