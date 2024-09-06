from scipy.integrate import cumulative_trapezoid

import numpy as np
import dcmri as dc


def test__conc_u():
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, Fp, t=t, kinetics='U')
    C0 = Fp*cumulative_trapezoid(ca, t, initial=0)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_u():
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, Fp, t=t, kinetics='U')
    J0 = np.zeros(len(t))
    assert np.linalg.norm(J-J0) < 0.01

def test__conc_1c():
    n = 10
    Ta = 10
    Fp = 2
    v = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, Fp, v, t=t, kinetics='FX')
    C0 = Fp*dc.biexpconv(Ta, v/Fp, t)*v/Fp
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, 0, v, t=t, kinetics='FX')
    assert np.linalg.norm(C) == 0

def test__flux_1c():
    n = 10
    Ta = 10
    Fp = 2
    v = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, Fp, v, t=t, kinetics='FX')
    J0 = Fp*dc.biexpconv(Ta, v/Fp, t)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, 0, v, t=t, kinetics='FX')
    assert np.linalg.norm(J) == 0


def test__conc_wv():
    n = 10
    Ta = 10
    Ktrans = 2
    ve = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, Ktrans, ve, t=t, kinetics='WV')
    C0 = Ktrans*dc.biexpconv(Ta, ve/Ktrans, t)*ve/Ktrans
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, Ktrans, ve, t=t, kinetics='WV', sum=False)
    C = C[0,:] + C[1,:]
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, 0, ve, t=t, kinetics='WV')
    assert np.linalg.norm(C) == 0

def test__flux_wv():
    n = 10
    Ta = 10
    Ktrans = 2
    kep = 5
    ve = Ktrans/kep
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, Ktrans, ve, t=t, kinetics='WV')
    J0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, 0, ve, t=t, kinetics='WV')
    assert np.linalg.norm(J[0,1,:]) == 0
    

def test__conc_hfu():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, vp, Ktrans, t=t, kinetics='HFU', sum=False)
    C0 = vp*ca
    C1 = Ktrans*dc.conc_trap(ca, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue(ca, vp, Ktrans, t=t, kinetics='HFU', sum=True)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_hfu():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, vp, Ktrans, t=t, kinetics='HFU')
    J0 = Ktrans*ca
    assert np.linalg.norm(J[1,0,:]-J0)/np.linalg.norm(J0) < 0.01

def test__conc_hf():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    ve = 0.2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, vp, Ktrans, ve, t=t, kinetics='HF', sum=False)
    C0 = vp*ca
    C1 = Ktrans*dc.biexpconv(Ta, ve/Ktrans, t)*ve/Ktrans
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue(ca, vp, Ktrans, ve, t=t, kinetics='HF', sum=True)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, vp, Ktrans, 0, t=t, kinetics='HF', sum=False)
    C0 = vp*ca
    C1 = np.zeros(len(ca))
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1) == 0
    C = dc.conc_tissue(ca, vp, 0, ve, t=t, kinetics='HF', sum=False)
    assert 0==np.linalg.norm(C[1,:])

def test__flux_hf():
    n = 10
    Ta = 10
    vp = 0.3
    Ktrans = 2
    kep = 5
    ve = Ktrans/kep
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, vp, Ktrans, ve, t=t, kinetics='HF')
    J0 = kep*Ktrans*dc.biexpconv(Ta, 1/kep, t)/kep
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, vp, 0, ve, t=t, kinetics='HF')
    assert 0==np.linalg.norm(J[0,1,:])

def test__conc_2cu():
    n = 10
    Ta = 10
    vp = 0.3
    PS = 2
    Fp = 4
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, Fp, vp, PS, t=t, sum=False, kinetics='2CU')
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, Fp, vp, PS, t=t, sum=True, kinetics='2CU')
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, Fp, 0, PS, t=t, sum=True, kinetics='2CU')
    Ktrans = Fp*PS/(Fp+PS)
    C0 = dc.conc_tissue(ca, Ktrans, t=t, kinetics='U')
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, 0, vp, 0, t=t, sum=True, kinetics='2CU')
    assert np.linalg.norm(Cs)==0

def test__flux_2cu():
    n = 10
    Ta = 10
    vp = 0.3
    PS = 2
    Fp = 4
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, Fp, vp, PS, t=t, kinetics='2CU')
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, Fp, 0, PS, t=t, kinetics='2CU')
    assert np.linalg.norm(J[0,0,:]-Fp*ca)/np.linalg.norm(Fp*ca) < 0.01

def test__conc_2cx():
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
    C0 = dc.conc_tissue(ca, Fp, vp, PS, ve, t=t, sum=False, kinetics='2CX')
    C = dc.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = dc.conc_tissue(ca, Fp, vp, PS, ve, t=t, sum=True, kinetics='2CX')
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, 0, vp, 0, ve, t=t, sum=True, kinetics='2CX')
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue(ca, 0, vp, 0, ve, t=t, sum=False, kinetics='2CX')
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue(ca, Fp, vp, 0, ve, t=t, sum=True, kinetics='2CX')
    C0 = dc.conc_tissue(ca, Fp, vp, t=t, kinetics='NX')
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = dc.conc_tissue(ca, Fp, vp, 0, ve, t=t, sum=False, kinetics='2CX')
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    C = dc.conc_tissue(ca, 0, 0, 0, 0, t=t, sum=True, kinetics='2CX')
    assert np.linalg.norm(C) == 0
    
def test__flux_2cx():
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
    Jo0 = dc.flux_tissue(ca, Fp, vp, PS, ve, t=t, kinetics='2CX') 
    Jo = dc.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3
    Jo0 = dc.flux_tissue(ca, 0, vp, 0, ve, t=t, kinetics='2CX') 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = dc.flux_tissue(ca, Fp, vp, 0, ve, t=t, kinetics='2CX') 
    Jo = dc.flux_tissue(ca, Fp, vp, t=t, kinetics='NX')
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3


def test__conc_2cf():
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
    C0 = dc.conc_tissue(ca, Fp, vp, PS, Te, t=t, sum=False, kinetics='2CF') 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    C = dc.conc_2comp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-2
    Cs = dc.conc_tissue(ca, Fp, vp, PS, Te, t=t, sum=True, kinetics='2CF')
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, Fp, vp, 0, Te, t=t, sum=False, kinetics='2CF')
    C0 = dc.conc_comp(J[0,:], vp/Fp, t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
    C = dc.conc_tissue(ca, Fp, 0, PS, Te, t=t, sum=False, kinetics='2CF')
    assert np.linalg.norm(C[0,:]) == 0
    Cs = dc.conc_tissue(ca, 0, vp, 0, Te, t=t, sum=True, kinetics='2CF')
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue(ca, 0, vp, 0, Te, t=t, sum=False, kinetics='2CF')
    assert np.linalg.norm(Cs) == 0


def test__flux_2cf():
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
    Jo0 = dc.flux_tissue(ca, Fp, vp, PS, Te, t=t, kinetics='2CF') 
    Emat = [
        [1-E, 0],
        [E  , 1]]
    Jo = dc.flux_2comp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3
    Jo0 = dc.flux_tissue(ca, 0, vp, 0, Te, t=t, kinetics='2CF') 
    assert np.linalg.norm(Jo0) == 0


def test_conc_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        dc.conc_tissue(ca, Fp, t=t, kinetics='blabla')
    except:
        assert True
    else:
        assert False
    

def test_flux_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fp = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        dc.flux_tissue(ca, Fp, t=t, kinetics='blabla')
    except:
        assert True
    else:
        assert False


if __name__ == "__main__":

    test__conc_u()
    test__flux_u()

    test__conc_1c()
    test__flux_1c()

    test__conc_wv()
    test__flux_wv()

    test__conc_hfu()
    test__flux_hfu()

    test__conc_hf()
    test__flux_hf()

    test__conc_2cu()
    test__flux_2cu()

    test__conc_2cx()
    test__flux_2cx()

    test__conc_2cf()
    test__flux_2cf()

    test_conc_tissue()
    test_flux_tissue()

    print('All pk_tissue tests passing!')