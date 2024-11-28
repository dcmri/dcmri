from scipy.integrate import cumulative_trapezoid

import numpy as np
import dcmri as dc



def test__conc_u():
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, kinetics='U', Fb=Fb)
    C0 = Fb*cumulative_trapezoid(ca, t, initial=0)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_u():
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, t=t, kinetics='U', Fb=Fb)
    J0 = np.zeros(len(t))
    assert np.linalg.norm(J-J0) < 0.01

def test__conc_fx():
    n = 10
    Ta = 10
    Fb = 2
    ve = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, kinetics='FX', H=H, ve=ve, Fb=Fb)
    Fp = Fb*(1-H)
    C0 = Fp*dc.biexpconv(Ta, ve/Fp, t)*ve/Fp/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='FX', H=H, ve=ve, Fb=0)
    assert np.linalg.norm(C) == 0

def test__flux_fx():
    n = 10
    Ta = 10
    Fb = 2
    ve = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, t=t, kinetics='FX', H=H, ve=ve, Fb=Fb)
    Fp = Fb*(1-H)
    J0 = Fp*dc.biexpconv(Ta, ve/Fp, t)/(1-H)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, t=t, kinetics='FX', H=H, ve=ve, Fb=0)
    assert np.linalg.norm(J) == 0


def test__conc_wv():
    n = 10
    Ta = 10
    Ktrans = 2
    vi = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, kinetics='WV', H=H,  vi=vi, Ktrans=Ktrans)
    C0 = Ktrans*dc.biexpconv(Ta, vi/Ktrans, t)*vi/Ktrans/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='WV', H=H, vi=vi, Ktrans=Ktrans)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='WV', H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(C) == 0


def test__flux_wv():
    n = 10
    Ta = 10
    Ktrans = 2
    kep = 5
    H = 0.45
    vi = Ktrans/kep
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, t=t, kinetics='WV', H=H, vi=vi, Ktrans=Ktrans)
    J0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, t=t, kinetics='WV', H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(J[0,1,:]) == 0
    

def test__conc_hfu():
    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, kinetics='HFU', sum=False, H=H, vb=vb, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.conc_trap(ca/(1-H), t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='HFU', sum=True, H=H, vb=vb, PS=PS)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_hfu():
    n = 10
    Ta = 10
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, kinetics='HFU', H=H, PS=PS)
    J0 = PS*ca/(1-H)
    assert np.linalg.norm(J[1,0,:]-J0)/np.linalg.norm(J0) < 0.01

def test__conc_hf():
    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    vi = 0.2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, kinetics='HF', sum=False, 
                       H=H, vb=vb, vi=vi, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.biexpconv(Ta, vi/PS, t)*vi/PS/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='HF', sum=True, 
                       H=H, vb=vb, vi=vi, PS=PS)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue(ca, t=t, kinetics='HF', sum=False, 
                       H=H, vb=vb, vi=0, PS=PS)
    C0 = vb*ca
    C1 = np.zeros(len(ca))
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1) == 0
    C = dc.conc_tissue(ca, t=t, kinetics='HF', sum=False, 
                       H=H, vb=vb, vi=vi, PS=0)
    assert 0==np.linalg.norm(C[1,:])

def test__flux_hf():
    n = 10
    Ta = 10
    PS = 2
    kep = 5
    vi = PS/kep
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, t=t, kinetics='HF', H=H, vi=vi, PS=PS)
    J0 = kep*PS*dc.biexpconv(Ta, 1/kep, t)/kep/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, t=t, kinetics='HF', H=H, vi=vi, PS=0)
    assert 0==np.linalg.norm(J[0,1,:])

def test__conc_2cu():
    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    Fb = 4
    H = 0.45
    Fp = (1-H)*Fb
    vp = (1-H)*vb
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue(ca, t=t, sum=False, kinetics='2CU', 
                       H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, t=t, sum=True, kinetics='2CU', 
                        H=H, vb=vb, Fb=Fb, PS=PS)
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, t=t, sum=True, kinetics='2CU', 
                        H=H, vb=0, Fb=Fb, PS=PS)
    C0 = dc.conc_tissue(ca/(1-H), t=t, kinetics='U', 
                        Fb=PS*Fp/(PS+Fp))
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca, t=t, sum=True, kinetics='2CU', 
                        H=H, vb=vb, Fb=0, PS=0)
    assert np.linalg.norm(Cs)==0

def test__flux_2cu():
    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    Fb = 4
    H = 0.45
    vp = (1-H)*vb
    Fp = (1-H)*Fb
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue(ca, t=t, kinetics='2CU', 
                       H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue(ca, t=t, kinetics='2CU', 
                       H=H, vb=0, Fb=Fb, PS=PS)
    assert np.linalg.norm(J[0,0,:]-Fb*ca)/np.linalg.norm(Fb*ca) < 0.01

def test__conc_2cx():
    # Compare against general ncomp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    vi = T[1]*PS
    vp = T[0]*(Fp+PS)
    H = 0.45
    Fb = Fp/(1-H)
    vb = vp/(1-H)
    C0 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C = dc.conc_ncomp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = dc.conc_tissue(ca*(1-H), t=t, sum=True, kinetics='2CX',
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue(ca*(1-H), t=t, sum=True, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=0, PS=0)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=0, PS=0)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue(ca*(1-H), t=t, sum=True, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=0)
    C0 = dc.conc_tissue(ca*(1-H), t=t, kinetics='NX', 
                        vb=vb, Fb=Fb)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                       H=H, vb=vb, vi=vi, Fb=Fb, PS=0)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    C = dc.conc_tissue(ca*(1-H), t=t, sum=True, kinetics='2CX', 
                       H=H, vb=0, vi=0, Fb=0, PS=0)
    assert np.linalg.norm(C) == 0

    # Test boundaries (Fp=inf)
    Fb = 0.01
    PS = 0.001
    vb = 0.1
    vi = 0.2
    C0 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.05
    C1 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.1
    C2 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 10.0
    C3 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = np.inf
    C4 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    # Check convergence to solution
    err0 = np.linalg.norm(C0[:,1:]-C4[:,1:])
    err1 = np.linalg.norm(C1[:,1:]-C4[:,1:])
    err2 = np.linalg.norm(C2[:,1:]-C4[:,1:])
    err3 = np.linalg.norm(C3[:,1:]-C4[:,1:])
    assert err1 < err0
    assert err2 < err1
    assert err3 < err2
    assert err3 < 1e-2
    # plt.plot(t, C0[0,:], 'r-')
    # plt.plot(t, C0[1,:], 'b-')
    # plt.plot(t, C1[0,:], 'r--')
    # plt.plot(t, C1[1,:], 'b--')
    # plt.plot(t, C2[0,:], 'r-.')
    # plt.plot(t, C2[1,:], 'b-.')
    # plt.plot(t, C3[0,:], 'r-.')
    # plt.plot(t, C3[1,:], 'b-.')
    # plt.plot(t, C4[0,:], 'ro')
    # plt.plot(t, C4[1,:], 'bo')
    # plt.show()

    # Test boundaries (PS+Fp=0)
    Fb = 0
    PS = 0
    vb = 0.1
    vi = 0.2
    C0 = dc.conc_tissue(ca*(1-H), t=t, sum=False, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    assert np.linalg.norm(C0) == 0

    # Test boundaries (PS=0)
    Fb = 0.01
    PS = 0
    vb = 0.1
    vi = 0.2
    C0 = dc.conc_tissue(ca*(1-H), t=t, sum=True, kinetics='2CX', 
                        H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C1 = dc.conc_tissue(ca*(1-H), t=t, kinetics='NX', 
                        vb=vb, Fb=Fb)
    assert np.linalg.norm(C0-C1) < 1e-9
    
def test__flux_2cx():
    # Compare against general ncomp solution
    t = np.linspace(0, 20, 10)
    J = np.ones((2,len(t)))
    J[1,:] = 0 # No inlet in 2nd compartment
    T = [6,12]
    E = 0.2
    Emat = [[1-E,1],[E,0]]
    Fp = 1/60 # mL/sec/mL
    ca = J[0,:]/Fp
    PS = Fp*E/(1-E)
    vi = T[1]*PS
    vp = T[0]*(Fp+PS)
    H = 0.45
    vb = vp/(1-H)
    Fb = Fp/(1-H) 
    Jo0 = dc.flux_tissue(ca*(1-H), t=t, kinetics='2CX', 
                         H=H, vb=vb, vi=vi, Fb=Fb, PS=PS) 
    Jo = dc.flux_ncomp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-1
    Jo0 = dc.flux_tissue(ca*(1-H), t=t, kinetics='2CX', 
                         H=H, vb=vb, vi=vi, Fb=0, PS=0) 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = dc.flux_tissue(ca*(1-H), t=t, kinetics='2CX', 
                         H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    Jo = dc.flux_tissue(ca*(1-H), t=t, kinetics='NX', 
                        vb=vb, Fb=Fb)
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3

    # Test boundary
    Fb = np.inf
    vb = 0.1
    PS = 0.001
    vi = 0.2
    H = 0.45
    J = dc.flux_tissue(ca, t=t, kinetics='2CX', 
                       H=H, vb=vb, vi=vi, Fb=np.inf, PS=PS)
    assert np.isinf(J[0,0,0])



# def test__conc_2cf():
#     # Compare against general ncomp solution
#     t = np.linspace(0, 20, 10)
#     J = np.ones((2,len(t)))
#     J[1,:] = 0 # No inlet in 2nd compartment
#     T = [6,12]
#     E = 0.2
#     Fp = 1/60 # mL/sec/mL
#     ca = J[0,:]/Fp
#     PS = Fp*E/(1-E)
#     Te = T[1]
#     vp = T[0]*(Fp+PS)
#     C0 = dc.conc_tissue(ca, vp, Fp, PS, Te, t=t, sum=False, kinetics='2CF') 
#     Emat = [
#         [1-E, 0],
#         [E  , 1]]
#     C = dc.conc_ncomp(J, T, Emat, t)
#     assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-2
#     Cs = dc.conc_tissue(ca, vp, Fp, PS, Te, t=t, sum=True, kinetics='2CF')
#     C0 = np.sum(C,axis=0)
#     assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
#     C = dc.conc_tissue(ca, vp, Fp, 0, Te, t=t, sum=False, kinetics='2CF')
#     C0 = dc.conc_comp(J[0,:], vp/Fp, t)
#     assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-12
#     C = dc.conc_tissue(ca, 0, Fp, PS, Te, t=t, sum=False, kinetics='2CF')
#     assert np.linalg.norm(C[0,:]) == 0
#     Cs = dc.conc_tissue(ca, vp, 0, 0, Te, t=t, sum=True, kinetics='2CF')
#     assert np.linalg.norm(Cs) == 0
#     Cs = dc.conc_tissue(ca, vp, 0, 0, Te, t=t, sum=False, kinetics='2CF')
#     assert np.linalg.norm(Cs) == 0


# def test__flux_2cf():
#     # Compare against general ncomp solution
#     t = np.linspace(0, 20, 10)
#     J = np.ones((2,len(t)))
#     J[1,:] = 0 # No inlet in 2nd compartment
#     T = [6,12]
#     E = 0.2
#     Fp = 1/60 # mL/sec/mL
#     ca = J[0,:]/Fp
#     PS = Fp*E/(1-E)
#     Te = T[1]
#     vp = T[0]*(Fp+PS)
#     Jo0 = dc.flux_tissue(ca, vp, Fp, PS, Te, t=t, kinetics='2CF') 
#     Emat = [
#         [1-E, 0],
#         [E  , 1]]
#     Jo = dc.flux_ncomp(J, T, Emat, t)
#     assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-3
#     Jo0 = dc.flux_tissue(ca, vp, 0, 0, Te, t=t, kinetics='2CF') 
#     assert np.linalg.norm(Jo0) == 0


def test_relax_tissue():

    t = np.arange(0, 300, 1.5)
    ca = dc.aif_parker(t, BAT=20)
    R10, r1 = 1/dc.T1(), dc.relaxivity() 
    H = 0.45

    # Test WV limit - exact
    p = {'H':H, 'vb':0.0, 'vi':0.3, 'Fb':0.01, 'PS':0.005}
    R1_0, _, _ = dc.relax_tissue(ca*(1-H), R10, r1, t=t, 
                                 kinetics='2CX', water_exchange='RR', **p)
    p = {'H':H, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005)}
    R1_1, _, _ = dc.relax_tissue(ca*(1-H), R10, r1, t=t, 
                                 kinetics='WV', water_exchange='RR', **p)
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-9

    # Test WV limit - approx
    p = {'H':H, 'vb':0.5*1e-3, 'vi':0.3, 'Fb':0.01, 'PS':0.005}
    R1_0, _, _ = dc.relax_tissue(ca*(1-H), R10, r1, t=t, 
                                 kinetics='2CX', water_exchange='RR', **p)
    p = {'H':H, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005)}
    R1_1, _, _ = dc.relax_tissue(ca*(1-H), R10, r1, t=t, 
                                 kinetics='WV', water_exchange='RR', **p)
    assert np.linalg.norm(R1_0[1:,:]-R1_1)< 1e-3*np.linalg.norm(R1_0[1:,:])

    # Test HF limit - exact
    p = {'H':H, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005}
    R1_0, _, _ = dc.relax_tissue(ca, R10, r1, t=t, 
                                 kinetics='2CX', water_exchange='RR', **p)
    p = {'H':H, 'vb':0.05, 'vi':0.3, 'PS':0.005}
    R1_1, _, _ = dc.relax_tissue(ca, R10, r1, t=t, 
                                 kinetics='HF', water_exchange='RR', **p)
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    # Test HF limit - approx
    p = {'H':0.4, 'vb':0.05, 'vi':0.3, 'Fb':1000, 'PS':0.005}
    R1_0, _, _ = dc.relax_tissue(ca, R10, r1, t=t, 
                                 kinetics='2CX', water_exchange='RR', **p)
    p = {'H':0.4, 'vb':0.05, 'vi':0.3, 'PS':0.005}
    R1_1, _, _ = dc.relax_tissue(ca, R10, r1, t=t, 
                                 kinetics='HF', water_exchange='RR', **p)
    assert np.linalg.norm(R1_0-R1_1) < 1e-3*np.linalg.norm(R1_0)



def test_conc_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        dc.conc_tissue(ca, t=t, kinetics='blabla', Fb=Fb)
    except:
        assert True
    else:
        assert False
    

def test_flux_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        dc.flux_tissue(ca, t=t, kinetics='blabla', Fb=Fb)
    except:
        assert True
    else:
        assert False


def test_params_tissue():

    try:
        dc.params_tissue('XX', 'RR')
    except:
        assert True
    else:
        assert False

    try:
        dc.params_tissue('2CX', 'XX')
    except:
        assert True
    else:
        assert False

    for kinetics in ['2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U']:
        for wxe in ['F','N', 'R']:
            for wxc in ['F','N', 'R']:
                dc.params_tissue(kinetics, wxe+wxc)


def test_signal_tissue():
    nt = 10
    R10 = 1
    r1 = 0.005
    ca = np.ones(nt)
    try:
        dc.signal_tissue(ca, R10, r1, dt=1.0, kinetics='2CX', 
                         water_exchange='FF')
    except:
        assert True
    else:
        assert False
    pars = {
        'H':0.45, 
        'vb': 0.1,
        'vi': 0.3,
        'Fb': 0.5,
        'PS': 0.005,
    }
    seq = {
        'model': 'SS', 
        'S0':1, 
        'FA':15, 
        'TR': 0.001, 
        'B1corr':1,
    }
    S = dc.signal_tissue(ca, R10, r1, dt=1.0, kinetics='2CX', 
                         water_exchange='FF', sequence=seq, **pars)
    assert 0.007 < S[0] < 0.008
    inflow = {'R10a': 0.7, 'B1corr_a':1}
    try:
        dc.signal_tissue(
            ca, R10, r1, dt=1.0, kinetics='2CU', water_exchange='FF', 
            sequence=seq, inflow=inflow, **pars)
    except:
        assert True
    else:
        assert False
    S = dc.signal_tissue(
        ca, R10, r1, dt=1.0, kinetics='2CX', water_exchange='FF', 
        sequence=seq, inflow=inflow, **pars)
    assert 0.007 < S[0] < 0.008
    
    seq = {
        'model': 'SR', 
        'S0':1, 
        'FA':15, 
        'TR': 0.001, 
        'TC': 0.2,
        'TP': 0.1,
        'B1corr':1,
    }
    S = dc.signal_tissue(ca, R10, r1, dt=1.0, kinetics='2CX', 
                         water_exchange='FF', sequence=seq, **pars)
    assert 0.007 < S[0] < 0.008
    inflow = {'R10a': 0.7, 'B1corr_a':1}
    try:
        dc.signal_tissue(
            ca, R10, r1, dt=1.0, kinetics='2CU', water_exchange='FF', 
            sequence=seq, inflow=inflow, **pars)
    except:
        assert True
    else:
        assert False
    S = dc.signal_tissue(
        ca, R10, r1, dt=1.0, kinetics='2CX', water_exchange='FF', 
        sequence=seq, inflow=inflow, **pars)
    assert 0.007 < S[0] < 0.008


if __name__ == "__main__":

    test_signal_tissue()
    test_params_tissue()

    test__conc_u()
    test__flux_u()

    test__conc_fx()
    test__flux_fx()

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

    test_conc_tissue()
    test_flux_tissue()

    test_relax_tissue()


    print('All pk_tissue tests passing!')