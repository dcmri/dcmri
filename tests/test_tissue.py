from scipy.integrate import cumulative_trapezoid

import numpy as np
import dcmri as dc

from dcmri import tissue
import dcmri.lexicon_utils as lexicon

def test__conc_nxp():
    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = tissue.Conc(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = tissue.Conc(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0

def test__flux_nxp():
    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = tissue.Flux(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = tissue.Flux(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    J = tissue.Flux(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb, T_a=0)
    assert J[0] == 0



def test__conc_nx():
    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = tissue.Conc(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = tissue.Conc(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb) 
    assert C[0,0] == 0

def test__flux_nx():
    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = tissue.Flux(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = tissue.Flux(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    J = tissue.Flux(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb, T_a=0)
    assert J[0] == 0


def test__conc_u():
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = tissue.Conc(kinetics='U')(ca, t=t, Fb=Fb)
    C0 = Fb*cumulative_trapezoid(ca, t, initial=0)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_u():
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = tissue.Flux(kinetics='U')(ca, t=t, Fb=Fb)
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
    C = tissue.Conc(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=Fb) 
    Fp = Fb*(1-H)
    C0 = Fp*dc.biexpconv(Ta, ve/Fp, t)*ve/Fp/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = tissue.Conc(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=0)
    assert np.linalg.norm(C) == 0

def test__flux_fx():
    n = 10
    Ta = 10
    Fb = 2
    ve = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = tissue.Flux(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=Fb)
    Fp = Fb*(1-H)
    J0 = Fp*dc.biexpconv(Ta, ve/Fp, t)/(1-H)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = tissue.Flux(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=0)
    assert np.linalg.norm(J) == 0


def test__conc_wv():
    n = 10
    Ta = 10
    Ktrans = 2
    vi = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = tissue.Conc(kinetics='WV')(ca, t=t, H=H,  vi=vi, Ktrans=Ktrans) 
    C0 = Ktrans*dc.biexpconv(Ta, vi/Ktrans, t)*vi/Ktrans/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = tissue.Conc(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = tissue.Conc(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=0)
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
    J = tissue.Flux(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    J0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = tissue.Flux(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(J[0,1,:]) == 0
    

def test__conc_hfu():
    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = tissue.Conc(kinetics='HFU')(ca, t=t, H=H, vb=vb, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.conc_trap(ca/(1-H), t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = tissue.Conc(kinetics='HFU')(ca, t=t, H=H, vb=vb, PS=PS).sum(axis=0)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

def test__flux_hfu():
    n = 10
    Ta = 10
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = tissue.Flux(kinetics='HFU')(ca, H=H, PS=PS)
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
    C = tissue.Conc(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.biexpconv(Ta, vi/PS, t)*vi/PS/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = tissue.Conc(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=PS).sum(axis=0)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = tissue.Conc(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=0, PS=PS)
    C0 = vb*ca
    C1 = np.zeros(len(ca))
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1) == 0
    C = tissue.Conc(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=0)
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
    J = tissue.Flux(kinetics='HF')(ca, t=t, H=H, vi=vi, PS=PS)
    J0 = kep*PS*dc.biexpconv(Ta, 1/kep, t)/kep/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = tissue.Flux(kinetics='HF')(ca, t=t, H=H, vi=vi, PS=0)
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
    C = tissue.Conc(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = tissue.Conc(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS).sum(axis=0)
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = tissue.Conc(kinetics='2CU')(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS).sum(axis=0)
    C0 = tissue.Conc(kinetics='U')(ca/(1-H), t=t, Fb=PS*Fp/(PS+Fp))
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = tissue.Conc(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=0, PS=0).sum(axis=0)
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
    J = tissue.Flux(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = tissue.Flux(kinetics='2CU')(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS)
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
    C0 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C = dc.conc_ncomp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(Cs) == 0
    Cs = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0)
    assert np.linalg.norm(Cs) == 0
    Cs = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0).sum(axis=0)
    C0 = tissue.Conc(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    C = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=0, vi=0, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(C) == 0

    # Test boundaries (Fp=inf)
    Fb = 0.01
    PS = 0.001
    vb = 0.1
    vi = 0.2
    C0 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.05
    C1 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.1
    C2 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 10.0
    C3 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = np.inf
    C4 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
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
    C0 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    assert np.linalg.norm(C0) == 0

    # Test boundaries (PS=0)
    Fb = 0.01
    PS = 0
    vb = 0.1
    vi = 0.2
    C0 = tissue.Conc(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C1 = tissue.Conc(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
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
    Jo0 = tissue.Flux(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS) 
    Jo = dc.flux_ncomp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-1
    Jo0 = tissue.Flux(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0) 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = tissue.Flux(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    Jo = tissue.Flux(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3

    Jo = tissue.Flux(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    assert Jo[0,0,0] == 0

    # Test boundary
    Fb = np.inf
    vb = 0.1
    PS = 0.001
    vi = 0.2
    H = 0.45
    J = tissue.Flux(kinetics='2CX')(ca, t=t, H=H, vb=vb, vi=vi, Fb=np.inf, PS=PS)
    assert np.isinf(J[0,0,0])

def test_flux_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
 
    ca = np.exp(-t/Ta)/Ta
    try:
        tissue.Flux(kinetics='blabla')(ca, t=t, Fb=Fb)
    except:
        assert True
    else:
        assert False




def test_conc_tissue():
    # Only need to test exceptions
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        tissue.Conc(kinetics='blabla')(ca, t=t, Fb=Fb)
    except:
        assert True
    else:
        assert False

    # Run all cases
    for kin in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX', 'NXP']:
        p = tissue.Conc(kin).params()
        p = {key:0.01 for key in p}
        tissue.Conc(kin)(ca, **p)

def test_relax_tissue():

    t = np.arange(0, 300, 1.5)
    ca = dc.aif_parker(t, BAT=20)
    H = 0.45

    # Test WV limit - exact
    p0 = {'H':H, 'T_a':0, 'vb':0.0, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='WV')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-9
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-9

    # Test WV limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.5*1e-3, 'vi':0.3, 'Fb':0.01, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p = {'H':H, 'T_a':0, 'vi':0.3, 'Ktrans':0.01*(1-H)*0.005/(0.01*(1-H)+0.005), 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='WV')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='WV', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0[1:,:]-C1) < 1e-3 * np.linalg.norm(C0[1:,:])
    assert np.linalg.norm(R1_0[1:,:]-R1_1) < 1e-3 * np.linalg.norm(R1_0[1:,:])

    # Test HF limit - exact

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='HF')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':np.inf, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='2CU')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='2CU', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='HFU')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='HFU', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-9
    assert np.linalg.norm(R1_0-R1_1) < 1e-9

    # Test HF limit - approx

    p0 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'Fb':10, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='2CX')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='2CX', water_exchange='RR')(C0, **p0)

    p1 = {'H':H, 'T_a':0, 'vb':0.05, 'vi':0.3, 'PS':0.005, 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='HF')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='HF', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-3 * np.linalg.norm(C0)
    assert np.linalg.norm(R1_0-R1_1) < 1e-3 * np.linalg.norm(R1_0)

    # Cover FX limit - ve = 0

    p0 = {'H':H, 've':1e-3, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics='FX')(ca, t, **p0)
    R1_0 = tissue.Relax(kinetics='FX', water_exchange='RR')(C1, **p0)

    p1 = {'H':H, 've':0, 'Fb':0.01, 'vb':0.0, 'R10': 1, 'r1': 0.005}
    C1 = tissue.Conc(kinetics='FX')(ca, t, **p1)
    R1_1 = tissue.Relax(kinetics='FX', water_exchange='RR')(C1, **p1)

    assert np.linalg.norm(C0-C1) < 1e-3
    assert np.linalg.norm(R1_0-R1_1) < 1e-3



def test_magn_tissue():
    nt = 10
    ca = np.ones(nt)
    kinetics='2CX'

    p0 = {'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'R10_a': 1, 'R10': 1, 'r1': 0.005}
    C0 = tissue.Conc(kinetics=kinetics)(ca, **p0)
    R1a = dc.relax(ca, p0['R10_a'], p0['r1'])

    R1 = tissue.Relax(kinetics, 'RR')(C0, **p0)
    Mz = tissue.Mz(kinetics, 'RR', 'SS')(R1, R1a, **p0)
    assert 0.01 < Mz[0,0] < 0.02

    R1 = tissue.Relax(kinetics, 'FF')(C0, **p0)
    Mz = tissue.Mz(kinetics, 'FF', 'SS', 'None')(R1, R1a, **p0)
    assert 0.1 < Mz[0,0] < 0.2

    R1 = tissue.Relax(kinetics, 'FR')(C0, **p0)
    Mz = tissue.Mz(kinetics, 'FR', 'SS', 'None')(R1, R1a, **p0)
    assert 0.09 < Mz[0,0] < 0.11
    
    try:
        Mz = tissue.Mz(kinetics, 'FR', 'XX', 'None')
    except:
        pass
    else:
        assert False


def test_signal_tissue():
    nt = 10
    ca = np.ones(nt)

    p0 = {'H':0.45, 'T_a':0, 'vb':0.1, 'vi':0.3, 'Fb':0.5, 'PS':0.005, 'TR': 0.005, 'FA':15, 'PSe': 0.03, 'PSc': 0.03, 'S0':10, 'R10':1,'R10_a':1, 'r1':0.005}
    S = tissue.Signal('2CX', 'RR', 'SS', 'SS')(ca, **p0)
    assert 0.3 < S[0] < 0.4


def test_coverage():
    nt = 10
    ca = np.ones(nt)

    # Run for coverage
    for kin in ['HF', 'U', 'FX', 'NX', 'NXP', 'WV', 'HFU', '2CU', '2CX']:
        for wex in ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']:
            for seq in ['SS', 'SR', 'IR', 'free', 'None', 'SPGR', 'SSI']:
                for iseq in ['SS', 'SR', 'IR', 'free', 'None', 'SPGR', 'SSI']:
    # for kin in ['2CX']:
    #     for wex in ['FN']:
    #         for seq in ['SR']:
    #             for iseq in ['SS']:
                    signal = tissue.Signal(kin, wex, seq, iseq)
                    p = signal.params()
                    p = lexicon.init(p)
                    p = tissue.derive_params(p)
                    S = signal(ca, **p)


def test_exceptions():
    kin, wex, seq, iseq = '2CX', 'RR', 'SS', 'SS'
    nt = 10
    ca = np.ones(nt)

    try:
        p = tissue.Signal('XXX', wex, seq, iseq).params()
    except:
        pass
    else:
        assert False
    try:
        p = tissue.Signal(kin, 'SSS', seq, iseq).params()
    except:
        pass
    else:
        assert False

    p = tissue.Signal(kin, wex, seq, iseq).params()
    p = lexicon.init(p)
    p = tissue.derive_params(p)

    try:
        J0 = tissue.Flux(kinetics=kin)(ca)
    except:
        pass
    else:
        assert False

    try:
        J0 = tissue.Flux(kinetics='XXX')(ca, *p)
    except:
        pass
    else:
        assert False

    try:
        C0 = tissue.Conc(kinetics=kin)(ca)
    except:
        pass
    else:
        assert False

    C0 = tissue.Conc(kinetics=kin)(ca, **p)

    try:
        R1 = tissue.Relax('XXX', wex)(C0, **p)
    except:
        pass
    else:
        assert False

    try:
        R1 = tissue.Relax(kin, wex)(C0)
    except:
        pass
    else:
        assert False

    R1 = tissue.Relax(kin, wex)(C0, **p)
    R1a = dc.relax(ca, p['R10_a'], p['r1'])

    try:
        Mz = tissue.Mz('XXX', wex, seq, iseq)(R1, R1a, **p)
    except:
        pass
    else:
        assert False

    try:
        Mz = tissue.Mz(kin, 'XXX', seq, iseq)(R1, R1a, **p)
    except:
        pass
    else:
        assert False

    try:
        Mz = tissue.Mz(kin, wex, seq, 'XXX')(R1, R1a, **p)
    except:
        pass
    else:
        assert False

    try:
        Mz = tissue.Mz(kin, wex, seq, iseq)(R1, R1a)
    except:
        pass
    else:
        assert False


    Mz = tissue.Mz(kin, wex, seq, iseq)(R1, R1a, **p)

    try:
        S = tissue.Signal(kin, wex, seq, iseq)(ca)
    except:
        pass
    else:
        assert False

    S = tissue.Signal(kin, wex, seq, iseq)(ca, **p)

if __name__ == "__main__":

    test__conc_nx()
    test__flux_nx()

    test__conc_nxp()
    test__flux_nxp()

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
    test_magn_tissue()
    test_signal_tissue()

    test_coverage()
    test_exceptions()
    


    print('All tissue tests passing!')