import numpy as np
from scipy.integrate import cumulative_trapezoid

import numpy as np
import dcmri as dc
import dcmri.kinetics.lib as pk


from dcmri.kinetics import (
    ConcKidney,
    ConcLiver,
    ConcTissueX,
)



def test_conc_kidney():
    dt = 1.5
    t = dt*np.arange(20)
    ca = np.ones(20)
    p = {'Fp': 0.01, 'vp': 0.2, 'Ft': 0.005, 'Tt': 120}
    p['Tp'] = p['vp'] / (p['Fp'] + p['Ft'])
    C = ConcKidney('2CF', **p)(ca, dt=dt).sum(axis=0)
    assert round(C[10], 1) == 0.1
    C = ConcKidney('2CF', **p)(ca, dt=dt)
    assert round(C[1,10], 2) == 0.02
    C = ConcKidney('HF', **p)(ca, dt=dt).sum(axis=0)
    assert round(C[10], 1) == 0.3
    C = ConcKidney('HF', **p)(ca, dt=dt)
    assert round(C[1,10], 2) == 0.07
    h = [1,2,3,4,3,2,1]
    C = ConcKidney('FN', **p)(ca, dt=dt, ht=h).sum(axis=0)
    assert round(C[10], 1) == 0.2
    C = ConcKidney('FN', **p)(ca, dt=dt, ht=h)
    assert round(C[1,10], 2) == 0.02

    try:
        ConcKidney('X')(ca)
    except:
        assert True
    else:
        assert False


def test_conc_liver():

    # EC

    tmax = 60
    nt = 10
    Ta = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Ta)/Ta

    p = {'ve': 0.1, 
         'Fp': 0.01, 
         'fa': 1.0, 
         'T_a': 0, 
         'Tg': 0,
        }
    C0 = ConcLiver('1I-EC', **p)(ca, t)

    p = {'ve': 0.1, 
         'Te': 0.1 / 0.01, 
         'De': 1.0,
        }
    C1 = ConcLiver('1I-EC-D', **p)(ca, t)

    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-9


    p = {'ve': 0.1, 
         'Fp': 1000, 
         'fa': 1.0, 
         'T_a': 0, 
         'Tg': 0,
        }
    C0 = ConcLiver('1I-EC', **p)(ca, t)

    p = {'ve': 0.1, 
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = ConcLiver('2I-EC-HF', **p)((ca, ca), t, **p)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3

    p = {'ve': 0.1, 
         'Fp': 1000,
         'fa': 1.0, 
         'T_a': 0, 
        }
    C1 = ConcLiver('2I-EC', **p)((ca, ca), t)

    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-3


    # IC

    tmax = 60
    nt = 30
    Taif = 20
    t = np.linspace(0, tmax, nt)
    ca = np.exp(-t/Taif)/Taif

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = ConcLiver('1I-IC-HF', **p)(ca, t)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
        }
    C1 = ConcLiver('1I-IC-HF', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = ConcLiver('1I-IC-HF', 'E', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
        }
    C1 = ConcLiver('1I-IC-HF', 'UE', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = ConcLiver('1I-IC-HFD', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = ConcLiver('1I-IC-HFD', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = ConcLiver('1I-IC-HFD', 'E', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'Tg': 0,
         'Dg': 1,
        }
    C1 = ConcLiver('1I-IC-HFD', 'UE', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 1500,
         'Tg': 10,
         'Dg': 0.5,
        }
    C0 = ConcLiver('1I-IC-HFD', **p)(ca, t)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = ConcLiver('1I-IC-HFDU', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Tg': 10,
         'Dg': 0.5,
        }
    C1 = ConcLiver('1I-IC-HFDU', 'U', **p)(ca, t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
        }
    C0 = ConcLiver('1I-IC-HF', **p)(ca, t)

    C = ConcLiver('1I-IC-HF', **p)(ca, t)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-HF', **p)((ca, ca), t)

    p = {'ve': 0.1, 
         'khe_i': 0.005, 
         'khe_f': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-HF', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-3

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-HF', 'E', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe_i': 0.005,
         'khe_f': 0.005,
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-HF', 'UE', **p)((ca, ca), t)
    assert np.linalg.norm(C0-C1) / np.linalg.norm(C0) < 1e-1

    p = {'ve': 0.1, 
         'khe': 0.005, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C0 = ConcLiver('2I-IC-HF', **p)((ca, ca), t)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 0.1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,  
         'Th': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001,  
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC', 'E', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,  
         'Th_i': 15,
         'Th_f': 15,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC', 'UE', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'Th': 1500,
         'T_a': 0,
         'fa': 1,
        }
    C0 = ConcLiver('2I-IC', **p)((ca, ca), t)

    C = ConcLiver('2I-IC', **p)((ca, ca), t)
    assert np.array_equal(C, C0)

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E': 0.001, 
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1

    p = {'ve': 0.1 / (1 - 0.001), 
         'Fp': 5, 
         'E_i': 0.001, 
         'E_f': 0.001,
         'T_a': 0,
         'fa': 1,
        }
    C1 = ConcLiver('2I-IC-U', 'U', **p)((ca, ca), t)
    assert np.linalg.norm(C0[0,1:]-C1[0,1:]) / np.linalg.norm(C0[0,1:]) < 1e-1


    # Test exceptions
    
    try:
        C1 = ConcLiver('XX-YY', 'U', **p)((ca, ca), t)
    except:
        pass
    else:
        assert False


def test_conc_tissue():

    # conc_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = ConcTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0


    # conc_nx

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = ConcTissueX(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb) 
    assert C[0,0] == 0

    # conc_u

    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='U')(ca, t=t, Fb=Fb)
    C0 = Fb*cumulative_trapezoid(ca, t, initial=0)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

    # conc_fx

    n = 10
    Ta = 10
    Fb = 2
    ve = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=Fb) 
    Fp = Fb*(1-H)
    C0 = Fp*dc.convolution.biexpconv(Ta, ve/Fp, t)*ve/Fp/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = ConcTissueX(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=0)
    assert np.linalg.norm(C) == 0

    # conc_wv

    n = 10
    Ta = 10
    Ktrans = 2
    vi = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='WV')(ca, t=t, H=H,  vi=vi, Ktrans=Ktrans) 
    C0 = Ktrans*dc.convolution.biexpconv(Ta, vi/Ktrans, t)*vi/Ktrans/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = ConcTissueX(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = ConcTissueX(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(C) == 0

    # conc_hfu

    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='HFU')(ca, t=t, H=H, vb=vb, PS=PS)
    C0 = vb*ca
    C1 = PS*pk.conc_trap(ca/(1-H), t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = ConcTissueX(kinetics='HFU')(ca, t=t, H=H, vb=vb, PS=PS).sum(axis=0)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01

    # conc_hf

    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    vi = 0.2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = ConcTissueX(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.convolution.biexpconv(Ta, vi/PS, t)*vi/PS/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = ConcTissueX(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=PS).sum(axis=0)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = ConcTissueX(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=0, PS=PS)
    C0 = vb*ca
    C1 = np.zeros(len(ca))
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1) == 0
    C = ConcTissueX(kinetics='HF')(ca, t=t, H=H, vb=vb, vi=vi, PS=0)
    assert 0==np.linalg.norm(C[1,:])

    # conc_2cu

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
    C = ConcTissueX(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.convolution.biexpconv(Ta, Tp, t)/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = ConcTissueX(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS).sum(axis=0)
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = ConcTissueX(kinetics='2CU')(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS).sum(axis=0)
    C0 = ConcTissueX(kinetics='U')(ca/(1-H), t=t, Fb=PS*Fp/(PS+Fp))
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = ConcTissueX(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(Cs)==0

    # conc_2cx

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
    C0 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C = pk.conc_ncomp(J, T, Emat, t)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(Cs) == 0
    Cs = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0)
    assert np.linalg.norm(Cs) == 0
    Cs = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0).sum(axis=0)
    C0 = ConcTissueX(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    C = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=0, vi=0, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(C) == 0

    # Test boundaries (Fp=inf)
    Fb = 0.01
    PS = 0.001
    vb = 0.1
    vi = 0.2
    C0 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.05
    C1 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.1
    C2 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 10.0
    C3 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = np.inf
    C4 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
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
    C0 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    assert np.linalg.norm(C0) == 0

    # Test boundaries (PS=0)
    Fb = 0.01
    PS = 0
    vb = 0.1
    vi = 0.2
    C0 = ConcTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C1 = ConcTissueX(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(C0-C1) < 1e-9

    # Exceptions

    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    try:
        ConcTissueX(kinetics='blabla')(ca, t=t, Fb=Fb)
    except:
        assert True
    else:
        assert False

    # Run all cases
    for kin in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX', 'NXP']:
        p = ConcTissueX(kin).params()
        p = {key:0.01 for key in p}
        ConcTissueX(kin)(ca, **p)




if __name__ == '__main__':

    test_conc_kidney()
    test_conc_liver()
    test_conc_tissue()

    print('All kinetics tests passed!!')