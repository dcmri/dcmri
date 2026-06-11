import numpy as np
from scipy.integrate import cumulative_trapezoid

import dcmri as dc
from dcmri.kinetics.tissue import dpars_tissue


def test_conc_tissue():

    p = {
        'vb': 0.2,
        've': 0.2,
    }
    dpars_tissue(p)
    p = {
        'vb': 0.2,
        'vi': 0.2,
    }
    dpars_tissue(p)

    # conc_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue_nxp(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = dc.conc_tissue_nxp(ca, t=t, vb=vb, Fb=Fb)
    assert C[0,0] == 0


    # conc_nx

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue_nx(ca, t=t,  vb=vb, Fb=Fb)
    assert C[0,0] == 0
    Fb = 0
    C = dc.conc_tissue_nx(ca, t=t, vb=vb, Fb=Fb) 
    assert C[0,0] == 0

    # conc_u

    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue_u(ca, t=t, Fb=Fb)
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
    C = dc.conc_tissue_fx(ca, t=t, H=H, ve=ve, Fb=Fb) 
    Fp = Fb*(1-H)
    C0 = Fp*dc.biexpconv(Ta, ve/Fp, t)*ve/Fp/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue_fx(ca, t=t, H=H, ve=ve, Fb=0)
    assert np.linalg.norm(C) == 0

    # conc_wv

    n = 10
    Ta = 10
    Ktrans = 2
    vi = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue_wv(ca, t=t, H=H,  vi=vi, Ktrans=Ktrans) 
    C0 = Ktrans*dc.biexpconv(Ta, vi/Ktrans, t)*vi/Ktrans/(1-H)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue_wv(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue_wv(ca, t=t, H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(C) == 0

    # conc_hfu

    n = 10
    Ta = 10
    vb = 0.3
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    C = dc.conc_tissue_hfu(ca, t=t, H=H, vb=vb, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.conc_trap(ca/(1-H), t)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue_hfu(ca, t=t, H=H, vb=vb, PS=PS).sum(axis=0)
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
    C = dc.conc_tissue_hf(ca, t=t, H=H, vb=vb, vi=vi, PS=PS)
    C0 = vb*ca
    C1 = PS*dc.biexpconv(Ta, vi/PS, t)*vi/PS/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1)/np.linalg.norm(C1) < 0.01
    C = dc.conc_tissue_hf(ca, t=t, H=H, vb=vb, vi=vi, PS=PS).sum(axis=0)
    C0 = C0+C1
    assert np.linalg.norm(C-C0)/np.linalg.norm(C0) < 0.01
    C = dc.conc_tissue_hf(ca, t=t, H=H, vb=vb, vi=0, PS=PS)
    C0 = vb*ca
    C1 = np.zeros(len(ca))
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    assert np.linalg.norm(C[1,:]-C1) == 0
    C = dc.conc_tissue_hf(ca, t=t, H=H, vb=vb, vi=vi, PS=0)
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
    C = dc.conc_tissue_2cu(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    C0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)/(1-H)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue_2cu(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS).sum(axis=0)
    C0 = C0+C[1,:]
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue_2cu(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS).sum(axis=0)
    C0 = dc.conc_tissue_u(ca/(1-H), t=t, Fb=PS*Fp/(PS+Fp))
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue_2cu(ca, t=t, H=H, vb=vb, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(Cs)==0
    Cs = dc.conc_tissue_2cu(ca, t=t, H=H, vb=vb, Fb=np.inf, PS=1)
    Cn = dc.conc_tissue_hfu(ca, t=t, H=H, vb=vb, PS=1)
    assert np.linalg.norm(Cs-Cn)==0

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
    C0 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    C = dc.conc_ncomp(J, t, T=T, E=Emat)
    assert np.linalg.norm(C-C0)/np.linalg.norm(C) < 1e-3
    Cs = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C0 = np.sum(C,axis=0)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C0) < 0.01
    Cs = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0)
    assert np.linalg.norm(Cs) == 0
    Cs = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0).sum(axis=0)
    C0 = dc.conc_tissue_nx(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Cs-C0)/np.linalg.norm(C) < 1e-3
    C = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0)
    assert np.linalg.norm(C[0,:]-C0)/np.linalg.norm(C0) < 1e-3
    C = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=0, vi=0, Fb=0, PS=0).sum(axis=0)
    assert np.linalg.norm(C) == 0

    # Test boundaries (Fp=inf)
    Fb = 0.01
    PS = 0.001
    vb = 0.1
    vi = 0.2
    C0 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.05
    C1 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 0.1
    C2 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = 10.0
    C3 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    Fb = np.inf
    C4 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
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
    C0 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    assert np.linalg.norm(C0) == 0

    # Test boundaries (PS=0)
    Fb = 0.01
    PS = 0
    vb = 0.1
    vi = 0.2
    C0 = dc.conc_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS).sum(axis=0)
    C1 = dc.conc_tissue_nx(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(C0-C1) < 1e-9




def test_flux_tissue():

    # flux_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_nxp(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = dc.flux_tissue_nxp(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0

    # flux_nx

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_nx(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = dc.flux_tissue_nx(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0

    # flux_u

    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_u(ca, Fb=Fb)
    J0 = np.zeros(len(t))
    assert np.linalg.norm(J-J0) < 0.01

    # flux_fx

    n = 10
    Ta = 10
    Fb = 2
    ve = 0.1
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_fx(ca, t=t, H=H, ve=ve, Fb=Fb)
    Fp = Fb*(1-H)
    J0 = Fp*dc.biexpconv(Ta, ve/Fp, t)/(1-H)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue_fx(ca, t=t, H=H, ve=ve, Fb=0)
    assert np.linalg.norm(J) == 0

    # flux_wv

    n = 10
    Ta = 10
    Ktrans = 2
    kep = 5
    H = 0.45
    vi = Ktrans/kep
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_wv(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    J0 = Ktrans*dc.biexpconv(Ta, 1/kep, t)/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue_wv(ca, t=t, H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(J[0,1,:]) == 0

    # flux_hfu

    n = 10
    Ta = 10
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_hfu(ca, H=H, PS=PS)
    J0 = PS*ca/(1-H)
    assert np.linalg.norm(J[1,0,:]-J0)/np.linalg.norm(J0) < 0.01

    # flux_hf

    n = 10
    Ta = 10
    PS = 2
    kep = 5
    vi = PS/kep
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = dc.flux_tissue_hf(ca, t=t, H=H, vi=vi, PS=PS)
    J0 = kep*PS*dc.biexpconv(Ta, 1/kep, t)/kep/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue_hf(ca, t=t, H=H, vi=vi, PS=0)
    assert 0==np.linalg.norm(J[0,1,:])

    # flux_2cu

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
    J = dc.flux_tissue_2cu(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = dc.flux_tissue_2cu(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS)
    assert np.linalg.norm(J[0,0,:]-Fb*ca)/np.linalg.norm(Fb*ca) < 0.01
    
    # flux_2cx

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
    Jo0 = dc.flux_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS) 
    Jo = dc.flux_ncomp(J, t, T=T, E=Emat)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-1
    Jo0 = dc.flux_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0) 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = dc.flux_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    Jo = dc.flux_tissue_nx(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3

    Jo = dc.flux_tissue_2cx(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    assert Jo[0,0,0] == 0

    # Test boundary
    Fb = np.inf
    vb = 0.1
    PS = 0.001
    vi = 0.2
    H = 0.45
    J = dc.flux_tissue_2cx(ca, t=t, H=H, vb=vb, vi=vi, Fb=np.inf, PS=PS)
    assert np.isinf(J[0,0,0])




if __name__ == '__main__':
    test_conc_tissue()
    test_flux_tissue()

    print('All kinetics tests passed!!')