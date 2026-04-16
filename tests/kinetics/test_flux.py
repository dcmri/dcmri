import numpy as np

import dcmri as dc
import dcmri.kinetics.lib as pk


from dcmri.kinetics import (
    FluxTissueX,
)



def test_flux_tissue():

    # flux_nxp

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = FluxTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = FluxTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    J = FluxTissueX(kinetics='NXP')(ca, t=t, vb=vb, Fb=Fb, T_a=0)
    assert J[0] == 0

    # flux_nx

    n = 10
    Ta = 10
    Fb = 2
    vb = 0.1
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = FluxTissueX(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    Fb = 0
    J = FluxTissueX(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb)
    assert J[0] == 0
    J = FluxTissueX(kinetics='NX')(ca, t=t, vb=vb, Fb=Fb, T_a=0)
    assert J[0] == 0

    # flux_u

    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = FluxTissueX(kinetics='U')(ca, t=t, Fb=Fb)
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
    J = FluxTissueX(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=Fb)
    Fp = Fb*(1-H)
    J0 = Fp*dc.convolution.biexpconv(Ta, ve/Fp, t)/(1-H)
    assert np.linalg.norm(J-J0)/np.linalg.norm(J0) < 0.01
    J = FluxTissueX(kinetics='FX')(ca, t=t, H=H, ve=ve, Fb=0)
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
    J = FluxTissueX(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=Ktrans)
    J0 = Ktrans*dc.convolution.biexpconv(Ta, 1/kep, t)/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = FluxTissueX(kinetics='WV')(ca, t=t, H=H, vi=vi, Ktrans=0)
    assert np.linalg.norm(J[0,1,:]) == 0

    # flux_hfu

    n = 10
    Ta = 10
    PS = 2
    H = 0.45
    t = np.linspace(0, 20, n)
    ca = np.exp(-t/Ta)/Ta
    J = FluxTissueX(kinetics='HFU')(ca, H=H, PS=PS)
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
    J = FluxTissueX(kinetics='HF')(ca, t=t, H=H, vi=vi, PS=PS)
    J0 = kep*PS*dc.convolution.biexpconv(Ta, 1/kep, t)/kep/(1-H)
    assert np.linalg.norm(J[0,1,:]-J0)/np.linalg.norm(J0) < 0.01
    J = FluxTissueX(kinetics='HF')(ca, t=t, H=H, vi=vi, PS=0)
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
    J = FluxTissueX(kinetics='2CU')(ca, t=t, H=H, vb=vb, Fb=Fb, PS=PS)
    Tp = vp/(Fp+PS)
    J0 = Tp*Fp*dc.convolution.biexpconv(Ta, Tp, t)*Fp/vp
    assert np.linalg.norm(J[0,0,:]-J0)/np.linalg.norm(J0) < 0.01
    J = FluxTissueX(kinetics='2CU')(ca, t=t, H=H, vb=0, Fb=Fb, PS=PS)
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
    Jo0 = FluxTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS) 
    Jo = pk.flux_ncomp(J, T, Emat, t)
    assert np.linalg.norm(Jo-Jo0)/np.linalg.norm(Jo) < 1e-1
    Jo0 = FluxTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=0, PS=0) 
    assert np.linalg.norm(Jo0) == 0
    Jo0 = FluxTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    Jo = FluxTissueX(kinetics='NX')(ca*(1-H), t=t, vb=vb, Fb=Fb)
    assert np.linalg.norm(Jo-Jo0[0,0,:])/np.linalg.norm(Jo) < 1e-3

    Jo = FluxTissueX(kinetics='2CX')(ca*(1-H), t=t, H=H, vb=vb, vi=vi, Fb=Fb, PS=0) 
    assert Jo[0,0,0] == 0

    # Test boundary
    Fb = np.inf
    vb = 0.1
    PS = 0.001
    vi = 0.2
    H = 0.45
    J = FluxTissueX(kinetics='2CX')(ca, t=t, H=H, vb=vb, vi=vi, Fb=np.inf, PS=PS)
    assert np.isinf(J[0,0,0])

    # flux_tissue

    # Only need to test exceptions
    n = 10
    Ta = 10
    Fb = 2
    t = np.linspace(0, 20, n)
 
    ca = np.exp(-t/Ta)/Ta
    try:
        FluxTissueX(kinetics='blabla')(ca, t=t, Fb=Fb)
    except:
        assert True
    else:
        assert False


if __name__ == '__main__':

    test_flux_tissue()

    print('All kinetics tests passed!!')