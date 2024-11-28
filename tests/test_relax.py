import numpy as np
import dcmri as dc

import matplotlib.pyplot as plt


def test_Mz_free():

    R1 = 1
    TI = 0.1*np.arange(100)
    f = 0.5

    Mz = dc.Mz_free(R1, TI, n0=-1)
    Mz_e = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=f)
    Mz_i = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=-f)

    assert 21 < np.linalg.norm(Mz + Mz_e + Mz_i) < 22

    R1 = [1,2]
    v = [0.3, 0.7]
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]
    Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=[f, 0])

    assert 7 < np.linalg.norm(Mz) < 8

    TI = 0.5
    nt = 1000
    t = 0.1*np.arange(nt)
    R1 = np.stack((1-t/np.amax(t), np.ones(nt)))
    j = np.stack((f*np.ones(nt), np.zeros(nt)))
    Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)

    assert 5 < np.linalg.norm(Mz) < 6

    TI = 0.1*np.arange(10)
    Mzi = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)

    assert np.linalg.norm(Mzi[:,:,5]-Mz) == 0


def test_Mz_ss():

    # Steady state magnetization with different inflows

    R1 = 1
    FA = 12
    TR = 0.005
    f = 0.5
    v = 0.7

    # Check that inflow with steady-state magnetization 
    # is the same as no inflow
    m_c = dc.Mz_ss(R1, TR, FA, v)/v
    m_f = dc.Mz_ss(R1, TR, FA, v, Fw=f, j=f*m_c)/v
    assert np.abs(m_f-m_c) < 0.01*np.abs(m_c)

    # Repeat for a two-compartment system:
    R1 = [0.5, 1.5]
    v = [0.3, 0.6]
    PS = 0.1

    Fw = [[0, PS], [PS, 0]]
    M_c = dc.Mz_ss(R1, TR, FA, v, Fw)

    Fw = [[f, PS], [PS, 0]]
    m_c = M_c[0]/v[0]
    m_f = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f*m_c, 0])[0]/v[0]

    assert np.abs(m_c-m_f) < 0.01*np.abs(m_c)

    # Check fast-exchange limit
    v = [0.3, 0.5]
    M_c = dc.Mz_ss(R1, TR, FA, v, 1e9)
    M_c_fex = dc.Mz_ss(R1, TR, FA, v, np.inf)
    assert np.linalg.norm(M_c-M_c_fex) < 1e-3*np.linalg.norm(M_c_fex)

    # Check no-exchange limit
    v = [0.3, 0.5]
    M_c = dc.Mz_ss(R1, TR, FA, v, 1e-9)
    M_c_nex = dc.Mz_ss(R1, TR, FA, v, 0)
    assert np.linalg.norm(M_c-M_c_nex) < 1e-6*np.linalg.norm(M_c_nex)

    # Check exceptions
    Fw = [[0, np.inf], [np.inf, 0]]
    M_c_fex1 = dc.Mz_ss(R1, TR, FA, v, Fw)
    M_c_fex2 = dc.Mz_ss(R1, TR, FA, v, np.inf)
    assert np.linalg.norm(M_c_fex1-M_c_fex2) < 1e-9*np.linalg.norm(M_c_fex2)

    Fw = [[0, 0], [0, 0]]
    M_c_nex1 = dc.Mz_ss(R1, TR, FA, v, Fw)
    M_c_nex2 = dc.Mz_ss(R1, TR, FA, v, 0)
    assert np.linalg.norm(M_c_nex1-M_c_nex2) < 1e-9*np.linalg.norm(M_c_nex2)

    try:
        Fw = [[0, np.inf], [0, 0]]
        M_c_fex1 = dc.Mz_ss(R1, TR, FA, v, Fw)  
    except:
        assert True
    else:
        assert False   


def test_Mz_spgr():

    FA = 12
    TR = 0.005
    TI = np.linspace(0,3,100)

    R1 = [1, 0.5]
    v = [0.3, 0.7]
    f = 0.5
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]
    Mspgr = dc.Mz_spgr(R1, TI, TR, FA, v, Fw, j=[f, 0], n0=-1) 
    Mss = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f, 0])

    # Check that SPGR converges to steady state
    assert np.linalg.norm(Mspgr[:,-1]-Mss) < 1e-4*np.linalg.norm(Mss)


if __name__ == "__main__":

    test_Mz_free()
    test_Mz_ss()
    test_Mz_spgr()

    print('All relaxation tests passing!')