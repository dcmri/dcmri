import numpy as np
import dcmri as dc


def test_Mz_free():

    # Tests 1 (may duplicate some of Tests 2 below)
    R1 = 1
    TI = 0.1*np.arange(100)
    f = 0.5

    Mz = np.array([dc.Mz('free', R1, n_init=-1, TC=ti) for ti in TI])
    Mz_e = np.array([dc.Mz('free', R1, n_init=-1, Fw=f, j=f, TC=ti) for ti in TI])
    Mz_i = np.array([dc.Mz('free', R1, n_init=-1, Fw=f, j=-f, TC=ti) for ti in TI])
    assert 21 < np.linalg.norm(Mz + Mz_e + Mz_i) < 22

    # Two compartments, no time
    R1 = [1,2]
    v = [0.3, 0.7]
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]

    Mz =np.array([dc.Mz('free', R1, v, Fw, j=[f, 0], n_init=-1, TC=ti) for ti in TI])
    assert 7 < np.linalg.norm(Mz) < 8

    TI = 0.5
    nt = 1000
    t = 0.1*np.arange(nt)
    R1 = np.stack((1-t/np.amax(t), np.ones(nt)))
    j = np.stack((f*np.ones(nt), np.zeros(nt)))
    Mz = dc.Mz('free', R1, v, Fw, j=j, n_init=-1, TC=TI)

    assert 5 < np.linalg.norm(Mz) < 6

    TI = 0.1*np.arange(10)
    Mzi = np.stack([dc.Mz('free', R1, v, Fw, j=j, n_init=-1, TC=ti) for ti in TI], axis=-1)

    assert np.linalg.norm(Mzi[:,:,5]-Mz) == 0

    # Tests 2
    R1 = 1
    T = [1, 2]
    S = np.stack([dc.Mz('free', R1, TC=t) for t in T], axis=-1)
    assert 0.6 < S[0] < 0.7
    R1 = [1,1]
    S = np.stack([dc.Mz('free', R1, TC=t) for t in T], axis=-1)
    assert 0.6 < S[0,0] < 0.7
    R1 = 1
    T = 1
    S = dc.Mz('free', R1, TC=T)
    assert 0.6 < S < 0.7
    R1 = [1,1]
    S = dc.Mz('free', R1, TC=T)
    assert 0.6 < S[0] < 0.7
    v = [0.2, 0.3]
    S = dc.Mz('free', R1, v, TC=T)
    assert 0.1 < S[0] < 0.2
    R1 = np.ones((2,3))
    j = np.ones((2,3))
    S = dc.Mz('free', R1, v, j=j, TC=T)
    assert 0.7 < S[0,0] < 0.8
    S = dc.Mz('free', R1, v, TC=T)
    assert 0.1 < S[0,0] < 0.2
    S = dc.Mz('free', R1, v, n_init=[0,0], TC=T)
    assert 0.1 < S[0,0] < 0.2


def test_Mz_ss():

    # Functional tests

    # Steady state magnetization with different inflows
    R1 = 1
    FA = 12
    TR = 0.005
    f = 0.5
    v = 0.7

    # Check that inflow with steady-state magnetization 
    # is the same as no inflow
    m_c = dc.Mz('SS', R1, v, TR=TR, FA=FA)/v
    m_f = dc.Mz('SS', R1, v, Fw=f, j=f*m_c, TR=TR, FA=FA)/v
    assert np.abs(m_f-m_c) < 0.01*np.abs(m_c)

    # Repeat for a two-compartment system:
    R1 = [0.5, 1.5]
    v = [0.3, 0.6]
    PS = 0.1

    Fw = [[0, PS], [PS, 0]]
    M_c = dc.Mz('SS', R1, v, Fw, TR=TR, FA=FA)

    Fw = [[f, PS], [PS, 0]]
    m_c = M_c[0]/v[0]
    m_f = dc.Mz('SS', R1, v, Fw, j=[f*m_c, 0], TR=TR, FA=FA)[0]/v[0]

    assert np.abs(m_c-m_f) < 0.01*np.abs(m_c)

    # Check fast-exchange limit
    v = [0.3, 0.5]
    M_c = dc.Mz('SS', R1, v, 1e9, TR=TR, FA=FA)
    M_c_fex = dc.Mz('SS', R1, v, np.inf, TR=TR, FA=FA)
    assert np.linalg.norm(M_c-M_c_fex) < 1e-3*np.linalg.norm(M_c_fex)

    # Check no-exchange limit
    v = [0.3, 0.5]
    M_c = dc.Mz('SS', R1, v, 1e-9, TR=TR, FA=FA)
    M_c_nex = dc.Mz('SS', R1, v, 0, TR=TR, FA=FA)
    assert np.linalg.norm(M_c-M_c_nex) < 1e-6*np.linalg.norm(M_c_nex)

    # Check exceptions
    Fw = [[0, np.inf], [np.inf, 0]]
    M_c_fex1 = dc.Mz('SS', R1, v, Fw, TR=TR, FA=FA)
    M_c_fex2 = dc.Mz('SS', R1, v, np.inf, TR=TR, FA=FA)
    assert np.linalg.norm(M_c_fex1-M_c_fex2) < 1e-9*np.linalg.norm(M_c_fex2)

    Fw = [[0, 0], [0, 0]]
    M_c_nex1 = dc.Mz('SS', R1, v, Fw, TR=TR, FA=FA)
    M_c_nex2 = dc.Mz('SS', R1, v, 0, TR=TR, FA=FA)
    assert np.linalg.norm(M_c_nex1-M_c_nex2) < 1e-9*np.linalg.norm(M_c_nex2)

    try:
        Fw = [[0, np.inf], [0, 0]]
        M_c_fex1 = dc.Mz('SS', R1, v, Fw, TR=TR, FA=FA)  
    except:
        assert True
    else:
        assert False   

    # Check all cases
    R1 = 1
    TR = 0.005
    FA = 15
    S = dc.Mz('SS', R1, TR=TR, FA=FA)
    assert 0.1 < S < 0.2
    S = dc.Mz('SS', R1, j=1, TR=TR, FA=FA)
    assert 0.2 < S < 0.3
    S = dc.Mz('SS', 0, TR=TR, FA=FA)
    assert S==0
    R1 = [1,1]
    S = dc.Mz('SS', R1, TR=TR, FA=FA)
    assert 0.1 < S[0] < 0.2
    v = [0.2, 0.3]
    S = dc.Mz('SS', R1, v, TR=TR, FA=FA)
    assert 0.02 < S[0] < 0.03
    S = dc.Mz('SS', R1, v, Fw=0.1, TR=TR, FA=FA)
    assert 0.02 < S[0] < 0.03
    R1 = np.ones((2,3))
    S = dc.Mz('SS', R1, v, Fw=0.1, TR=TR, FA=FA)
    assert 0.02 < S[0,0] < 0.03
    j = np.zeros((2,3))
    S = dc.Mz('SS', R1, v, Fw=0.1, j=j, TR=TR, FA=FA)
    assert 0.02 < S[0,0] < 0.03
    S = dc.Mz('SS', R1, v, Fw=0.0, j=j, TR=TR, FA=FA)
    assert 0.02 < S[0,0] < 0.03
    j = np.ones((2,3))
    S = dc.Mz('SS', R1, v, j=j, TR=TR, FA=FA)
    assert 0.1 < S[0,0] < 0.2


def test_Mz_spgr():

    # Functional test
    FA = 12
    TR = 0.005
    TI = np.linspace(0,3,100)
    TP = 0

    R1 = [1, 0.5]
    v = [0.3, 0.7]
    f = 0.5
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]
    Mspgr = np.stack([dc.Mz('SPGR', R1, v, Fw, j=[f, 0], n_init=-1, TC=ti, TR=TR, FA=FA, TP=TP) for ti in TI], axis=-1)
    Mss = dc.Mz('SS', R1, v, Fw, j=[f, 0], TR=TR, FA=FA)

    # Check that SPGR converges to steady state
    assert np.linalg.norm(Mspgr[:,-1]-Mss) < 1e-4*np.linalg.norm(Mss)

    # Test cases
    R1 = 1
    T = 2
    TR = 0.005
    FA = 15
    TP= 0
    S = dc.Mz('SPGR', R1, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.1 < S < 0.2
    S = dc.Mz('SPGR', R1, TC=T, TR=TR, FA=FA, TP=10)
    assert 0.1 < S < 0.2
    R1 = [1,1]
    S = dc.Mz('SPGR', R1, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.1 < S[0] < 0.2
    R1 = 1
    T = 1
    S = dc.Mz('SPGR', R1, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.1 < S < 0.2
    R1 = [1,1]
    S = dc.Mz('SPGR', R1, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.1 < S[0] < 0.2
    v = [0.2, 0.3]
    S = dc.Mz('SPGR', R1, v, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.02 < S[0] < 0.03
    R1 = np.ones((2,3))
    j = np.zeros((2,3))
    S = dc.Mz('SPGR', R1, v, j=j, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.02 < S[0,0] < 0.03
    j = np.ones((2,3))
    S = dc.Mz('SPGR', R1, v, j=j, TC=T, TR=TR, FA=FA, TP=TP)
    assert 0.1 < S[0,0] < 0.2
    j = np.ones((2,3))
    S = dc.Mz('SPGR', R1, v, j=j, TC=T, TR=TR, FA=FA, TP=10)
    assert 0.1 < S[0,0] < 0.2

def test_Mz_sr():

    # Functional test
    FA = 12
    TR = 0.005
    TI = np.linspace(0,3,100)
    TP = 0

    R1 = [1, 0.5]
    v = [0.3, 0.7]
    f = 0.5
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]
    Msr = np.stack([dc.Mz('SR', R1, v, Fw, j=[f, 0], TC=ti, TR=TR, FA=FA, TP=TP) for ti in TI], axis=-1)
    Mss = dc.Mz('SS', R1, v, Fw, j=[f, 0], TR=TR, FA=FA)

    # Check that SPGR converges to steady state
    assert np.linalg.norm(Msr[:,-1]-Mss) < 1e-4*np.linalg.norm(Mss)

def test_Mz_ssi():

    # Functional test
    FA = 12
    TR = 0.005
    TI = np.linspace(0,3,100)
    TP = 0

    R1 = [1, 0.5]
    v = [0.3, 0.7]
    f = 0.5
    PS = 0.1
    Fw = [[f, PS], [PS, 0]]
    Msr = np.stack([dc.Mz('SSI', R1, v, Fw, j=[f, 0], TF=ti, TR=TR, FA=FA, TP=TP) for ti in TI], axis=-1)
    Mss = dc.Mz('SS', R1, v, Fw, j=[f, 0], TR=TR, FA=FA)

    # Check that SPGR converges to steady state
    assert np.linalg.norm(Msr[:,-1]-Mss) < 1e-4*np.linalg.norm(Mss)




def test_params_Mz():
    assert dc.params_Mz('SS') == ['TR', 'FA']

if __name__ == "__main__":

    test_params_Mz()
    test_Mz_free()
    test_Mz_ss()
    test_Mz_spgr()
    test_Mz_sr()
    test_Mz_ssi()

    print('All mz tests passing!')