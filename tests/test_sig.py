import numpy as np
import dcmri as dc


def test_signal_dsc():
    S0 = 1
    TE = 0
    TR = 0
    R2 = 1
    R1 = 1
    S = dc.signal_dsc(R1, R2, S0, TR, TE)
    assert S==0


def test_signal_t2w():
    S0 = 1
    TE = 0
    R2 = 1
    S = dc.signal_t2w(R2, S0, TE)
    assert S==1

def test_Mz_free():

    # Tests 1 (may duplicate some of Tests 2 below)
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

    # Tests 2
    R1 = 1
    T = [1,2]
    S = dc.Mz_free(R1, T)
    assert 0.6 < S[0] < 0.7
    R1 = [1,1]
    S = dc.Mz_free(R1, T)
    assert 0.6 < S[0,0] < 0.7
    R1 = 1
    T = 1
    S = dc.Mz_free(R1, T)
    assert 0.6 < S < 0.7
    R1 = [1,1]
    S = dc.Mz_free(R1, T)
    assert 0.6 < S[0] < 0.7
    v = [0.2, 0.3]
    S = dc.Mz_free(R1, T, v)
    assert 0.1 < S[0] < 0.2
    R1 = np.ones((2,3))
    j = np.ones((2,3))
    S = dc.Mz_free(R1, T, v, j=j)
    assert 0.7 < S[0,0] < 0.8
    S = dc.Mz_free(R1, T, v)
    assert 0.1 < S[0,0] < 0.2
    S = dc.Mz_free(R1, T, v, n0=[0,0])
    assert 0.1 < S[0,0] < 0.2
    S = dc.Mz_free(R1, T, v, n0=np.zeros((2,3)))
    assert 0.1 < S[0,0] < 0.2
    v = 1
    try:
        S = dc.Mz_free(R1, T, v)
    except:
        assert True
    else:
        assert False

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

    # Check all cases
    R1 = 1
    TR = 0.005
    FA = 15
    S = dc.Mz_ss(R1, TR, FA)
    assert 0.1 < S < 0.2
    S = dc.Mz_ss(0, TR, FA)
    assert S==0
    R1 = [1,1]
    S = dc.Mz_ss(R1, TR, FA)
    assert 0.1 < S[0] < 0.2
    v = [0.2, 0.3]
    S = dc.Mz_ss(R1, TR, FA, v)
    assert 0.02 < S[0] < 0.03
    S = dc.Mz_ss(R1, TR, FA, v, Fw=0.1)
    assert 0.02 < S[0] < 0.03
    R1 = np.ones((2,3))
    j = np.ones((2,3))
    S = dc.Mz_ss(R1, TR, FA, v, j=j)
    assert 0.02 < S[0,0] < 0.03
    S = dc.Mz_ss(R1, TR, FA, v)
    assert 0.02 < S[0,0] < 0.03
    v = 1
    try:
        S = dc.Mz_ss(R1, TR, FA, v)
    except:
        assert True
    else:
        assert False


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
    Mspgr = dc.Mz_spgr(R1, TI, TR, FA, TP, v, Fw, j=[f, 0], n0=-1) 
    Mss = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f, 0])

    # Check that SPGR converges to steady state
    assert np.linalg.norm(Mspgr[:,-1]-Mss) < 1e-4*np.linalg.norm(Mss)

    # Test cases
    R1 = 1
    T = [1, 2]
    TR = 0.005
    FA = 15
    TP= 0
    S = dc.Mz_spgr(R1, T, TR, FA, TP)
    assert 0.1 < S[0] < 0.2
    R1 = [1,1]
    S = dc.Mz_spgr(R1, T, TR, FA, TP)
    assert 0.1 < S[0,0] < 0.2
    R1 = 1
    T = 1
    S = dc.Mz_spgr(R1, T, TR, FA, TP)
    assert 0.1 < S < 0.2
    R1 = [1,1]
    S = dc.Mz_spgr(R1, T, TR, FA, TP)
    assert 0.1 < S[0] < 0.2
    v = [0.2, 0.3]
    S = dc.Mz_spgr(R1, T, TR, FA, TP, v)
    assert 0.02 < S[0] < 0.03
    R1 = np.ones((2,3))
    j = np.ones((2,3))
    S = dc.Mz_spgr(R1, T, TR, FA, TP, v, j=j)
    assert 0.02 < S[0,0] < 0.03
    S = dc.Mz_spgr(R1, T, TR, FA, TP, v)
    assert 0.02 < S[0,0] < 0.03
    try:
        S = dc.Mz_spgr(R1, T, TR, FA, TP, v=1)
    except:
        assert True
    else:
        assert False

def test_signal_ss():
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    S = dc.signal_ss(S0, R1, TR, FA)
    assert S==0
    v = [0.5, 0.5]
    try:
        S = dc.signal_ss(S0, R1, TR, FA, v=v)
    except:
        assert True
    else:
        assert False
    R1 = [1,1]
    S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=np.inf)
    assert S==0
    S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=0)
    assert S==0
    Fw = [[0,1],[1,0]]
    S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw)
    assert S==0
    R1 = np.ones((2,10))
    S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=np.inf)
    assert 0==np.linalg.norm(S)
    S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=0)
    assert 0==np.linalg.norm(S)
    R1 = [1,1]
    S = dc.signal_ss(S0, R1, TR, FA)
    assert 0==np.linalg.norm(S)

    # Calibrate on other R10 reference
    R1 = 1
    S0 = 5
    TR = 1
    FA = 45
    S = dc.signal_ss(S0, R1, TR, FA, R10=R1)
    assert np.round(S)==5
    R1 = [1,1]
    v = [0.5, 0.5]
    S = dc.signal_ss(S0, R1, TR, FA, v=v, R10=R1, Fw=np.inf)
    assert np.round(S)==5

    # Check exceptions
    try:
        S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=[1,1])
    except:
        assert True
    else:
        assert False

    try:
        S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=np.ones((3,3)))
    except:
        assert True
    else:
        assert False

    v = [0.1, 0.4, 0.5]
    try:
        S = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=0)
    except:
        assert True
    else:
        assert False

    R1 = [1,1]
    try:
        S = dc.signal_ss(S0, R1, TR, FA, v=v)
    except:
        assert True
    else:
        assert False

    # Check boundary regimes with flow
    n = 100
    R1 = np.ones((2,n))
    j = np.ones((2,n))
    v = [0.5, 0.5]
    S0 = 1
    TR = 1
    FA = 10

    # No water exchange
    zero = 1e-6
    Fw = [[0.1,zero],[zero,1]]
    S1 = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw, j=j)
    Fw = [[0.1,0],[0,1]]
    S2 = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw, j=j)
    assert np.linalg.norm(S1-S2) < 1e-6*np.linalg.norm(S2)

    # Fast water exchange
    inf = 1e+6
    Fw = np.array([[0.1,inf],[inf,1]])
    S1 = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw, j=j)
    Fw = np.array([[0.1,np.inf],[np.inf,1]])
    S2 = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw, j=j)
    assert np.linalg.norm(S1-S2) < 1e-6*np.linalg.norm(S2)
    Fw = [[0.1,inf],[np.inf,1]]
    try:
        S2 = dc.signal_ss(S0, R1, TR, FA, v=v, Fw=Fw, j=j)
    except:
        assert True
    else:
        assert False
    # assert np.linalg.norm(S1-S2) < 1e-6*np.linalg.norm(S2)


def test_signal_spgr():
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    TC = 1
    S = dc.signal_spgr(S0, R1, TC, TR, FA, n0=0)
    assert S==0
   
    v = [0.5, 0.5]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, n0=0)
    except:
        assert True
    else:
        assert False
    R1 = [1,1]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, n0=0)
    assert S==0
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, Fw=0, n0=0)
    assert S==0
    Fw = [[0,1],[1,0]]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, Fw=Fw, n0=0)
    assert S==0
    R1 = np.ones((2,10))
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, n0=0)
    assert 0==np.linalg.norm(S)
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, Fw=0, n0=0)
    assert 0==np.linalg.norm(S)
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP=10, v=v, Fw=0, n0=0)
    assert 0==np.linalg.norm(S)
    S = dc.signal_spgr(S0, [1,1], TC, TR, FA, TP=10, v=v, Fw=0, n0=0)
    assert 0==np.linalg.norm(S)

    # Check exceptions
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, Fw=[1,1], n0=0)
    except:
        assert True
    else:
        assert False

    v = [0.1, 0.4, 0.5]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, Fw=0, n0=0)
    except:
        assert True
    else:
        assert False

    R1 = [1,1]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, n0=0)
    except:
        assert True
    else:
        assert False

    # Calibrate on other R10 reference
    R1 = 1
    S0 = 5
    TR = 1
    FA = 45
    S = dc.signal_spgr(S0, R1, TC, TR, FA, R10=R1, n0=0)
    assert np.round(S)==5
    R1 = [1,1]
    v = [0.5, 0.5]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, v=v, R10=R1, n0=0)
    assert np.round(S)==5

    # With preparation
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    TC = 1
    TP = 0
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP)
    assert S==0
    v = [0.5, 0.5]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v)
    except:
        assert True
    else:
        assert False
    R1 = [1,1]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v)
    assert S==0
    R1 = [1,1]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP)
    assert S[0]==0
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=0)
    assert S==0
    Fw = [[0,1],[1,0]]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=Fw)
    assert S==0
    R1 = np.ones((2,10))
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v)
    assert 0==np.linalg.norm(S)
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=0)
    assert 0==np.linalg.norm(S)

    # Check exceptions
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=[1,1])
    except:
        assert True
    else:
        assert False

    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=np.ones((3,3)))
    except:
        assert True
    else:
        assert False

    v = [0.1, 0.4, 0.5]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, Fw=0)
    except:
        assert True
    else:
        assert False

    R1 = [1,1]
    try:
        S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v)
    except:
        assert True
    else:
        assert False

    # Calibrate on other R10 reference
    R1 = 1
    S0 = 5
    TR = 1
    FA = 45
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, R10=R1)
    assert np.round(S)==5
    R1 = [1,1]
    v = [0.5, 0.5]
    S = dc.signal_spgr(S0, R1, TC, TR, FA, TP, v=v, R10=R1)
    assert np.round(S)==5

    # Check convergence to fast exchange


def test_signal_free():
    TC = 0
    R1 = 1
    S0 = 1 
    FA = 45
    S = dc.signal_free(S0, R1, TC, FA, R10=None)
    assert S==0
    TC = 1
    S0 = 0
    S = dc.signal_free(S0, R1, TC, FA, R10=1)
    assert S==0
    R1 = [1,1]
    S0 = 1
    S = dc.signal_free(S0, R1, TC, FA)
    assert 0.4 < S[0] < 0.5
    S = dc.signal_free(S0, R1, TC, FA, v=[0.2, 0.3])
    assert 0.2 < S < 0.3
    R1 = np.ones((2,3))
    S = dc.signal_free(S0, R1, TC, FA, v=[0.2, 0.3])
    assert 0.2 < S[0] < 0.3


def test_signal_lin():
    R1 = 2
    S0 = 3
    assert 6 == dc.signal_lin(S0, R1)


def test_conc_t2w():
    S = np.ones(10)
    TE = np.inf
    C = dc.conc_t2w(S, TE, r2=1, n0=1)
    assert 0 == np.linalg.norm(C)
    
def test_conc_ss():
    S = np.ones(10)
    TR = 1
    FA = 45
    T10 = np.inf
    C = dc.conc_ss(S, TR, FA, T10, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

    # Data without solution
    S = [1,2,0]
    T10 = 1
    C = dc.conc_ss(S, TR, FA, T10, r1=1, n0=1)
    assert C[-1] == -1

def test_conc_src():
    S = np.ones(10)
    TC = 1
    T10 = np.inf
    C = dc.conc_src(S, TC, T10, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

def test_conc_lin():
    S = np.ones(10)
    T10 = 1
    C = dc.conc_lin(S, T10, r1=1, n0=1)
    assert np.linalg.norm(C)==0

def test_signal():
    S = dc.signal('SRC', 1, 2, TC=0.1, FA=10)
    assert 0.03 < S < 0.04
    S = dc.signal('SR', 1, 2, TC=0.1, FA=10, TR=0.005)
    assert 0.02 < S < 0.03
    S = dc.signal('SS', 1, 2, FA=10, TR=0.005)
    assert 0.08 < S < 0.09
    try:
        S = dc.signal('X', 1, 2)
    except:
        assert True
    else:
        assert False


if __name__ == "__main__":

    test_signal()
    test_signal_dsc()
    test_signal_t2w()
    test_Mz_free()
    test_Mz_ss()
    test_Mz_spgr()
    test_signal_ss()
    test_signal_spgr()
    test_signal_free()
    test_signal_lin()

    test_conc_t2w()
    test_conc_ss()
    test_conc_src()
    test_conc_lin()

    print('All sig tests passing!')