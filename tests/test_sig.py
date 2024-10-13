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


def test_signal_ss():
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    S = dc.signal_ss(R1, S0, TR, FA)
    assert S==0
    v = [0.5, 0.5]
    try:
        S = dc.signal_ss(R1, S0, TR, FA, v=v)
    except:
        assert True
    else:
        assert False
    R1 = [1,1]
    S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=np.inf)
    assert S==0
    S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=0)
    assert S==0
    PSw = [[0,1],[1,0]]
    S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=PSw)
    assert S==0
    R1 = np.ones((2,10))
    S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=np.inf)
    assert 0==np.linalg.norm(S)
    S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=0)
    assert 0==np.linalg.norm(S)

    # Check exceptions
    try:
        S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=[1,1])
    except:
        assert True
    else:
        assert False

    try:
        S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=np.ones((3,3)))
    except:
        assert True
    else:
        assert False

    v = [0.1, 0.4, 0.5]
    try:
        S = dc.signal_ss(R1, S0, TR, FA, v=v, PSw=0)
    except:
        assert True
    else:
        assert False

    R1 = [1,1]
    try:
        S = dc.signal_ss(R1, S0, TR, FA, v=v)
    except:
        assert True
    else:
        assert False

    # Calibrate on other R10 reference
    R1 = 1
    S0 = 5
    TR = 1
    FA = 45
    S = dc.signal_ss(R1, S0, TR, FA, R10=R1)
    assert np.round(S)==5
    R1 = [1,1]
    v = [0.5, 0.5]
    S = dc.signal_ss(R1, S0, TR, FA, v=v, R10=R1[0], PSw=np.inf)
    assert np.round(S)==5



def test_signal_sr():
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    TC = 1
    TP = 0
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP)
    assert S==0
    v = [0.5, 0.5]
    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v)
    except:
        assert True
    else:
        assert False
    R1 = [1,1]
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v)
    assert S==0
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=0)
    assert S==0
    PSw = [[0,1],[1,0]]
    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=PSw)
    except NotImplementedError:
        assert True
    else:
        assert False
    #assert S==0
    R1 = np.ones((2,10))
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v)
    assert 0==np.linalg.norm(S)
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=0)
    assert 0==np.linalg.norm(S)

    # Check exceptions
    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TC+1, v=v)
    except:
        assert True
    else:
        assert False

    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=[1,1])
    except:
        assert True
    else:
        assert False

    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=np.ones((3,3)))
    except:
        assert True
    else:
        assert False

    v = [0.1, 0.4, 0.5]
    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, PSw=0)
    except:
        assert True
    else:
        assert False

    R1 = [1,1]
    try:
        S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v)
    except:
        assert True
    else:
        assert False

    # Calibrate on other R10 reference
    R1 = 1
    S0 = 5
    TR = 1
    FA = 45
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, R10=R1)
    assert np.round(S)==5
    R1 = [1,1]
    v = [0.5, 0.5]
    S = dc.signal_sr(R1, S0, TR, FA, TC, TP, v=v, R10=R1[0])
    assert np.round(S)==5


def test_signal_er():
    R1 = 1
    S0 = 1
    TR = 1
    FA = 0
    TC = 1
    S = dc.signal_er(R1, S0, TR, FA, TC)
    assert S==0


def test_signal_src():
    TC = 0
    R1 = 1
    S0 = 1 
    S = dc.signal_src(R1, S0, TC, R10=None)
    assert S==0
    TC = 1
    S0 = 0
    S = dc.signal_src(R1, S0, TC, R10=1)
    assert S==0


def test_signal_lin():
    R1 = 2
    S0 = 3
    assert 6 == dc.signal_lin(R1, S0)


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


if __name__ == "__main__":

    test_signal_dsc()
    test_signal_t2w()
    test_signal_ss()
    test_signal_sr()
    test_signal_er()
    test_signal_src()
    test_signal_lin()

    test_conc_t2w()
    test_conc_ss()
    test_conc_src()
    test_conc_lin()

    print('All sig tests passing!')