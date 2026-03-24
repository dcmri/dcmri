import numpy as np
import dcmri as dc



def test_conc_t2w():
    S = np.ones(10)
    TE = np.inf
    r2 = 1
    C = dc.conc('T2w', S, TE=TE, r2=r2, n0=1)
    assert 0 == np.linalg.norm(C)

    # Test reconstruction
    R20 = 1
    C = np.arange(10)
    R2 = R20 + r2 * C
    TE = 0.005
    S0 = 5
    S = S0 * np.exp(-TE * R2)
    Crec = dc.conc('T2w', S, TE=TE, r2=r2, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    
def test_conc_ss():
    S = np.ones(10)
    C = dc.conc('SS', S, R10=0, TR=1, FA=45, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

    # Data without solution
    S = [1,2,0]
    C = dc.conc('SS', S, TR=1, FA=45, R10=1, r1=1, n0=1)
    assert C[-1] == -1

    # Test reconstruction
    R10 = 1
    r1 = 0.5
    C = np.arange(10)
    R1 = R10 + r1 * C
    TR, FA, S0 = 0.002, 30, 5
    Mz = dc.Mz('SS', R1, TR=TR, FA=FA)
    S = dc.signal(Mz, S0=S0, FA=FA)
    Crec = dc.conc('SS', S, R10=R10, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('SS', S, S0=S0, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

def test_conc_free():
    S = np.ones(10)
    C = dc.conc('free', S, FA=45, TC=1, R10=0, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

    # Test reconstruction
    R10 = 1
    r1 = 0.5
    C = np.arange(10)
    R1 = R10 + r1 * C
    TC, FA, S0 = 0.2, 30, 5
    Mz = dc.Mz('free', R1, TC=TC)
    S = dc.signal(Mz, S0=S0, FA=FA)
    Crec = dc.conc('free', S, R10=R10, TC=TC, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('free', S, S0=S0, TC=TC, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

def test_conc_lin():
    S = np.ones(10)
    C = dc.conc('lin', S, R10=1, r1=1, n0=1)
    assert np.linalg.norm(C)==0

    R10 = 1
    r1 = 0.5
    C = np.arange(10)
    R1 = R10 + r1 * C
    S0 = 5
    S = S0 * R1
    Crec = dc.conc('lin', S, R10=R10, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('lin', S, S0=S0, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

def test_conc_spgr():
    S = np.ones(10)
    C = dc.conc('SPGR', S, R10=0, TC=0.2, TR=1, FA=45, TP=0.1, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

    # Test reconstruction
    R10 = 1
    r1 = 0.5
    TC, TP, TR, FA, S0 = 0.2, 0.1, 0.002, 30, 5
    n_init = 1
    C = np.arange(10)
    R1 = R10 + r1 * C
    Mz = dc.Mz('SPGR', R1, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA)
    S = dc.signal(Mz, S0=S0, FA=FA)
    Crec = dc.conc('SPGR', S, R10=R10, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('SPGR', S, S0=S0, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # SR
    n_init = 0
    Mz = dc.Mz('SPGR', R1, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA)
    S = dc.signal(Mz, S0=S0, FA=FA)
    Crec = dc.conc('SR', S, R10=R10, TC=TC, TP=TP, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('SR', S, S0=S0, TC=TC, TP=TP, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # IR
    n_init = -1
    Mz = dc.Mz('SPGR', R1, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA)
    S = dc.signal(Mz, S0=S0, FA=FA)
    Crec = dc.conc('IR', S, R10=R10, TC=TC, TP=TP, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    Crec = dc.conc('IR', S, S0=S0, TC=TC, TP=TP, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # 2D Signal array
    n_init = 1
    C = np.arange(10)
    R1 = R10 + r1 * C
    Mz = dc.Mz('SPGR', R1, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA)
    S = dc.signal(Mz, S0=S0, FA=FA)
    S = np.stack((S, S), axis=0) # two samples

    R10 = np.full(S.shape[0], R10)
    Crec = dc.conc('SPGR', S, R10=R10, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    # Given S0
    S0 = np.full(S.shape[0], S0)
    Crec = dc.conc('SPGR', S, S0=S0, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    assert np.linalg.norm(C - Crec) < 1e-6

    try:
        Crec = dc.conc('XXX', S, S0=S0, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    except: 
        pass
    else:
        assert False

    # Test exceptions
    try:
        Crec = dc.conc('SPGR', [1], R10=R10, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    except:
        pass
    else:
        assert False
    
    R10 = np.full(3, R10[0])
    try:
        Crec = dc.conc('SPGR', S, R10=R10, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    except:
        pass
    else:
        assert False

    S0 = np.full(3, S0[0])
    try:
        Crec = dc.conc('SPGR', S, S0=S0, TC=TC, TP=TP, n_init=n_init, TR=TR, FA=FA, r1=r1, n0=1)
    except: 
        pass
    else:
        assert False



def test_params_conc():
    assert dc.params_conc('SR') == ['TC', 'TR', 'FA', 'TP', 'r1', 'n0']

if __name__ == "__main__":

    test_params_conc()
    test_conc_t2w()
    test_conc_ss()
    test_conc_free()
    test_conc_lin()
    test_conc_spgr()

    print('All conc_inv tests passing!')