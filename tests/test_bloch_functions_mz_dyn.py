import numpy as np
import dcmri as dc

def test_Mz_dyn():
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, TE, FA = 1.0, 0.5, 0.05, 15.0

    tR1 = dt * np.arange(R1.shape[1])
    pulses_per_period = [[FA, TE / 2], [180, TR - TE/2]]
    t, Mz = dc.Mz_dyn(tR1, R1, v, Fw, j, me, pulses_per_period, tj=tR1)
    assert Mz.shape == (2, 40, 2)
    assert t.shape == (40, 2)

def test_Mz_dyn_k0():
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, TE, FA = 1.0, 0.5, 0.05, 15.0

    tR1 = dt * np.arange(R1.shape[1])
    pulses_per_period = [[FA, TE / 2], [180, TR - TE/2]]
    t, Mz = dc.Mz_dyn_k0(tR1, R1, v, Fw, j, me, pulses_per_period, Nk0=1, tj=tR1)
    assert Mz.shape == (2, 40)
    assert t.shape == (40,)

def test_Mz_dyn_spgr():
    """Test Mz_dyn_pr_spgr loop tracking and shape parsing."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph = 1.0, 0.005, 15.0, 128

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, tj=tR1)
    assert Mz.shape == (2, 31, 128)
    assert t.shape == (31, 128)

def test_Mz_dyn_k0_spgr():
    """Test Mz_dyn_pr_spgr loop tracking and shape parsing."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, Nk0 = 1.0, 0.005, 15.0, 128, 64

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_k0_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, Nk0, tj=tR1)
    assert Mz.shape == (2, 31)
    assert t.shape == (31,)

def test_Mz_dyn_pr_spgr():
    """Test Mz_dyn_pr_spgr loop tracking and shape parsing."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, TP, TD, PA = 1.0, 0.005, 15.0, 128, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_pr_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, tj=tR1)
    assert Mz.shape == (2, 21, 129)
    assert t.shape == (21, 129)

def test_Mz_dyn_k0_pr_spgr():
    """Test Mz_dyn_pr_spgr loop tracking and shape parsing."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, Nk0, TP, TD, PA = 1.0, 0.005, 15.0, 128, 64, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_k0_pr_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, Nk0, TP, TD, PA, tj=tR1)
    assert Mz.shape == (2, 21)
    assert t.shape == (21,)

def test_Mz_dyn_ss_pr_spgr():
    """Test steady-state preparation SPGR execution loop and output shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, TP, TD, PA = 1.0, 0.005, 15.0, 128, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_ss_pr_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, tj=tR1)
    assert Mz.shape == (2, 21, 129)
    assert t.shape == (21, 129)

def test_Mz_dyn_ss_k0_pr_spgr():
    """Test steady-state preparation SPGR execution loop and output shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, Nk0, TP, TD, PA = 1.0, 0.005, 15.0, 128, 64, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_ss_k0_pr_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, Nk0, TP, TD, PA, tj=tR1)
    assert Mz.shape == (2, 21)
    assert t.shape == (21,)

    t_all, Mz_all = dc.Mz_dyn_ss_pr_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, TP, TD, PA, tj=tR1)
    assert np.linalg.norm(Mz_all[:, :, 1 + Nk0] - Mz) < 1e-12

def test_Mz_dyn_ss_spgr():
    """Test Mz_dyn_ss_spgr shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph = 1.0, 0.005, 15.0, 128
    
    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_ss_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, tj=tR1)
    assert Mz.shape == (2, 31, Nph)
    assert t.shape == (31, Nph)

def test_Mz_dyn_ss_k0_spgr():
    """Test Mz_dyn_ss_spgr shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, Nk0 = 1.0, 0.005, 15.0, 128, 64
    
    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_ss_k0_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, Nk0, tj=tR1)
    assert Mz.shape == (2, 31)
    assert t.shape == (31,)

    # Test against full simulation
    t_all, Mz_all = dc.Mz_dyn_ss_spgr(tR1, R1, v, Fw, j, me, FA, TR, Nph, tj=tR1)
    assert np.linalg.norm(Mz_all[:, :, Nk0] - Mz) < 1e-12

def test_Mz_dyn_ss():
    """Test Mz_dyn_se shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, TR, FA = 1.0, 0.05, 2.0, 90.0

    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    pulses_per_period = [[FA, TE / 2], [180, TR - TE/2]]
    t, Mz = dc.Mz_dyn_ss(tR1, R1, v, Fw, j, me, pulses_per_period, tj=tR1)
    assert Mz.shape == (2, 10, 2)
    assert t.shape == (10, 2)

def test_Mz_dyn_ss_k0():
    """Test Mz_dyn_se shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, TR, FA = 1.0, 0.05, 2.0, 90.0
    Nk0=1

    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    pulses_per_period = [[FA, TE / 2], [180, TR - TE/2]]
    t, Mz = dc.Mz_dyn_ss_k0(tR1, R1, v, Fw, j, me, pulses_per_period, Nk0=Nk0, tj=tR1)
    assert Mz.shape == (2, 10)
    assert t.shape == (10,)

    t_all, Mz_all = dc.Mz_dyn_ss(tR1, R1, v, Fw, j, me, pulses_per_period, tj=tR1)
    assert np.linalg.norm(Mz_all[:, :, Nk0] - Mz) < 1e-12

def test_Mz_dyn_se():
    """Test Mz_dyn_se shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, TR, FA = 1.0, 0.05, 2.0, 90.0

    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_se(tR1, R1, v, Fw, j, me, TE, FA, TR, tj=tR1)
    assert Mz.shape == (2, 10, 2)
    assert t.shape == (10, 2)

def test_Mz_dyn_k0_se():
    """Test Mz_dyn_se shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, TR, FA = 1.0, 0.05, 2.0, 90.0

    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, Mz = dc.Mz_dyn_k0_se(tR1, R1, v, Fw, j, me, TE, FA, TR, tj=tR1)
    assert Mz.shape == (2, 10)
    assert t.shape == (10,)

def test_Mz_dyn_ss_spgri():
    """Test Mz_ss_spgri behavior handling both 1-compartment and multi-compartment execution blocks."""
    dt = 10
    Fw, me, TR, FA, TF, SA, Nph = 1.0, 1.0, 0.01, 15.0, 0.05, 90.0, 128
    
    # Case 1: Single Compartment (nc == 1)
    R1_1d = np.ones((1, 3))
    j_1d = np.ones((1, 3))
    v = 1

    tR1 = dt * np.arange(R1_1d.shape[1])
    t, Mz_1d = dc.Mz_dyn_ss_spgri(tR1, R1_1d, v, Fw, j_1d, me, FA, TR, Nph, TF, SA, tj=tR1)
    assert Mz_1d.shape == (1, 15, 128)
    
    # Case 2: Multi-Compartment (nc == 2)
    R1_2d = np.ones((2, 3))
    j_2d = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)

    tR1 = dt * np.arange(R1_2d.shape[1])
    t, Mz_2d = dc.Mz_dyn_ss_spgri(tR1, R1_2d, v, Fw, j_2d, me, FA, TR, Nph, TF, SA, tj=tR1)
    assert Mz_2d.shape == (2, 15, 128)




if __name__=='__main__':

    test_Mz_dyn()
    test_Mz_dyn_k0()
    test_Mz_dyn_spgr()
    test_Mz_dyn_k0_spgr()
    test_Mz_dyn_pr_spgr()
    test_Mz_dyn_k0_pr_spgr()
    test_Mz_dyn_ss_pr_spgr()
    test_Mz_dyn_ss_k0_pr_spgr()
    test_Mz_dyn_ss_spgr()
    test_Mz_dyn_ss_k0_spgr()
    test_Mz_dyn_se()
    test_Mz_dyn_k0_se()
    test_Mz_dyn_ss()
    test_Mz_dyn_ss_k0()
    test_Mz_dyn_ss_spgri()

    print('All bloch.seqs tests passed!')