import numpy as np
from dcmri.bloch import functions_dynamic_sequences

def test_Mz_se():
    """Test Mz_se shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, FA = 1.0, 0.05, 90.0

    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, res_finite = functions_dynamic_sequences.Mz_se(tR1, R1, v, Fw, j, me, TE, TR=2.0, FA=FA)
    assert res_finite.shape == (2, 10, 2)
    assert t.shape == (10,)


def test_Mz_spgr_in_ss():
    """Test Mz_spgr_in_ss shortcut branch and standard loop shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph = 1.0, 0.005, 15.0, 128
    
    # Running actual underlying pulse simulation loop
    tR1 = dt * np.arange(R1.shape[1])
    t, res_finite = functions_dynamic_sequences.Mz_spgr_in_ss(tR1, R1, v, Fw, j, me, TR, FA, Nph)
    assert res_finite.shape == (2, 32, Nph)
    assert t.shape == (32,)


def test_Mz_pr_spgr():
    """Test Mz_pr_spgr loop tracking and shape parsing."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, TP, TD, PA = 1.0, 0.005, 15.0, 128, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, res = functions_dynamic_sequences.Mz_pr_spgr(tR1, R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA)
    assert res.shape == (2, 22, 129)
    assert t.shape == (22, )


def test_Mz_pr_spgr_in_ss():
    """Test steady-state preparation SPGR execution loop and output shape."""
    dt = 10.0
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TR, FA, Nph, TP, TD, PA = 1.0, 0.005, 15.0, 128, 0.1, 0.2, 120

    tR1 = dt * np.arange(R1.shape[1])
    t, res = functions_dynamic_sequences.Mz_pr_spgr_in_ss(tR1, R1, v, Fw, j, me, TR, FA, Nph, TP, TD, PA)
    assert res.shape == (2, 22, 129)
    assert t.shape == (22, )


def test_Mz_spgr_in_ssi():
    """Test Mz_ssi behavior handling both 1-compartment and multi-compartment execution blocks."""
    dt = 10
    Fw, me, TR, FA, TF, SA, Nph, Nctr = 1.0, 1.0, 0.01, 15.0, 0.05, 90.0, 128, 64
    
    # Case 1: Single Compartment branch (nc == 1)
    R1_1d = np.ones((1, 3))
    j_1d = np.ones((1, 3))
    v = 1

    tR1 = dt * np.arange(R1_1d.shape[1])
    t, res_1d = functions_dynamic_sequences.Mz_spgr_in_ssi(tR1, R1_1d, v, Fw, j_1d, me, TR, FA, Nph, TF, SA)
    assert res_1d.shape == (1, 16, 128)
    
    # Case 2: Multi-Compartment branch (nc == 2)
    R1_2d = np.ones((2, 3))
    j_2d = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)

    tR1 = dt * np.arange(R1_2d.shape[1])
    t, res_2d = functions_dynamic_sequences.Mz_spgr_in_ssi(tR1, R1_2d, v, Fw, j_2d, me, TR, FA, Nph, TF, SA)
    assert res_2d.shape == (2, 16, 128)

if __name__=='__main__':
    test_Mz_se()
    test_Mz_spgr_in_ss()
    test_Mz_pr_spgr()
    test_Mz_pr_spgr_in_ss()
    test_Mz_spgr_in_ssi()

    print('All bloch.seqs tests passed!')