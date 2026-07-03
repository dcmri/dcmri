import numpy as np
from dcmri.bloch import functions_seqs

def test_mz_readout():
    """Test mz_readout outputs the correct array shape and values."""
    Mz = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    R2 = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    S0 = 100.0
    FA = 90.0  # sin(90) = 1.0
    TE = 0.0   # exp(0) = 1.0
    noise_sdev = 0.0  # No noise, clean signal
    
    # Expected: Mxy = 1.0*1.0*1.0 + 1.0*1.0*0.5 = 1.5 per timepoint
    # signal = 100 * 1.5 = 150
    expected = np.array([150.0, 150.0, 150.0])
    
    result = functions_seqs.mz_readout(Mz, R2, S0, FA, TE, noise_sdev)
    np.testing.assert_array_almost_equal(result, expected)


def test_signal_rice():
    """Test signal_rice for clean math path and its fallback logic."""
    # Case 1: Zero noise standard deviation returns input directly
    res_zero_sigma = functions_seqs.signal_rice(np.array([10.0]), sigma=0.0)
    np.testing.assert_array_almost_equal(res_zero_sigma, np.array([10.0]))
    
    # Case 2: Standard math path
    res_standard = functions_seqs.signal_rice(np.array([2.0, 5.0]), sigma=1.0)
    assert res_standard.shape == (2,)
    
    # Case 3: Extreme inputs trigger the fallback logic cleanly
    res_fallback = functions_seqs.signal_rice(np.array([1e10]), sigma=1e-10)
    np.testing.assert_array_almost_equal(res_fallback, np.array([1e10]))


def test_Mz_se():
    """Test Mz_se shortcut branch and standard loop shape."""
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TE, FA = 1.0, 0.05, 90.0
    
    # Case 1: Infinite TR shortcut
    res_inf = functions_seqs.Mz_se(R1, v, Fw, j, me, TE, np.inf, FA)
    np.testing.assert_array_almost_equal(res_inf, np.full_like(R1, me))
    
    # Case 2: Running actual underlying pulse simulation loop
    res_finite = functions_seqs.Mz_se(R1, v, Fw, j, me, TE, TR=2.0, FA=FA)
    assert res_finite.shape == (2, 3)


def test_Mz_spgr_in_ss():
    """Test Mz_spgr_in_ss shortcut branch and standard loop shape."""
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, FA = 1.0, 15.0
    
    # Case 1: Infinite TR shortcut
    res_inf = functions_seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, np.inf, FA)
    np.testing.assert_array_almost_equal(res_inf, np.full_like(R1, me))
    
    # Case 2: Running actual underlying pulse simulation loop
    res_finite = functions_seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, TR=0.1, FA=FA)
    assert res_finite.shape == (2, 3)


def test_Mz_pr_spgr():
    """Test Mz_pr_spgr loop tracking and shape parsing."""
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TC, TR, FA, TP, TA, PA = 1.0, 0.2, 0.01, 15.0, 0.05, 0.1, 90.0
    
    res = functions_seqs.Mz_pr_spgr(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA)
    assert res.shape == (2, 3)


def test_Mz_pr_spgr_in_ss():
    """Test steady-state preparation SPGR execution loop and output shape."""
    R1 = np.ones((2, 3))
    j = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    me, TC, TR, FA, TP, TA, PA = 1.0, 0.2, 0.01, 15.0, 0.05, 0.1, 90.0
    
    res = functions_seqs.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA)
    assert res.shape == (2, 3)


def test_Mz_ssi():
    """Test Mz_ssi behavior handling both 1-compartment and multi-compartment execution blocks."""
    Fw, me, TR, FA, TF, SA = 1.0, 1.0, 0.01, 15.0, 0.05, 90.0
    
    # Case 1: Single Compartment branch (nc == 1)
    R1_1d = np.ones((1, 3))
    j_1d = np.ones((1, 3))
    v = 1
    res_1d = functions_seqs.Mz_ssi(R1_1d, v, Fw, j_1d, me, TR, FA, TF, SA)
    assert res_1d.shape == (1, 3)
    
    # Case 2: Multi-Compartment branch (nc == 2)
    R1_2d = np.ones((2, 3))
    j_2d = np.ones((2, 3))
    v = np.ones(2) / 2
    Fw = np.identity(2)
    res_2d = functions_seqs.Mz_ssi(R1_2d, v, Fw, j_2d, me, TR, FA, TF, SA)
    assert res_2d.shape == (2, 3)

if __name__=='__main__':
    test_mz_readout()
    test_signal_rice()
    test_Mz_se()
    test_Mz_spgr_in_ss()
    test_Mz_pr_spgr()
    test_Mz_pr_spgr_in_ss()
    test_Mz_ssi()

    print('All bloch.seqs tests passed!')