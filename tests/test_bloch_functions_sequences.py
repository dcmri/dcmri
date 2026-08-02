import numpy as np

from dcmri.bloch import functions_sequences
import dcmri as dc

def test_mz_readout():
    """Test mz_readout outputs the correct array shape and values."""
    Mz = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    R2 = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    FA = 90.0  # sin(90) = 1.0
    TE = 0.0   # exp(0) = 1.0
    
    # Expected: Mxy = 1.0*1.0*1.0 + 1.0*1.0*0.5 = 1.5 per timepoint
    # signal = 100 * 1.5 = 150
    expected = np.array([[1. , 1. , 1. ],
                         [0.5, 0.5, 0.5]])
    
    result = functions_sequences.mz_readout(Mz, R2, FA, TE)
    np.testing.assert_array_almost_equal(result, expected)


def test_nc_function():
    # Compare two ways of computing SPGR SS
    # pulse.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = np.array([1,2])
    v = np.array([0.1, 0.4])
    Fw = np.array([[1,2], [3,4]])
    j = np.array([3,4])
    M0 = np.array([1,1])
                                
    me = 5
    FA = 15
    TR = 0.01
    Nph = 512
    seq = [[FA, TR] for _ in range(Nph)]
    TC = Nph * TR
    TP = 0.1
    TD = 0

    ss_approx_1 = dc.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = dc.Mz_prop(M0, R1, v, Fw, j, me, seq)[:, -1]
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, 2 * TP + 2 * (TC-TP) )

    # No exchange
    Fw = np.array([[1,0], [0,4]])
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Fast exchange
    Fw = np.array([[1,np.inf], [np.inf,4]])
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Not implemented
    try:
        Fw = np.array([[1,np.inf], [0,4]])
        ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)
    except NotImplementedError:
        pass
    else:
        assert False

def test_1c_function():
    # Compare two ways of computing SPGR SS
    # pulse.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = np.array([3])
    v = np.array([0.6])
    Fw = np.array([0.3]).reshape(1,1)
    j = np.array([5])
    M0 = np.array([0.8])

    me = 2
    FA = 15
    TR = 0.01
    Nph = 512
    seq = [[FA, TR] for _ in range(Nph)]
    TC = Nph * TR
    TP = 0.5
    TD = 0

    ss_approx_1 = dc.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = dc.Mz_prop(M0, R1, v, Fw, j, me, seq)[:, -1]
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, 2 * TP + 2 * (TC-TP) )

def test_1c_scalar_function():
    # Compare two ways of computing SPGR SS
    # pulse.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = 3
    v = 0.6
    Fw = 0.3
    j = 5
    M0 = 0.8

    me = 2
    FA = 15
    TR = 0.01
    Nph = 512
    seq = [[FA, TR] for _ in range(Nph)]
    TC = Nph * TR
    TP = 0.5
    TD = 0

    ss_approx_1 = dc.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = dc.Mz_prop(M0, R1, v, Fw, j, me, seq)[:, -1]
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, TR, FA, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6


def test_complete_coverage():
    functions_sequences._Mz_ss_aex(np.zeros(2), np.ones(2), np.zeros((2,2)), 0, 1, 1, 1)



if __name__=='__main__':
    test_mz_readout()
    test_nc_function()
    test_1c_function()
    test_1c_scalar_function()
    test_complete_coverage()

    print('All bloch.functions_sequences tests passed!')