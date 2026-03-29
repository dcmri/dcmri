import numpy as np

from dcmri import mz_lib


def test_nc_function():
    # Compare two ways of computing SPGR SS
    # mz_lib.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = np.array([1,2])
    v = np.array([0.1, 0.4])
    Fw = np.array([[1,2], [3,4]])
    j = np.array([3,4])
    M0 = np.array([1,1])
                                
    me = 5
    FA = 15
    TR = 0.01
    seq = [[FA, TR] for _ in range(512)]
    TC = 512 * TR
    TP = 0.1

    ss_approx_1 = mz_lib.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = mz_lib.Mz_prop(M0, R1, v, Fw, j, me, seq)
    ss_approx_3, _ = mz_lib.Mz_pr_spgr_prop(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = mz_lib.Mz_pr_spgr_ss(R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    mz_lib.Mz_pr_spgr_prop(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, 2 * TP + 2 * (TC-TP) )

    # No exchange
    Fw = np.array([[1,0], [0,4]])
    ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Fast exchange
    Fw = np.array([[1,np.inf], [np.inf,4]])
    ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    # Not implemented
    try:
        Fw = np.array([[1,np.inf], [0,4]])
        ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)
    except NotImplementedError:
        pass
    else:
        assert False

def test_1c_function():
    # Compare two ways of computing SPGR SS
    # mz_lib.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = np.array([3])
    v = np.array([0.6])
    Fw = np.array([0.3]).reshape(1,1)
    j = np.array([5])
    M0 = np.array([0.8])

    me = 2
    FA = 15
    TR = 0.01
    seq = [[FA, TR] for _ in range(512)]
    TC = 512 * TR
    TP = 0.5

    ss_approx_1 = mz_lib.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = mz_lib.Mz_prop(M0, R1, v, Fw, j, me, seq)
    ss_approx_3, _ = mz_lib.Mz_pr_spgr_prop(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = mz_lib.Mz_pr_spgr_ss(R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    mz_lib.Mz_pr_spgr_prop(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, 2 * TP + 2 * (TC-TP) )

def test_1c_scalar_function():
    # Compare two ways of computing SPGR SS
    # mz_lib.Mz_ss(R1, v, Fw, j, me, seq)

    R1 = 3
    v = 0.6
    Fw = 0.3
    j = 5
    M0 = 0.8

    me = 2
    FA = 15
    TR = 0.01
    seq = [[FA, TR] for _ in range(512)]
    TC = 512 * TR
    TP = 0.5

    ss_approx_1 = mz_lib.Mz_ss(R1, v, Fw, j, me, seq)
    ss_approx_2 = mz_lib.Mz_prop(M0, R1, v, Fw, j, me, seq)
    ss_approx_3, _ = mz_lib.Mz_pr_spgr_prop(M0, R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_approx_4 = mz_lib.Mz_pr_spgr_ss(R1, v, Fw, j, me, 0, TP, TC, TR, FA, TP + 2 * (TC-TP))
    ss_exact = mz_lib.Mz_ss_spgr(R1, v, Fw, j, me, TR, FA)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6



if __name__=='__main__':

    test_nc_function()
    test_1c_function()
    test_1c_scalar_function()

    print('All mz_lib tests passed!')