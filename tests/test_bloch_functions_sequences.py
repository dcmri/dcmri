import numpy as np

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
    
    result = dc.mz_readout(Mz, R2, FA, TE)
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
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, FA, TR, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, FA, TR, 2 * TP + 2 * (TC-TP) )

    # No exchange
    Fw = np.array([[1,0], [0,4]])
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # Fast exchange
    Fw = np.array([[1,np.inf], [np.inf,4]])
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    # Not implemented
    try:
        Fw = np.array([[1,np.inf], [0,4]])
        ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)
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
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, FA, TR, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6

    # Extend coverage

    # Include wait time at the end
    dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, FA, TR, 2 * TP + 2 * (TC-TP) )

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
    ss_approx_3, _ = dc.Mz_prop_pr_spgr(M0, R1, v, Fw, j, me, 0, TP, TC, FA, TR, TP + 2 * (TC-TP))
    ss_approx_4 = dc.Mz_ss_pr_spgr(R1, v, Fw, j, me, FA, TR, Nph, TP, TD, 0)
    ss_exact = dc.Mz_ss_spgr(R1, v, Fw, j, me, FA, TR)

    assert np.linalg.norm(ss_approx_1 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_2 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_3 - ss_exact) < 1e-6
    assert np.linalg.norm(ss_approx_4 - ss_exact) < 1e-6


def test_vectorization():

    me = 5
    FA = 15
    TR = 0.01
    TF = 0.50
    SA = 60
    Nph = 512
    seq = [[FA, TR] for _ in range(Nph)]

    # Two compartments - ss

    nc, nt = 2, 10
    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))
    v = np.array([0.1, 0.4])
    Kw = np.array([[1,2], [3,4]])
    M0 = np.array([1,1])
                              
    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss(R1[:, i], v, Kw, j[:, i], me, seq)

    M_2 = dc.Mz_ss_vectorized(R1, v, Kw, j, me, seq)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR)

    M_2 = dc.Mz_ss_spgr_vectorized(R1, v, Kw, j, me, FA, TR)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_spgri(R1[:, i], v, Kw, j[:, i], me, FA, TR, TF, SA)

    M_2 = dc.Mz_ss_spgri_vectorized(R1, v, Kw, j, me, FA, TR, TF, SA)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    # Two compartments - prop
    nt = Nph

    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))

    M_1 = dc.Mz_prop(M0, R1, v, Kw, j, me, seq)
    M_2 = dc.Mz_prop_vectorized(M0, R1, v, Kw, j, me, [[FA, TR]])

    assert np.linalg.norm(M_1 - M_2) < 1e-12


    # One compartment - ss

    nc, nt = 1, 10
    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))
    v = np.array([0.1])
    Kw = np.array([[1]])
                              
    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss(R1[:, i], v, Kw, j[:, i], me, seq)

    M_2 = dc.Mz_ss_vectorized(R1, v, Kw, j, me, seq)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR)

    M_2 = dc.Mz_ss_spgr_vectorized(R1, v, Kw, j, me, FA, TR)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR)

    M_2 = dc.Mz_ss_spgr_vectorized(R1, v, Kw, j, me, FA, TR)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_spgri(R1[:, i], v, Kw, j[:, i], me, FA, TR, TF, SA)

    M_2 = dc.Mz_ss_spgri_vectorized(R1, v, Kw, j, me, FA, TR, TF, SA)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    # One compartment - prop

    nt = Nph
    seq = [[FA, TR] for _ in range(Nph)]

    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))

    M_1 = dc.Mz_prop(M0, R1, v, Kw, j, me, seq)
    M_2 = dc.Mz_prop_vectorized(M0, R1, v, Kw, j, me, [[FA, TR]])

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = dc.Mz_prop_vectorized(M0, R1, v, Kw, j, me, Nph * [[FA, TR]])
    M_2 = dc.Mz_prop_spgr_vectorized(M0, R1, v, Kw, j, me, FA, TR, Nph)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    PA = 60
    TP = 0.2
    TD = 0.1
    Nk0 = 100

    pulses_per_period_from_k0 = (Nph - Nk0 - 1) * [[FA, TR]] + [[FA, TR + TD]] + [[PA, TP]] + Nk0 * [[FA, TR]]

    M_1 = dc.Mz_prop_vectorized(M0, R1, v, Kw, j, me, pulses_per_period_from_k0)
    M_2 = dc.Mz_prop_pr_spgr_vectorized(M0, R1, v, Kw, j, me, FA, TR, Nph, TP, TD, PA, Nk0)

    assert np.linalg.norm(M_1 - M_2) < 1e-12


def test_Mz_ss_k0_pr_spgr():
    me = 5
    FA = 15
    TR = 0.01
    TP = 0.25
    TD = 0.10
    PA = 120
    Nph = 512
    Nk0 = 128
    pulses_to_k0 = [[PA, TP]] + Nk0 * [[FA, TR]]

    # Two compartments - ss at start vs center

    nc, nt = 2, 10
    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))
    v = np.array([0.1, 0.4])
    Kw = np.array([[1,2], [3,4]])
                              
    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA)
        M_1[:, i] = dc.Mz_prop(M_1[:, i], R1[:, i], v, Kw, j[:, i], me, pulses_to_k0)[:, -1]

    M_2 = np.zeros((nc, nt))
    for i in range(nt):
        M_2[:, i] = dc.Mz_ss_k0_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA, Nk0)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    # Two compartments - vectorized vs iterative

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA)

    M_2 = dc.Mz_ss_pr_spgr_vectorized(R1, v, Kw, j, me, FA, TR, Nph, TP, TD, PA)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_k0_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA, Nk0)

    M_2 = dc.Mz_ss_k0_pr_spgr_vectorized(R1, v, Kw, j, me, FA, TR, Nph, TP, TD, PA, Nk0)

    assert np.linalg.norm(M_1 - M_2) < 1e-12


    # One compartment - ss at start vs center


    nc, nt = 1, 10
    R1 = np.arange(nc * nt).reshape((nc, nt))
    j = 2 * np.arange(nc * nt).reshape((nc, nt))
    v = np.array([0.1])
    Kw = np.array([[1]])
                              
    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA)
        M_1[:, i] = dc.Mz_prop(M_1[:, i], R1[:, i], v, Kw, j[:, i], me, pulses_to_k0)[:, -1]

    M_2 = np.zeros((nc, nt))
    for i in range(nt):
        M_2[:, i] = dc.Mz_ss_k0_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA, Nk0)

    assert np.linalg.norm(M_1 - M_2) < 1e-12


    # One compartment - vectorized vs iterative

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA)

    M_2 = dc.Mz_ss_pr_spgr_vectorized(R1, v, Kw, j, me, FA, TR, Nph, TP, TD, PA)

    assert np.linalg.norm(M_1 - M_2) < 1e-12

    M_1 = np.zeros((nc, nt))
    for i in range(nt):
        M_1[:, i] = dc.Mz_ss_k0_pr_spgr(R1[:, i], v, Kw, j[:, i], me, FA, TR, Nph, TP, TD, PA, Nk0)

    M_2 = dc.Mz_ss_k0_pr_spgr_vectorized(R1, v, Kw, j, me, FA, TR, Nph, TP, TD, PA, Nk0)

    assert np.linalg.norm(M_1 - M_2) < 1e-12




if __name__=='__main__':
    test_mz_readout()
    test_nc_function()
    test_1c_function()
    test_1c_scalar_function()
    test_vectorization()
    test_Mz_ss_k0_pr_spgr()

    print('All bloch.functions_sequences tests passed!')