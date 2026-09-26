import numpy as np

import dcmri as dc
from dcmri.relaxivity.functions_relaxivity import mix_fast_exchange


def test_conc_t1():
    R1 = np.arange(10) + 5
    c = dc.conc_t1(R1, 0.1)
    assert np.sum(c) == 450.0
    R1 = np.stack((R1, 2*R1))
    c = dc.conc_t1(R1, [5,4])
    assert np.sum(c) == 31.5
    c = dc.conc_t1(R1, 5)
    assert np.sum(c) == 27


def test_relax_t1():

    # One compartment
    #################
    r1 = 1

    # One time point - ROI
    R1b = 1
    c = 1
    assert dc.relax_t1(c, R1b, r1) == 2

    # One time point - image
    shape = (3,3)
    R1b = np.ones(shape)
    c = np.ones(shape)
    assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full(shape,2))

    # 4 time points - ROI
    R1b = 1
    c = np.ones(4)
    assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full((4,),2))

    # # 4 time points - image
    # shape = (3,3,4)
    # R1b = np.ones(shape[:2])
    # c = np.ones(shape)
    # assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full(shape,2))

    # Two compartments
    ##################
    r1 = [1,1]

    # One time point - ROI
    R1b = [1,1]
    c = [1,1]
    assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full((2,),2))

    # # One time point - image
    # shape = (2,3,3)
    # R1b = np.ones(shape)
    # c = np.ones(shape)
    # assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full(shape,2))

    # # 4 time points - ROI
    # R1b = [1,1]
    # c = np.ones((2,4))
    # assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full((2,4),2))

    # # 4 time points - image
    # shape = (2,3,3,4)
    # R1b = np.ones(shape[:3])
    # c = np.ones(shape)
    # assert np.array_equal(dc.relax_t1(c, R1b, r1), np.full(shape,2))


def test_relax_t2s():
    """Test all models and shapes for the relax_t2s function."""
    
    # --- Case 1: 'lin' model with scalar values ---
    c_scalar = 0.5
    R2sb_scalar = 1.0
    r2s_scalar = 2.0
    expected_scalar = 1.0 + 2.0 * 0.5  # 2.0
    
    result_scalar = dc.relax_t2s(c_scalar, R2sb_scalar, r2s=r2s_scalar, model='lin')
    assert np.isclose(result_scalar, expected_scalar)

    # --- Case 2: 'lin' model with a 1D array ---
    c_1d = np.array([0.1, 0.2, 0.3])
    R2sb_1d = 1.5
    r2s_1d = 2.5
    expected_1d = 1.5 + 2.5 * c_1d
    
    result_1d = dc.relax_t2s(c_1d, R2sb_1d, r2s=r2s_1d, model='lin')
    np.testing.assert_array_almost_equal(result_1d, expected_1d)

    # --- Case 3: 'quad' model with linear and quadratic terms ---
    c_quad = np.array([1.0, 2.0])
    R2sb_quad = 0.8
    r2s_quad_lin = 1.5
    r2s_quad_term = 0.3
    expected_quad = R2sb_quad + r2s_quad_lin * c_quad + r2s_quad_term * c_quad**2
    
    result_quad = dc.relax_t2s(c_quad, R2sb_quad, r2s=r2s_quad_lin, r2s_quad=r2s_quad_term, model='quad')
    np.testing.assert_array_almost_equal(result_quad, expected_quad)

    # --- Case 4: 'leakage' model with a 2D multi-compartment array ---
    c_2d = np.array([
        [0.5, 0.8, 0.2],  # Compartment 0
        [0.1, 0.4, 0.9]   # Compartment 1
    ])
    R2sb_leak = 1.2
    r2s_vasc = 3.0
    r2s_ees = 1.5
    expected_leak = 1.2 + 3.0 * np.abs(c_2d[0,:] - c_2d[1,:]) + 1.5 * c_2d[1,:]
    
    result_leak = dc.relax_t2s(c_2d, R2sb_leak, r2s_vasc=r2s_vasc, r2s_ees=r2s_ees, model='leakage')
    np.testing.assert_array_almost_equal(result_leak, expected_leak)


def test_relax_t2():
    """Test the valid linear path and the error path for the relax_t2 function."""
    
    # --- Case 1: Valid 'lin' model array calculation ---
    c = np.array([0.0, 1.5, 3.0])
    R2b = 0.5
    r2 = 4.0
    expected = 0.5 + 4.0 * c
    
    result = dc.relax_t2(c, R2b, r2=r2, model='lin')
    np.testing.assert_array_almost_equal(result, expected)
    
    # --- Case 2: Ensure an error is thrown for an invalid model name ---
    try:
        dc.relax_t2(c, R2b, r2=2.0, model='invalid_model_name')
        # If the line above doesn't throw an error, force the test to fail
        assert False, "relax_t2 should have raised a ValueError for an invalid model."
    except ValueError as e:
        # The test passes if the correct error message is caught
        assert 'Model invalid_model_name not recognized' in str(e)


def test_mix_fast_exchange():
    nt = 5
    v = [0.1, 0.4, 0.5]
    R = np.vstack((
        3 * np.ones(nt),
        4 * np.ones(nt),
        5 * np.ones(nt)
    ))
    fx = [[0,1]]
    v, R = mix_fast_exchange(v, R, fx)
    assert v.size == 2
    assert R.shape == (2, nt)
    assert v[0] == 0.5
    assert R[0,0] == (0.1 * 3 + 0.4 * 4) / 0.5
    assert R[1,0] == 5

    fx = [[0,1], [1]]
    try:
        v, R = mix_fast_exchange(v, R, fx)
    except:
        pass
    else:
        assert False


if __name__ == "__main__":

    test_conc_t1()
    test_relax_t1()
    test_mix_fast_exchange()

    print('All relaxivity tests passing!')