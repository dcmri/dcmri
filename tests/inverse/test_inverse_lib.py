import numpy as np

import dcmri.inverse.lib as inv


def test_conc_dce():
    """Test conc_dce interpolation shapes, baseline norms, and TE overrides."""
    # Define a simple mock signal model that matches the expected signature
    def dummy_sn_model(R1, TE, R2s, S0, v, Fw, me, R1i, Fi):
        # Return something linearly increasing so np.interp works predictably
        return 0.5 + 0.1 * R1

    # Setup shape parameters: (nc=1 compartment, nt=3 timepoints)
    S = np.array([[10.0, 11.0, 12.0]])
    R10 = np.array([0.5])
    r1 = 4.5
    
    # Case 1: Run with automatic baseline normalization (S0=None, R20s=None)
    res_normalized = inv.conc_dce(dummy_sn_model, S, n0=1, R10=R10, S0=None, r1=r1, R20s=None)
    assert res_normalized.shape == (1, 3)

    # Case 2: Run with pre-defined S0 and explicit R20s / TE params dictionary
    res_explicit = inv.conc_dce(dummy_sn_model, S, n0=1, R10=R10, S0=np.array([10.0]), r1=r1, R20s=1.5, TE=0.02)
    assert res_explicit.shape == (1, 3)

    res_explicit = inv.conc_dce(dummy_sn_model, S, n0=1, R10=R10, S0=np.array([10.0]), r1=r1, R20s=1.5)
    res_explicit = inv.conc_dce(dummy_sn_model, S, n0=1, R10=R10, S0=None, r1=r1, R20s=np.array([1.5]))
    res_explicit = inv.conc_dce(dummy_sn_model, S, n0=1, R10=None, S0=np.array([10.0]), r1=r1, R20s=1.5)


def test_conc_dsc():
    """Test conc_dsc math conversions and division guard rail paths."""
    # Shapes: nc=2 compartments, nt=3 timepoints
    S = np.array([
        [10.0, 5.0, 2.0],
        [20.0, 10.0, 4.0]
    ])
    n0 = 1
    r2 = 2.0
    TE = 0.05

    # Check safe execution
    C = inv.conc_dsc(S, n0, r2, TE)
    assert C.shape == (2, 3)
    # The concentration should increase as the signal drops below baseline
    assert C[0, 2] > C[0, 0] 


def test_conc_ss():
    """Test conc_ss analytical steady-state signal inversion."""
    S = np.array([[10.0, 12.0, 14.0]])
    n0 = 1
    R10 = np.array([0.7])
    r1 = 4.0
    sequence_params = {
        'FA': 15.0,
        'B1corr': 1.0,
        'TR': 0.005,
        'TE': 0.002
    }

    # Case 1: Standard computation with auto baseline calculation
    C_auto = inv.conc_ss(S, n0=n0, R10=R10, S0=None, R20s=1.5, r1=r1, **sequence_params)
    assert C_auto.shape == (1, 3)

    # Case 2: Explicit baseline array injection profile override
    C_explicit = inv.conc_ss(S, n0=n0, R10=R10, S0=np.array([100.0]), R20s=None, r1=r1, **sequence_params)
    assert C_explicit.shape == (1, 3)

    C_explicit = inv.conc_ss(S, n0=n0, R10=None, S0=np.array([100.0]), R20s=None, r1=r1, **sequence_params)


def test_conc_dce_lin():
    """Test conc_dce_lin simplified linear relaxation estimation."""
    S = np.array([[5.0, 10.0, 15.0]])
    n0 = 1
    R10 = np.array([1.0])
    r1 = 3.5

    # Case 1: Compute without S0 to trigger the initialization path
    C_auto = inv.conc_dce_lin(S, n0=n0, R10=R10, S0=None, r1=r1)
    assert C_auto.shape == (1, 3)

    # Case 2: Direct calculation via pre-specified scaling factor
    C_explicit = inv.conc_dce_lin(S, n0=n0, R10=R10, S0=np.array([5.0]), r1=r1)
    assert C_explicit.shape == (1, 3)

    C_explicit = inv.conc_dce_lin(S, n0=n0, R10=None, S0=np.array([5.0]), r1=r1)


def test_vfa_nonlinear():
    """Test vfa_nonlinear parsing for 1D profiles, 2D images, and error exceptions."""
    tr = 0.005
    flip_angles = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
    alphas = np.deg2rad(flip_angles)
    
    # Clean target parameters
    target_r1 = 1.0
    target_s0 = 1000.0
    e1 = np.exp(-tr * target_r1)
    signals_1d = target_s0 * np.sin(alphas) * (1 - e1) / (1 - np.cos(alphas) * e1)

    # Case 1: Valid 1D array profile fit execution
    r1_fit, s0_fit = inv.vfa_nonlinear(signals_1d, flip_angles, tr)
    assert np.isclose(r1_fit, 1.0, atol=1e-1)
    assert np.isclose(s0_fit, 1000.0, atol=10.0)

    # Case 2: Shape validation error branch (mismatched sizes)
    try:
        inv.vfa_nonlinear(signals_1d, flip_angles[:-1], tr)
        assert False, "Should have raised ValueError due to length mismatch."
    except ValueError:
        pass

    # Case 3: Test the multi-dimensional pixel loop using a 2D array
    # signals_2d shape: (number_of_pixels, number_of_flip_angles) -> (2, 5)
    signals_2d = np.zeros((2, len(flip_angles)))
    signals_2d[0, :] = signals_1d
    signals_2d[1, :] = signals_1d
            
    r1_grid, s0_grid = inv.vfa_nonlinear(signals_2d, flip_angles, tr)
    assert r1_grid.shape == (2,)
    assert s0_grid.shape == (2,)


def test_vfa_linear():
    """Test vfa_linear regression slope handling, formatting boundaries, and grid shapes."""
    # Use the same linearized transformation logic to build a flawless clean array
    # y = mx + c where m = E1, x = S/tan(a), y = S/sin(a)
    tr = 0.005
    flip_angles = np.array([5.0, 10.0, 15.0, 20.0])
    alphas = np.deg2rad(flip_angles)
    target_e1 = np.exp(-tr * 1.2)  # target R1 = 1.2
    target_s0 = 500.0
    
    # Work backwards to get matching clean signals
    # y = target_e1 * x + target_s0 * (1 - target_e1)
    # S / sin(a) = target_e1 * S / tan(a) + target_s0 * (1 - target_e1)
    # S * (1/sin(a) - target_e1/tan(a)) = target_s0 * (1 - target_e1)
    denom = (1.0 / np.sin(alphas)) - (target_e1 / np.tan(alphas))
    signals_1d = (target_s0 * (1 - target_e1)) / denom

    # Case 1: Clean linear fit estimation matching expected targets
    r1_fit, s0_fit = inv.vfa_linear(signals_1d, flip_angles, tr)
    assert np.isclose(r1_fit, 1.2, atol=1e-2)
    assert np.isclose(s0_fit, 500.0, atol=1e-1)

    # Case 2: Validation boundary error (mismatched size profiles)
    try:
        inv.vfa_linear(signals_1d, flip_angles[:-1], tr)
        assert False, "Should have raised ValueError due to length mismatch."
    except ValueError:
        pass

    # Case 3: Flat data
    flat_signals = np.array([100.0, 100.0, 100.0, 100.0])
    custom_bounds = ([0.1, 10.0], [10.0, 2000.0])
    r1, s0 = inv.vfa_linear(flat_signals, flip_angles, tr, bounds=custom_bounds, verbose=0)

    # Case 4: Multidimensional image mapping block verification (e.g., 2 pixels)
    signals_2d = np.zeros((2, len(flip_angles)))
    signals_2d[0, :] = signals_1d
    signals_2d[1, :] = signals_1d
    
    r1_arr, s0_arr = inv.vfa_linear(signals_2d, flip_angles, tr)
    assert r1_arr.shape == (2,)
    assert s0_arr.shape == (2,)

    flat_signals = np.array([100.0])
    flip_angles = np.array([15.0])
    custom_bounds = ([0.1, 10.0], [10.0, 2000.0])
    r1, s0 = inv.vfa_linear(flat_signals, flip_angles, tr, bounds=custom_bounds, verbose=1)

# =========================================================================
    # EXTENSION: Unphysical Slope Coverage
    # =========================================================================
    
    custom_bounds = ([0.2, 50.0], [5.0, 1000.0])
    flip_angles = np.array([5.0, 10.0, 15.0, 20.0])
    # Case 5: Unphysical Slope >= 1 (e1 >= 1)
    # Signal that increases unrealistically fast with flip angle yields a steep slope >= 1.
    # This leads to a negative or undefined R1 because ln(e1) becomes >= 0.
    unphysical_high_signals = np.array([10.0, 50.0, 200.0, 800.0])
    r1_high, s0_high = inv.vfa_linear(
        unphysical_high_signals, flip_angles, tr, bounds=custom_bounds, verbose=1
    )
    # Assert it falls back exactly to the lower bounds provided
    assert r1_high == custom_bounds[0][0]
    assert s0_high == custom_bounds[0][1]

    # Case 6: Unphysical Slope <= 0 (e1 <= 0)
    # Signal that drops off far quicker than the SPGR equation expects yields a negative slope.
    # This leads to an undefined R1 because ln(e1) cannot evaluate a negative number.
    unphysical_low_signals = np.array([800.0, 200.0, 50.0, 10.0])
    r1_low, s0_low = inv.vfa_linear(
        unphysical_low_signals, flip_angles, tr, bounds=custom_bounds, verbose=1
    )
    # Assert it falls back exactly to the lower bounds provided
    assert r1_low == custom_bounds[0][0]
    assert s0_low == custom_bounds[0][1]

if __name__ == "__main__":
    test_conc_dce()
    test_conc_dsc()
    test_conc_ss()
    test_conc_dce_lin()
    test_vfa_nonlinear()
    test_vfa_linear()
    print('All inverse.lib tests passing!')