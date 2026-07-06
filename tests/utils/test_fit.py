import pytest
import numpy as np
import dcmri.utils.fit as mm  

# =====================================================================
# 1. Tests for Math Utility Functions (normalize, renormalize)
# =====================================================================

def test_normalize():
    bounds = (10.0, 20.0)
    # Midpoint should be 0.5
    assert mm.normalize(15.0, bounds) == 0.5
    # Lower bound should be 0.0
    assert mm.normalize(10.0, bounds) == 0.0
    # Upper bound should be 1.0
    assert mm.normalize(20.0, bounds) == 1.0

def test_renormalize():
    bounds = (10.0, 20.0)
    # 0.5 should scale back to 15.0
    assert mm.renormalize(0.5, bounds) == 15.0
    assert mm.renormalize(0.0, bounds) == 10.0
    assert mm.renormalize(1.0, bounds) == 20.0


# =====================================================================
# 2. Tests for Internal Helper Functions
# =====================================================================

def test__compute_normalized_pars():
    pars = {'alpha': 15.0, 'beta': 30.0}
    free = {'alpha': (10.0, 20.0)}  # beta is fixed, alpha is free
    
    # Test without index x
    res = mm._compute_normalized_pars(pars, free, x=None)
    assert res == [0.5]  # 15.0 normalized between 10 and 20

    # Test with index x (array parameters)
    pars_arr = {'alpha': np.array([10.0, 15.0, 20.0])}
    res_x = mm._compute_normalized_pars(pars_arr, free, x=1)
    assert res_x == [0.5]

def test__update_original_pars():
    pars = {'alpha': 0.0}
    free = {'alpha': (10.0, 20.0)}
    
    # 0.5 normalized -> 15.0 original
    mm._update_original_pars(pars, [0.5], free, x=None)
    assert pars['alpha'] == 15.0

    # Array parameter update at index x
    pars_arr = {'alpha': np.array([0.0, 0.0, 0.0])}
    mm._update_original_pars(pars_arr, [0.5], free, x=1)
    assert pars_arr['alpha'][1] == 15.0

def test__sdev():
    pcov = np.array([[0.25]]) # sqrt(0.25) = 0.5
    free = {'param1': (0.0, 10.0)}
    # sdev should be renormalize(0.5, (0.0, 10.0)) -> 0.5 * 10 + 0 = 5.0
    res = mm._sdev(pcov, free)
    assert res['param1'] == 5.0


# =====================================================================
# 3. Tests for Optimization Core (train)
# =====================================================================

def test_train():
    # Mock prediction function: linear model y = mx
    def mock_predict(time):
        return time * 2.0  

    time = np.array([1.0, 2.0, 3.0])
    signal = np.array([2.0, 4.0, 6.0])
    pars = {'m': 1.5}
    free = {'m': (0.0, 5.0)}

    # Empty free parameters guard check
    vals, sdev, pcov = mm.train(mock_predict, time, signal, pars, free={})
    assert vals is None

    # Normal functional run
    vals, sdev, pcov = mm.train(mock_predict, time, signal, pars, free, reset=True)
    assert 'm' in vals
    # Verify parameter returned to state because reset=True
    assert pars['m'] == 1.5 

    # force training failure
    def mock_failing_predict(time):
        raise RuntimeError("Simulated curve_fit failure")
    mm.train(mock_failing_predict, time, signal, pars, free)

    # Tuple data
    def mock_predict(time):
        return (time[0] * 2.0, time[1] * 3.0)
    vals, sdev, pcov = mm.train(mock_predict, (time, time), (signal, signal), pars, free=free, sigma=(np.ones(3), np.ones(3)))




# =====================================================================
# 4. Tests for Batch Formatting & Pipelines
# =====================================================================

def test_format_batch_training():
    # Mock output tracking tuple structure (vals, sdev, pcov, optional model)
    mock_results = [
        ({'p1': 5.0}, {'p1': 0.1}, [[0.01]], "model_data_1"),
        ({'p1': 6.0}, {'p1': 0.2}, [[0.04]], "model_data_2")
    ]
    free = {'p1': (0.0, 10.0)}
    vals, sdev, pcov, model = mm.format_batch_training(mock_results, free)
    
    assert np.array_equal(vals['p1'], np.array([5.0, 6.0]))
    assert np.array_equal(sdev['p1'], np.array([0.1, 0.2]))
    assert pcov[0] == [[0.01]]
    assert model[1] == "model_data_2"

    free = {'p2': (0.0, 10.0)}
    vals, sdev, pcov, model = mm.format_batch_training(mock_results, free)

    mock_results = [
        ({'p1': 5.0}, {'p1': 0.1}, [[0.01]]),
        ({'p1': 6.0}, {'p1': 0.2}, [[0.04]])
    ]
    free = {'p1': (0.0, 10.0)}
    vals, sdev, pcov, model = mm.format_batch_training(mock_results, free)

def test_train_batch():
    def mock_predict(time, x=None):
        return time * 2.0

    time = np.array([1, 2])
    signal = np.array([[2, 4]]) # Shape (1, 2) -> Single-pixel shortcut path
    pars = {'m': np.array([2])}
    free = {'m': [0.0, 4.0]}

    # Patch training execution logic
    results = mm.train_batch(mock_predict, time, signal, pars, free)
    assert len(results) == 1

    signal = np.ones((2,2)) # Shape (2, 2) -> Single-pixel shortcut path
    pars = {'m': np.array([2, 2])}
    free = {'m': [0.0, 4.0]}

    # Patch training execution logic
    results = mm.train_batch(mock_predict, time, signal, pars, free)
    assert len(results) == 2


# =====================================================================
# 5. Tests for Loss Calculations
# =====================================================================

def test_loss():
    ypred = np.array([2.0, 4.0])
    ydata = np.array([2.0, 5.0])

    # RMS Loss calculation checking
    assert mm.loss(ypred, ydata, metric='RMS') == pytest.approx(1.0)
    
    # Check that lack of nfree raises ValueError for AIC metrics
    with pytest.raises(ValueError):
        mm.loss(ypred, ydata, metric='AIC', nfree=None)
    
    # Check AIC with proper elements
    loss_val = mm.loss(ypred, ydata, metric='AIC', nfree=10)
    assert isinstance(loss_val, float)

    loss_val = mm.loss(ypred, ydata, metric='NRMS')
    assert isinstance(loss_val, float)

    loss_val = mm.loss(ypred, ydata, metric='cAIC', nfree=10)
    assert isinstance(loss_val, float)

    with pytest.raises(ValueError):
        mm.loss(ypred, ydata, metric='cAIC', nfree=None)

    loss_val = mm.loss(ypred, ydata, metric='BIC', nfree=10)
    assert isinstance(loss_val, float)

    with pytest.raises(ValueError):
        mm.loss(ypred, ydata, metric='BIC', nfree=None)

    # Ensure incorrect metrics break safely
    with pytest.raises(ValueError):
        mm.loss(ypred, ydata, metric='UNKNOWN')



if __name__ == '__main__':
    print("Running module tests...")
    
    # Math Utility Tests
    test_normalize()
    test_renormalize()
    
    # Internal Helper Tests
    test__compute_normalized_pars()
    test__update_original_pars()
    test__sdev()
    
    # Optimization Core & Pipeline Tests
    test_train()
    test_format_batch_training()
    test_train_batch()
    
    # Loss Calculation Tests
    test_loss()
    
    print("All tests passed successfully!")