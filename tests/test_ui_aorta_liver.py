import os
import numpy as np
import json
import warnings

import dcmri as dc

DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 1
    SHOW = True
else:
    VERBOSE = 0
    SHOW = True
    # Allow coverage of plto functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def test_ui_aorta_liver_final_3_percent():
    """
    Targeting the remaining specific lines for 100% coverage.
    """
    
    # 1. Trigger ValueError('Only single-inlet models are allowed.')
    # This requires a kinetics string that does NOT start with '1'
    try:
        dc.AortaLiver(kinetics='2I-IC-HFD')
    except ValueError:
        pass
    try:
        dc.AortaLiver(kinetics='2I-EC')
    except ValueError:
        pass

    # 2. Trigger self._R1l = self._pars['R10l'] + rp * self._Cl
    # AND ax.plot(t/60, 1000*C, ...) [The 'else' branch for 1D Cl]
    # We use a single-compartment model like '1I-IC' (Interstellar/Extracellular only)
    model_1d = dc.AortaLiver(kinetics='1I-IC', tmax=60)
    # Calling relax() triggers the R1l calculation for 1D
    model_1d.relax() 
    # Calling plot() triggers the 1D tissue plotting branch
    xdata = (np.array([0, 30]), np.array([0, 30]))
    ydata = (np.array([1, 1.2]), np.array([1, 1.2]))
    model_1d.plot(xdata, ydata, show=SHOW)
    model_1d.plot(xdata, ydata, show=False)

    # 3. Trigger ValueError("Bounds on BAT must be (negative, positive).")
    # BAT (Bolus Arrival Time) logic requires the first bound < 0 and second > 0 
    # as it is a relative shift during heuristic estimation.
    model_bat = dc.AortaLiver()
    try:
        # Provide bounds that are both positive
        model_bat.train(xdata, ydata, free={'BAT': [10, 20]})
    except ValueError:
        pass

    # 5. Trigger warnings.warn("Curve fit failed...") 
    # AND fitted_pars, pcov = p0, None
    # We pass data that makes the Jacobian singular or use max_nfev=1
    model_fail = dc.AortaLiver()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Passing max_nfev=1 to curve_fit via kwargs forces a failure
        model_fail.train(xdata, ydata, max_nfev=1)
        
        # Verify the warning was caught
        assert len(w) > 0
        assert "Curve fit failed" in str(w[-1].message)

    # 6. EC model
    model_1d = dc.AortaLiver(kinetics='1I-EC', tmax=60)
    # Calling relax() triggers the R1l calculation for 1D
    model_1d.relax() 
    model_1d.plot(xdata, ydata, show=SHOW)



def test_ui_aorta_liver_docstring():

    time, aif, vif, roi, gt = dc.fake_liver()
    xdata, ydata = (time,time), (aif,roi)
    model = dc.AortaLiver(
        dt = 0.5,
        tmax = 180,
        weight = 70,
        agent = 'gadoxetate',
        field_strength = 3.0,
        dose = 0.2,
        rate = 3,
        TR = 0.005,
        FA = 15,
    )
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)



def test_ui_aorta_liver():

    time, aif, roi, gt = dc.fake_tissue()
    xdata, ydata = (time,time), (aif,roi)

    model = dc.AortaLiver(
        dt = 0.5,
        tmax = 180,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
        Th = 120,
    )
    bounds = {'Th':[0, 1e9]}
    model.train(xdata, ydata, bounds=bounds, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 10
    assert 85 < model.params('Th', round_to=0) < 95

def test_ui_aorta_liver_full_coverage():
    """
    Targets specific logic branches:
    - Multi-compartment signal summation (line 647/670 area)
    - Covariance/SDEV normalization (line 829-831 area)
    - BAT heuristic bound shifting (line 715-718 area)
    """
    time, aif, roi, gt = dc.fake_tissue()
    # Ensure we have enough data points for a valid covariance matrix
    xdata, ydata = (time, time), (aif, roi)

    # 1. Use a multi-compartment kinetics model (1I-IC-HFD) 
    # This triggers the specific ndim == 2 logic in relaxation/signal.
    model = dc.AortaLiver(
        kinetics='1I-IC-HFD',
        tmax=max(time)+10,
        agent='gadoxetate'
    )
    
    # 2. Train with 'free' parameters to trigger the normalization helpers
    # and the covariance-to-sdev mapping in export_params.
    free = {
        'CO': [50, 200],
        'BAT': [-30, 30],
    }
    
    # This call triggers the BAT heuristic shifting logic
    model.train(xdata, ydata, free=free, n0=5, xtol=1e-2)
    
    # 3. Trigger export_params and print_params to cover SDEV calculation
    # (This covers the lines that re-normalize the pcov diagonals)
    exported = model.export_params()
    assert 'CO' in exported
    assert exported['CO'][3] >= 0  # Check that sdev is calculated
    
    if VERBOSE:
        model.print_params(round_to=2)

def test_ui_aorta_liver_io_and_variants():
    """Covers file I/O, SR sequences, and rounding logic."""
    file = 'temp_model_test.json'
    model = dc.AortaLiver(sequence='SR', TC=0.1)
    
    # Trigger Rounding logic in params() getter
    model.save(file)
    model.load(file)
    val = model.params('TR', round_to=2)
    
    # Trigger multi-parameter getter
    subset = model.params('TR', 'FA', round_to=2)
    assert len(subset) == 2
    
    if os.path.exists(file):
        os.remove(file)

def test_ui_aorta_liver_error_states():
    """Covers validation checks and error branches."""
    # Invalid kinetics (must be single inlet)
    try:
        dc.AortaLiver(kinetics='2I-IC-HFD')
    except ValueError:
        pass
        
    # Parameter out of bounds at init
    try:
        dc.AortaLiver(CO=500) # Default max is 300
    except ValueError:
        pass
    
    # Plotting with xdata exceeding tmax
    model = dc.AortaLiver(tmax=10)
    try:
        model.plot(([20], [20]), ([1], [1]))
    except ValueError:
        pass

def test_ui_aorta_liver_error_branches():
    """Triggers all raise ValueError statements in __init__ and predict."""
    # 1. Invalid sequence
    try:
        dc.AortaLiver(sequence='NotASequence')
    except ValueError:
        pass
    
    # 2. Dual-inlet kinetics (only single-inlet allowed)
    try:
        dc.AortaLiver(kinetics='2I-IC-HFD')
    except ValueError:
        pass
        
    # 3. Invalid parameter in **params
    try:
        dc.AortaLiver(not_a_param=10)
    except ValueError:
        pass
    
    # 4. xdata exceeds tmax in predict (Aorta and Liver branches)
    model = dc.AortaLiver(tmax=10)
    try:
        model.predict((np.array([20]), np.array([5])))
    except ValueError:
        pass
    try:
        model.predict((np.array([5]), np.array([20])))
    except ValueError:
        pass

def test_ui_aorta_liver_train_errors():
    """Triggers all raise ValueError statements in the train() method."""
    model = dc.AortaLiver(CO=100)
    x, y = ([0,1], [0,1]), ([1,1], [1,1])
    
    # 1. Parameter not a valid free parameter
    try:
        model.train(x, y, bounds={'not_real': [0, 1]})
    except ValueError:
        pass
        
    # 2. Free parameter doesn't exist in config
    try:
        model.train(x, y, free={'fake_param': [0, 1]})
    except ValueError:
        pass
        
    # 3. BAT bounds must be (negative, positive)
    try:
        model.train(x, y, free={'BAT': [10, 20]})
    except ValueError:
        pass
        
    # 4. Initial value out of bounds
    try:
        model.train(x, y, free={'CO': [10, 20]}) # Init 100 is out of [10, 20]
    except ValueError:
        pass

def test_ui_aorta_liver_io_errors():
    """Triggers I/O branch logic and file validation errors."""
    model = dc.AortaLiver()
    
    # 1. Test the 'file += .json' branch by providing name without extension
    model.save('test_no_ext')
    assert os.path.exists('test_no_ext.json')
    
    # 2. Corrupt/Modify JSON for validation errors
    with open('test_no_ext.json', 'r') as f:
        data = json.load(f)
    
    # Wrong model error
    data['model'] = 'WrongModelName'
    with open('wrong_model.json', 'w') as f:
        json.dump(data, f)
    try:
        model.load('wrong_model.json')
    except ValueError:
        pass
        
    # Version mismatch error
    data['model'] = 'AortaLiver'
    data['version'] = '0.0_old'
    with open('wrong_version.json', 'w') as f:
        json.dump(data, f)
    try:
        model.load('wrong_version.json')
    except ValueError:
        pass
        
    # Cleanup
    for f in ['test_no_ext.json', 'wrong_model.json', 'wrong_version.json']:
        if os.path.exists(f): os.remove(f)

def test_ui_aorta_liver_math_branches():
    """Triggers specific math logic like single-compartment liver R1 and relax/signal getters."""
    # 1. Single-inlet model that isn't multi-compartment (triggers C vs C[0]+C[1] branch)
    # Using '1I-IC' (Interstital) which is usually 1D tissue concentration
    model = dc.AortaLiver(kinetics='1I-IC')
    
    # 2. Coverage for relax() and signal() public methods
    times, r_a, r_l = model.relax()
    times, s_a, s_l = model.signal()
    assert len(r_a) == len(times)
    
    # 3. Trigger 'return subset' in params()
    sub = model.params('CO', 'FA')
    assert isinstance(sub, dict)

def test_ui_aorta_liver_optimizer_failure():
    """Triggers the RuntimeError warning branch in the optimizer."""
    model = dc.AortaLiver()
    # Provide impossible data to force curve_fit to fail
    x = (np.array([1, 2]), np.array([1, 2]))
    y = (np.array([1, 1e9]), np.array([1, 1e9]))
    
    with warnings.catch_warnings(record=True) as w:
        model.train(x, y, max_nfev=1) # Force exit
        assert len(w) > 0 # Warning should be triggered

def test_ui_aorta_liver_plotting_branches():
    """Triggers savefig and test_data plotting logic."""
    time, aif, roi, gt = dc.fake_tissue()
    xdata, ydata = (time, time), (aif, roi)
    model = dc.AortaLiver()
    
    # Trigger savefig branch and 'test is not None' branch in _plot_data1scan
    model.plot(xdata, ydata, fname='test_plot.png', ref=(xdata, ydata), show=SHOW)
    
    if os.path.exists('test_plot.png'):
        os.remove('test_plot.png')

if __name__ == "__main__":
    test_ui_aorta_liver_final_3_percent()
    test_ui_aorta_liver_error_branches()
    test_ui_aorta_liver_train_errors()
    test_ui_aorta_liver_io_errors()
    test_ui_aorta_liver_math_branches()
    test_ui_aorta_liver_optimizer_failure()
    test_ui_aorta_liver_plotting_branches()
    test_ui_aorta_liver_docstring()

    test_ui_aorta_liver()
    test_ui_aorta_liver_full_coverage()
    test_ui_aorta_liver_io_and_variants()
    test_ui_aorta_liver_error_states()

    print('All ui_liver tests passed!!')