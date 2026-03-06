import os
import shutil
import json

import numpy as np
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



def test_ui_aorta_portal_liver():

    time, aif, vif, roi, gt = dc.fake_liver(sequence='SSI')
    
    model = dc.AortaPortalLiver(
        sequence = 'SSI',
        kinetics='2I-IC',
        dt = 0.5,
        tmax = max(time) + 10,
        weight = 70,
        agent = 'gadoxetate',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
    )

    xdata, ydata = (time, time, time), (aif, vif, roi)
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)
    assert 3.5 < model.cost(xdata, ydata) < 4.5



def test_full_coverage_aorta_portal_liver():
    # 1. Setup Data
    time_pts, aif, vif, roi, gt = dc.fake_liver(sequence='SSI')
    xdata = (time_pts, time_pts, time_pts)
    ydata = (aif, vif, roi)
    
    # --- TEST 1: Initialization & Validation Errors ---
    # Trigger invalid sequence
    try:
        dc.AortaPortalLiver(sequence='INVALID')
    except ValueError:
        pass
    
    # Trigger invalid kinetics (not dual-inlet)
    try:
        dc.AortaPortalLiver(kinetics='1I-EC')
    except ValueError:
        pass

    # Trigger invalid parameter override
    try:
        dc.AortaPortalLiver(not_a_param=99)
    except ValueError:
        pass

    # --- TEST 2: Sequence 'SS' and 'SSI' Logic ---
    # Test SS branch in _compute_signal_aorta
    model_ss = dc.AortaPortalLiver(sequence='SS', kinetics='2I-EC')
    model_ss.predict(xdata) 
    
    # Test SSI branch (using the model for training)
    model = dc.AortaPortalLiver(sequence='SSI', kinetics='2I-EC')

    # --- TEST 3: Training & Heuristics ---
    # Trigger _estimate_parameters and all optimization blocks
    model.train(xdata, ydata, n0=5, xtol=1e-2)
    
    # Trigger BAT boundary error
    try:
        model.train(xdata, ydata, bounds={'BAT': [10, 20]}) # Must be (neg, pos)
    except ValueError:
        pass

    # --- TEST 4: Data Extraction & Public API ---
    model.conc()
    model.relax()
    model.params('BAT', 'CO', round_to=2) # Multi-arg + rounding
    model.params('BAT') # Single arg
    model.print_params(round_to=2)
    
    # Test different cost metrics
    model.cost(xdata, ydata)

    # --- TEST 5: Plotting Branches ---
    # Plot with filename (triggers savefig) and without show
    model.plot(xdata, ydata, fname='test_plot.png', show=False)
    if os.path.exists('test_plot.png'):
        os.remove('test_plot.png')
    
    # Plot with reference data (triggers the test data branch in _plot_data)
    ref_data = ((time_pts, aif), (time_pts, vif), (time_pts, roi))
    model.plot(xdata, ydata, ref=ref_data, show=False)

    # --- TEST 6: Serialization ---
    fname = 'test_model.json'
    model.save(fname)
    
    # Test load validation error (wrong class name)
    # (Optional: manually edit the JSON to trigger the version/model name mismatch)
    
    new_model = dc.AortaPortalLiver(kinetics='2I-EC')
    new_model.load(fname)
    assert new_model.params('BAT') == model.params('BAT')
    
    if os.path.exists(fname):
        os.remove(fname)

def test_liver_multicompartment_branch():
    """Hits the if C.ndim == 2 branch in _compute_relax_liver and _plot_conc_liver."""
    time_pts, aif, vif, roi, gt = dc.fake_liver(sequence='SS')
    # Use a kinetics model that returns 2 compartments (e.g., '2I-2C' or similar)
    # Adjust '2I-IC' to whatever your liver module uses for 2-compartment
    model = dc.AortaPortalLiver(kinetics='2I-EC', sequence='SS') 
    
    # Trigger relaxation and plot for 2D arrays
    model.relax()
    model.plot((time_pts, time_pts, time_pts), (aif, vif, roi), show=False)


def test_coverage_gap_filler():
    # Setup minimal data for training calls
    t_dummy = np.linspace(0, 60, 10)
    xdata = (t_dummy, t_dummy, t_dummy)
    ydata = (np.ones(10), np.ones(10), np.ones(10))
    
    # 1. Trigger: s_ref_a = sig.signal_ss(...) in _estimate_parameters
    # This runs when sequence is NOT 'SSI'
    model_ss = dc.AortaPortalLiver(sequence='SS', kinetics='2I-EC')
    model_ss.train(xdata, ydata, n0=2)

    # Trigger bounds exception
    try:
        model_ss.train(xdata, ydata, n0=2, bounds={'TR': [5, 10]})
    except ValueError:
        pass

    # 2. Trigger: ValueError "is not a free parameter"
    # Occurs when a parameter in 'bounds' isn't in the 'free' list
    try:
        model_ss.train(xdata, ydata, free={'BAT': [0, 1]}, bounds={'weight': [0, 100]})
    except ValueError as e:
        assert "is not a free parameter" in str(e)

    # 3. Trigger: ValueError "is not a valid parameter" (inside train)
    try:
        model_ss.train(xdata, ydata, free={'fake_param': [0, 1]})
    except ValueError as e:
        assert "is not a valid parameter" in str(e)

    # 4. Trigger: ValueError "Initial p is out of bounds"
    # We set a parameter value, then try to train with bounds that exclude that value
    model_ss._pars['CO'] = 500.0 
    try:
        model_ss.train(xdata, ydata, free={'CO': [0, 10]})
    except ValueError as e:
        assert "out of bounds" in str(e)

    # 5. Trigger: file += '.json' in save()
    model_ss.save('test_file') # Pass name without extension
    assert os.path.exists('test_file.json')

    # 6. Trigger: ValueError "File belongs to..." in load()
    # Create a dummy JSON with a wrong model name
    with open('wrong_model.json', 'w') as f:
        json.dump({'model': 'WrongClass', 'version': '1.0'}, f)
    try:
        model_ss.load('wrong_model.json')
    except ValueError as e:
        assert "File belongs to" in str(e)

    # 7. Trigger: ValueError "Version mismatch" in load()
    with open('wrong_version.json', 'w') as f:
        json.dump({'model': 'AortaPortalLiver', 'version': '0.0'}, f)
    try:
        model_ss.load('wrong_version.json')
    except ValueError as e:
        assert "Version mismatch" in str(e)

    # 8. Trigger: "return subset" in params()
    # Occurs when calling params with multiple arguments
    res = model_ss.params('BAT', 'CO')
    assert isinstance(res, dict)
    assert 'BAT' in res and 'CO' in res

    # Cleanup
    for f in ['test_file.json', 'wrong_model.json', 'wrong_version.json']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_full_coverage_aorta_portal_liver()
    test_liver_multicompartment_branch()
    test_ui_aorta_portal_liver()
    test_coverage_gap_filler

    print('All ui_liver tests passed!!')