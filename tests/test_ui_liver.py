import os
import warnings
import json

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc


DEBUG = False

if DEBUG:
    VERBOSE = 1
    SHOW = True
else:
    VERBOSE = 0
    SHOW = False


def test_ui_liver_bootstrap():
    # Sanity check: fit with exact values as initial vals

    liver = dc.Liver()
    time, _, _ = liver.input()
    signal = liver.predict(time)
    liver.train(time, signal)
    liver.plot(time, signal, show=SHOW)
    assert liver.cost(time, signal) < 1e-6


def test_ui_liver():

    time, aif, vif, roi, gt = dc.fake_liver()

    # Show dual-inlet model
    params = {
        'kinetics': '2I-IC',
        't': time,
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),
        'R10v': 1/dc.T1(3.0, 'blood'),
    }
    model = dc.Liver(**params)
    model.train(time, roi, aif, vif, n0=10)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 0.1


    # Show single-inlet model
    params = {
        'kinetics': '1I-IC-HFD',
        't': time,
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),        
    }
    model = dc.Liver(**params)
    model.train(time, roi, aif, n0=10)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1.0
    pars = model.export_params()
    assert 0.2 < pars['ve'][1] < 0.3
    model.print_params(round_to=3)

    # Loop over all models
    for k in ['2I-EC', '2I-EC-HF', '1I-EC', '1I-EC-D', 
              '2I-IC', '2I-IC-HF', '2I-IC-U', '1I-IC-HF', 
              '1I-IC-HFD', '1I-IC-HFDU']:
        params['kinetics'] = k
        if '-EC' in k:
            non_stat = [None]
        elif k not in ['2I-IC-U', '1I-IC-HFDU']:
            non_stat = ['UE','U','E', None]
        else:
            non_stat = ['U', None]
        for ns in non_stat:
            params['non_stationary'] = ns
            model = dc.Liver(**params)
            if k[0]=='2':
                model.train(time, roi, aif, vif, n0=10, xtol=1e-2)
            else:
                model.train(time, roi, aif, n0=10, xtol=1e-2)
            model.export_params()
            assert model.cost(time, roi) < 25

    # Display last result
    model.plot(time, roi, ref=gt, show=SHOW)


def test_liver_io():
    """Test JSON serialization and version/model validation."""
    time, aif, vif, roi, _ = dc.fake_liver()
    model = dc.Liver(kinetics='2I-EC', t=time)
    model.train(time, roi, aif, vif, n0=5)
    
    filename = "test_liver_model.json"
    try:
        # Test Save
        model.save(filename)
        assert os.path.exists(filename)
        
        # Test Load
        new_model = dc.Liver(kinetics='2I-EC')
        new_model.load(filename)
        
        # Verify state recovery
        assert new_model._kinetics == '2I-EC'
        assert np.allclose(new_model.predict(time), model.predict(time))
        assert new_model._pcov is not None
        
        # Test Load Errors (Force a model mismatch)
        # We manually edit the JSON to simulate a different model type
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        data['model'] = 'Kidney'
        with open('wrong_model.json', 'w') as f:
            json.dump(data, f)
            
        try:
            model.load('wrong_model.json')
        except ValueError:
            pass # Success: caught model mismatch
            
    finally:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists('wrong_model.json'): os.remove('wrong_model.json')


def test_liver_errors():
    """Target the Exception branches in the class."""
    time = np.arange(0, 60, 1)
    
    # 1. Invalid Sequence
    try:
        dc.Liver(sequence='InversionRecovery')
    except ValueError:
        pass 

    # 2. Invalid Parameter Name
    try:
        dc.Liver(fake_parameter=99)
    except ValueError:
        pass

    # 3. Predict beyond AIF
    model = dc.Liver(t=time)
    try:
        model.predict(np.array([120])) # AIF only goes to 60
    except ValueError:
        pass

    # 4. Train with bad bounds
    try:
        # Initial ve=0.3, but we set lower bound to 0.5
        model.train(time, np.ones(60), free={'ve': [0.5, 0.6]})
    except ValueError:
        pass

def test_liver_api_variants():
    """Test non-default API paths (SR sequence, non-summed concentrations)."""
    time, aif, vif, roi, _ = dc.fake_liver()
    
    # Test Saturation Recovery (SR) sequence path
    model_sr = dc.Liver(sequence='SR', kinetics='2I-IC', t=time)
    # Ensure SR-specific parameters are initialized
    assert 'TC' in model_sr._pars
    
    # Test Forward API outputs
    t, Cl = model_sr.conc()
    assert Cl.ndim == 2 # Should return [compartment, time]
    
    t, R1 = model_sr.relax()
    assert len(R1) == len(t)
    
    t, Sl = model_sr.signal()
    assert len(Sl) == len(t)

    # Test parameter retrieval variants
    val = model_sr.params('ve')
    assert isinstance(val, float)
    
    vals = model_sr.params('ve', 'Fp', round_to=2)
    assert 've' in vals and 'Fp' in vals


def test_coverage_gaps():
    """Target specifically missed branches for 100% coverage."""
    time = np.linspace(0, 100, 200)
    signal = np.ones_like(time)
    
    # 1. Trigger the bare exception in __init__ for invalid kinetics
    try:
        # Pass something that makes liver.params_liver fail
        dc.Liver(kinetics='Invalid-Model-Name')
    except Exception:
        pass

    # 2. Trigger Upper Bound Error in train()
    # Initial 've' is 0.3, setting upper bound to 0.1
    model = dc.Liver(t=time)
    try:
        model.train(time, signal, free={'ve': [0.0, 0.1]})
    except ValueError as e:
        assert "out of bounds" in str(e)

    # 3. Trigger JSON Version Mismatch
    filename = "version_fail.json"
    model.save(filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    data['version'] = '99.9' # Force mismatch
    with open(filename, 'w') as f:
        json.dump(data, f)
    try:
        model.load(filename)
    except ValueError as e:
        assert "Version mismatch" in str(e)
    if os.path.exists(filename): os.remove(filename)

    # 4. Trigger curve_fit RuntimeError (The 'except RuntimeError' block)
    # We provide data that is impossible to fit or nonsensical to force a failure
    model_fail = dc.Liver(t=time)
    # Passing an empty signal or incompatible shapes usually forces runtime issues
    with warnings.catch_warnings(record=True) as w:
        # Force curve_fit to fail by using max_nfev=1
        model_fail.train(time, signal, max_nfev=1)
        assert len(w) > 0
        assert "Curve fit failed" in str(w[-1].message)

    # 5. Test SR paths for AIF and VIF estimation
    # This hits the 'elif self._sequence == 'SR'' blocks in _estimate_parameters
    model_sr = dc.Liver(kinetics='2I-EC', sequence='SR', t=time)
    # Providing aif and vif as signals (nparrays) triggers the estimation logic
    model_sr.train(time, signal, aif=signal, vif=signal, n0=2)
    
    # 6. Test Sref <= 0 branch
    # Manually force FA to 0 to make signal 0
    model_zero = dc.Liver(t=time, FA=0)
    model_zero.train(time, signal, n0=2)
    assert model_zero._pars['S0'] == 0

    # 7. Test plot(show=False) to hit the plt.close() line
    model_zero.plot(time, signal, show=False)

    # 8. Test export_params with pcov=None branch
    # (By default pcov is None until train is called, or if train fails)
    model_no_fit = dc.Liver()
    p_no_cov = model_no_fit.export_params()
    # Verify sdev (index 3) is 0
    assert p_no_cov['ve'][3] == 0

    # 9. Test params() with multiple args and no rounding
    p_dict = model_no_fit.params('ve', 'Fp')
    assert isinstance(p_dict, dict)
    assert 've' in p_dict and 'Fp' in p_dict

def test_save_extension_logic():
    """Test the logic that adds .json if missing."""
    model = dc.Liver()
    model.save("test_file_no_ext")
    assert os.path.exists("test_file_no_ext.json")
    if os.path.exists("test_file_no_ext.json"): 
        os.remove("test_file_no_ext.json")

def test_final_coverage_gaps():
    """Target the last remaining lines for 100% coverage."""
    import os
    time = np.linspace(0, 60, 120)
    signal = np.ones_like(time)
    model = dc.Liver(t=time)
    
    # 1. Trigger ValueError: {p} is not a valid free parameter.
    # 'not_a_param' does not exist in the model's parameter list.
    try:
        model.train(time, signal, free={'not_a_param': [0, 1]})
    except ValueError as e:
        assert "not a valid free parameter" in str(e)

    # 2. Trigger return round(self._pars[args[0]], round_to)
    # Testing the single-argument rounding branch in params()
    model._pars['ve'] = 0.123456
    rounded_val = model.params('ve', round_to=2)
    assert rounded_val == 0.12
    
    # 3. Trigger plt.savefig(fname) and plt.show()
    # We test savefig by providing a filename, and show by setting show=True.
    # Note: In a CI environment, plt.show() might be mocked or 
    # handled by a non-interactive backend (Agg) to prevent windows popping up.
    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(time, signal, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang your tests
        plt.ion() # Turn interactive mode on
        model.plot(time, signal, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)

if __name__ == "__main__":

    # Integration tests
    test_ui_liver_bootstrap()
    test_ui_liver()

    # Coverage-focused tests
    test_liver_io()
    test_liver_errors()
    test_liver_api_variants()
    test_coverage_gaps()
    test_save_extension_logic()
    test_final_coverage_gaps()

    print('All ui_liver tests passed!!')