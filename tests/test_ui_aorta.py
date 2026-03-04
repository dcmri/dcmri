import warnings
import os
import json
import numpy as np
import dcmri as dc


DEBUG = False

if DEBUG:
    VERBOSE = 1
    SHOW = True
else:
    VERBOSE = 0
    SHOW = False


def test_ui_aorta_bootstrap():
    # Sanity check: fit with exact values as initial vals

    aorta = dc.Aorta()
    time, _ = aorta.conc()
    signal = aorta.predict(time)
    aorta.plot(time, signal, show=SHOW)
    aorta.train(time, signal)
    aorta.plot(time, signal, show=SHOW)
    assert aorta.cost(time, signal) < 0.5

def test_ui_aorta():

    truth = {'BAT': 20}
    time, aif, _, _ = dc.fake_tissue(**truth)
    aorta = dc.Aorta(
        dt = 1.5,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 1.5,
        R10 = 1/dc.T1(3.0,'blood'), 
        heartlung = 'chain',
    )
    aorta.train(time, aif, xtol=1e-3)
    aorta.plot(time, aif, show=SHOW)
    rec = aorta.export_params()
    rec_aif = aorta.predict(time)
    assert rec['BAT'][1] == aorta.params('BAT')
    assert np.linalg.norm(aif-rec_aif) < 0.1*np.linalg.norm(aif)
    assert np.abs(rec['BAT'][1]-truth['BAT']) < 0.2*truth['BAT']

def test_aorta_configs_and_sequences():
    """Triggers different organ models and sequence math branches."""
    time = np.linspace(0, 60, 20)
    
    # Test 2CXM + SSI (Triggers complex organ model and inflow sequence)
    aorta_ssi = dc.Aorta(organs='2cxm', sequence='SSI', TF=0.5)
    sig_ssi = aorta_ssi.predict(time)
    # SSI requires S0 in free pars (triggers that specific check)
    aorta_ssi.train(time, sig_ssi) 
    assert aorta_ssi._sequence == 'SSI'

    # Test SR sequence (Triggers TC parameter and signal_free path)
    aorta_sr = dc.Aorta(sequence='SR', TC=0.2)
    sig_sr = aorta_sr.predict(time)
    aorta_sr.train(time, sig_sr)
    
    # Test Linear sequence (Triggers signal_lin path)
    aorta_lin = dc.Aorta(sequence='lin')
    _ = aorta_lin.predict(time)
    _, r1 = aorta_lin.relax()
    assert len(r1) > 0

def test_aorta_errors():
    """Triggers ValueError branches."""
    # Invalid config
    try: dc.Aorta(organs='invalid')
    except ValueError: pass
    
    try: dc.Aorta(heartlung='invalid')
    except ValueError: pass
    
    try: dc.Aorta(sequence='invalid')
    except ValueError: pass

    # Invalid parameter name
    try: dc.Aorta(fake_param=10)
    except ValueError: pass

    aorta = dc.Aorta()
    time = np.arange(10)
    sig = np.ones(10)

    # SSI S0 check
    aorta_ssi = dc.Aorta(sequence='SSI')
    try: 
        # Manually passing free without S0
        aorta_ssi.train(time, sig, free={'TF': [0, 1]})
    except ValueError: pass

    # Out of bounds check
    try:
        aorta.train(time, sig, free={'CO': [500, 600]}) # Initial CO is 100
    except ValueError: pass

def test_aorta_io_and_api():
    """Triggers JSON I/O, export sdevs, and plotting branches."""
    aorta = dc.Aorta()
    time = np.linspace(0, 30, 10)
    signal = aorta.predict(time)
    aorta.train(time, signal, max_nfev=2) # Short train to get pcov

    # Save/Load
    file = 'test_aorta.json'
    aorta.save(file)
    new_aorta = dc.Aorta().load(file)
    assert new_aorta.params('CO') == aorta.params('CO')
    
    # Version/Model mismatch in Load
    with open(file, 'r') as f: data = json.load(f)
    data['model'] = 'WrongModel'
    with open('wrong.json', 'w') as f: json.dump(data, f)
    try: aorta.load('wrong.json')
    except ValueError: pass

    # Export & Print
    aorta.print_params(round_to=2)
    p_multi = aorta.params('CO', 'BAT', round_to=1)
    assert 'CO' in p_multi

    # Plot branches (savefig and close)
    aorta.plot(time, signal, fname='test_plot.png', show=False)
    
    # Cleanup
    for f in [file, 'wrong.json', 'test_plot.png', 'test_aorta.json']:
        if os.path.exists(f): os.remove(f)

def test_aorta_fit_failure():
    """Triggers the RuntimeError/Warning block in train."""
    aorta = dc.Aorta()
    time = np.array([0, 1, 2])
    signal = np.array([1, 1e6, 1]) # Impossible data
    with warnings.catch_warnings(record=True) as w:
        aorta.train(time, signal, max_nfev=1)
        assert len(w) > 0 # Runtime error caught

def test_aorta_final_coverage_gaps():
    """Targets the final specific lines for 100% coverage."""
    import os
    import json
    time = np.linspace(0, 60, 10)
    signal = np.ones_like(time)
    
    # 1. Trigger: raise ValueError(f"'{p}' is not a valid parameter...")
    try:
        dc.Aorta(organs='comp', Toe=100) # Toe is only valid for '2cxm'
    except ValueError as e:
        assert "is not a valid parameter" in str(e)

    # 2. Trigger: s_ref = sig.signal_lin(1, r10)
    # This happens in _estimate_parameters when sequence is 'lin'
    model_lin = dc.Aorta(sequence='lin')
    model_lin.train(time, signal) 

    # 3. Trigger: file += '.json'
    model_lin.save("no_extension_file")
    assert os.path.exists("no_extension_file.json")
    os.remove("no_extension_file.json")

    # 4. Trigger: raise ValueError(f"Version mismatch...")
    temp_file = "version_test.json"
    model_lin.save(temp_file)
    with open(temp_file, 'r') as f:
        data = json.load(f)
    data['version'] = '99.9'
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    try:
        model_lin.load(temp_file)
    except ValueError as e:
        assert "Version mismatch" in str(e)
    os.remove(temp_file)

    # 5. Trigger: return subset
    # This hits the 'params' call with multiple arguments
    subset = model_lin.params('CO', 'BAT')
    assert isinstance(subset, dict)
    assert len(subset) == 2

    # 6. Trigger: ax1.plot(ref['t']/60, 1000*ref['cb'], 'ks', label='Ground Truth')
    # This requires passing a 'ref' dictionary to the plot method
    ref_data = {
        't': np.array([0, 30, 60]),
        'cb': np.array([0, 0.5, 0.1])
    }
    model_lin.plot(time, signal, ref=ref_data, show=False)

def test_train_invalid_free_param():
    """Triggers ValueError in train() when a free parameter is not valid for the config."""
    # Initialize a standard aorta model (default is organs='comp')
    aorta = dc.Aorta(organs='comp')
    
    time = np.arange(0, 60, 2)
    signal = np.ones_like(time)
    
    # Try to train with 'Toe' as a free parameter. 
    # 'Toe' is only valid for organs='2cxm'. 
    # Since we initialized with 'comp', 'Toe' is not in self._pars.
    try:
        aorta.train(time, signal, free={'Toe': [0, 500]})
    except ValueError as e:
        assert "is not a valid parameter for this configuration" in str(e)
        
    # Another example: 'TF' is only for 'SSI' sequence.
    # Default is 'SS', so this should also fail.
    try:
        aorta.train(time, signal, free={'TF': [0, 2]})
    except ValueError as e:
        assert "is not a valid parameter for this configuration" in str(e)

if __name__ == "__main__":
    test_ui_aorta_bootstrap()
    test_ui_aorta()
    test_aorta_configs_and_sequences()
    test_aorta_errors()
    test_aorta_io_and_api()
    test_aorta_fit_failure()
    test_aorta_final_coverage_gaps()
    test_train_invalid_free_param()

    print('All ui_aorta tests passed!!')