import os
import json

import numpy as np
from dcmri import AortaLiver


DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')




def test_options():

    aol = AortaLiver('1I-EC', FA=12)

    tacq = aol.time()
    data = aol.predict(tacq) 
    aol.plot(tacq, data)

    free = {'S0(a)': [0,10], 'S0(l)': [0,10]}
    bounds = {'CO': None}
    aol.train(tacq, data, free=free, bounds=bounds, verbose=VERBOSE, max_nfev=1)  

def test_utilities():
    """Covers I/O, Printing, and Parameter Export"""
    aol = AortaLiver(CO=100, kinetics='1I-EC')

    # Test conc
    ca, cl = aol.conc()
    R1a, R1l = aol.relax()
    
    # Test export and print
    params = aol.export_params()
    assert isinstance(params, dict)
    aol.print_params(round_to=2)
    aol.print_params()
    p = aol.params('FA', 'TR', as_dict=True)
    fa, tr = aol.params('FA', 'TR')
    fa = aol.params('FA')

    # Test Save/Load (I/O)
    tmp_file = "test_model.json"
    aol.save(tmp_file)
    aol.save("test_model")
    assert os.path.exists(tmp_file)
    
    new_model = AortaLiver()
    new_model.load(tmp_file)
    
    # Verify a key parameter matches
    assert new_model._kinetics == aol._kinetics
    
    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def test_load_validation_errors():
    """Specifically targets model name and version mismatch during loading."""
    aol = AortaLiver()
    tmp_file = "validation_test.json"
    
    # Create a valid starting point
    aol.save(tmp_file)
    
    with open(tmp_file, "r") as f:
        data = json.load(f)

    # 1. Test Model Name Mismatch
    original_model_name = data['model']
    data['model'] = "WrongModelName"
    with open(tmp_file, "w") as f:
        json.dump(data, f)
    
    try:
        aol.load(tmp_file)
    except ValueError as e:
        assert "File belongs to WrongModelName" in str(e)
    
    # Restore model name for the next test
    data['model'] = original_model_name

    # 2. Test Version Mismatch
    data['version'] = "99.9.9" # Non-existent version
    with open(tmp_file, "w") as f:
        json.dump(data, f)
        
    try:
        aol.load(tmp_file)
    except ValueError as e:
        assert "Version mismatch" in str(e)

    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

def test_errors():
    """Covers Error Handling and Edge Cases"""
    # 1. Test invalid sequence
    try:
        AortaLiver(sequence='INVALID')
    except ValueError:
        pass

    # 2. Test invalid kinetics (non-single inlet)
    try:
        AortaLiver(kinetics='2I-EC')
    except ValueError:
        pass
    try:
        AortaLiver(kinetics='INVALID')
    except ValueError:
        pass

    # 3. Test invalid parameter override
    try:
        AortaLiver(fake_param=99)
    except ValueError:
        pass

    # 4. Test training out of bounds
    tacq = (np.arange(10), np.arange(10))
    data = (np.ones(10), np.ones(10))
    try:
        # Pass a bound that excludes the current 'CO' (100)
        AortaLiver().train(tacq, data, bounds={'CO': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a bound that is not free
        AortaLiver().train(tacq, data, bounds={'dt': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a bound that is not a parameter
        AortaLiver().train(tacq, data, bounds={'xx': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a free parameter that is not a parameter
        AortaLiver().train(tacq, data, free={'xx': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on BAT
        AortaLiver().train(tacq, data, bounds={'BAT': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on S0
        AortaLiver().train(tacq, data, bounds={'S0(a)': [-1, 1]})
    except ValueError:
        pass

def test_function():

    aol = AortaLiver()
    data = aol.predict()

    # Intended scenario
    tacq = aol.time()
    data = aol.predict(tacq)
    tmp_file = 'tmp.png'
    aol.plot(tacq, data, fname=tmp_file)
    aol.plot(tacq, data, show=False)
    assert aol.cost(tacq, data) == 0

    # Training should not have much of an effect if we use the exact R102 values
    tacq = aol.time()
    data = aol.predict(tacq)
    aol.train(tacq, data, verbose=VERBOSE, xtol=0.1)
    pars = aol.export_params()

    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

if __name__ == "__main__":
    
    test_options()
    test_utilities()
    test_errors()
    test_load_validation_errors()
    test_function()
    print('All tests passed!')