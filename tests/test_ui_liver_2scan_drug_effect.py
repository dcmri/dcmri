import os
import json

import numpy as np
from dcmri import Liver2scanDrugEffect


DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')



def test_function():

    aol = Liver2scanDrugEffect()

    tacq = aol.time()
    data = aol.predict(tacq)

    tmp_file = 'tmp.png'
    aol.plot(tacq, data, fname=tmp_file)
    aol.plot(tacq, data, show=False)
    assert aol.cost(tacq, data) == 0

    # Training should not have much of an effect if we use the exact R102 values
    R1 = aol.relax()
    R102a = [R1[1][0], R1[5][0]]
    R102l = [R1[3][0], R1[7][0]]
    aol.train(tacq, data, R102a=R102a, R102l=R102l, verbose=VERBOSE, xtol=0.1)
    aol.plot(tacq, data)
    aol.export_params()
    assert aol.cost(tacq, data) < 2

    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

def test_options():

    aol = Liver2scanDrugEffect()

    tacq = aol.time()
    data = aol.predict(tacq) 

    free = {'C-S0(a)': [0,10], 'C-S0(l)': [0,10]}
    Liver2scanDrugEffect().train(tacq, data, free=free, verbose=VERBOSE, max_nfev=1)  

    bounds = {'GFR': [0,10], 'C-S0(l)': None}
    Liver2scanDrugEffect().train(tacq, data, bounds=bounds, verbose=VERBOSE, max_nfev=1)

def test_utilities():
    """Covers I/O, Printing, and Parameter Export"""
    aol = Liver2scanDrugEffect(CO=100)

    # Get pars options
    aol.params('C-k(he,i)', as_dict=True)
    aol.params('C-k(he,i)', 'C-k(he,f)')
    aol.params('C-k(he,i)')

    # Test conc
    aol.conc()
    
    # Test export and print
    params = aol.export_params()
    assert isinstance(params, dict)
    aol.print_params(round_to=2)
    aol.print_params()

    # Test Save/Load (I/O)
    tmp_file = "test_model.json"
    aol.save(tmp_file)
    aol.save("test_model")
    assert os.path.exists(tmp_file)
    
    new_model = Liver2scanDrugEffect()
    new_model.load(tmp_file)
    
    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def test_load_validation_errors():
    """Specifically targets model name and version mismatch during loading."""
    aol = Liver2scanDrugEffect()
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
    
    # Test invalid parameter override
    try:
        Liver2scanDrugEffect(fake_param=99)
    except ValueError:
        pass

    # 4. Test training out of bounds
    tacq = (np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10))
    data = (np.ones(10), np.ones(10), np.ones(10), np.ones(10), np.ones(10), np.ones(10), np.ones(10), np.ones(10))
    try:
        # Pass a bound that excludes the current 'CO' (100)
        Liver2scanDrugEffect().train(tacq, data, bounds={'CO': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a bound that is not free
        Liver2scanDrugEffect().train(tacq, data, bounds={'dt': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a bound that is not a parameter
        Liver2scanDrugEffect().train(tacq, data, bounds={'xx': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass a free parameter that is not a parameter
        Liver2scanDrugEffect().train(tacq, data, free={'xx': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on BAT
        Liver2scanDrugEffect().train(tacq, data, bounds={'C-BAT': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on BAT2
        Liver2scanDrugEffect().train(tacq, data, bounds={'D-BAT2': [10, 20]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on GFR
        Liver2scanDrugEffect().train(tacq, data, bounds={'GFR': [20, 10]})
    except ValueError:
        pass
    try:
        # Pass an invalid bound on S0
        Liver2scanDrugEffect().train(tacq, data, bounds={'C-S0(l)': [-1, 1]})
    except ValueError:
        pass

if __name__ == "__main__":
    test_function()
    test_options()
    test_utilities()
    test_errors()
    test_load_validation_errors()
    print('All tests passed!')