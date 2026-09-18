import os
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt

from dcmri import CortMed as Model
from dcmri import CortMedModel as Forward
from dcmri import AortaModel
from dcmri.core.exceptions import InvalidConfiguration


DEBUG = True

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def _run_single_config(cnfg):
    try:
        model = Model(**cnfg)
    except InvalidConfiguration as e:
        # print(e)
        return
    free = model.params('free')
    data = model.predict()
    model.train(data, verbose=VERBOSE, n0=5, n_bat=1, xtol=1e-3)
    model.plot(data, show=DEBUG)
    cost = model.cost(data)
    #print(f"{cnfg}: {cost}")
    print(cost)
    assert cost < 10, f"Cost {cost} of model {cnfg} exceeded threshold!"
    return cnfg, cost


def test_all_configs():
    if DEBUG:
        return
    
    start = time.perf_counter()

    result = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfg)
        for cnfg in Forward.all_configs(sample=1e5, seed=41)
    )
    # result = [
    #     _run_single_config(cnfg)
    #     for cnfg in Model.all_configs()
    # ]
    result = [r for r in result if r is not None]
    cost = [r[1] for r in result]
    cnfg = result[cost.index(max(cost))][0]

    end = time.perf_counter()
    print(f'Configuration coverage completed!')
    print(f'--> Number of configurations: {np.prod([len(v) for v in Forward.configs.values()])}')
    print(f'--> Total computation time: {(end - start) / 60:.1f} mins')
    print(f'--> Maximum cost: {np.max(cost)} %')
    print(f'--> Config with maximum cost: {cnfg}')


def test_single_config(): 
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'kinetics': '7C'}
    _run_single_config(cnfg) 


def test_api():
    model = Model()

    # params()
    assert 'T_hl' in model.params()
    
    # Test Forward API outputs
    data = model.predict()

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(data, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang the tests
        plt.ion() # Turn interactive mode on
        model.plot(data, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)


def test_function():

    # Simulation parameters
    ao = AortaModel()
    data = ao.dummy_data()
    aif = ao(data)

    # CortMedy signals
    params = data | {
        'c_ar': aif['C_ao'][0],
        'S0': 5,
    }

    # Tissue model
    model = Model(**params)
    data = model.predict()
    
    # Fit with AIF signal
    model.train(data, aif={'signal': aif['S'][0,0,:], 'time': aif['tS']}, verbose=VERBOSE)
    model.plot(data, show=DEBUG)
    # assert model.cost(data) < 1


if __name__ == "__main__":
    #test_single_config()
    test_all_configs()
    #test_api()
    # test_function()
    
    print('All CortMed tests passed!!')

