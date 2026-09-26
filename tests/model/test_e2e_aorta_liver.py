import os
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt

from dcmri import AortaLiver as Model
from dcmri.core.module import InvalidConfig
import dcmri as dc


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
    except InvalidConfig as e:
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


def test_all_config():
    if DEBUG:
        return
    
    start = time.perf_counter()

    result = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfg)
        for cnfg in dc.ForwardAortaLiver.all_configs(sample=1e4, seed=40)
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
    print(f'--> Number of configurations: {np.prod([len(v) for v in dc.ForwardAortaLiver.configs.values()])}')
    print(f'--> Total computation time: {(end - start) / 60:.1f} mins')
    print(f'--> Maximum cost: {np.max(cost)} %')
    print(f'--> Config with maximum cost: {cnfg}')


def test_single_config(): 
    config = {
        'bolus': 'single', 
        'heartlung': 'pfcomp',
        'organs': 'comp',
        'lagut': 'plucom',
        'liver': '1I-IC',
        'non_stationary': 'UE', 
        'water_exchange': 'R',
        't1_relaxation_ao': 'lin',
        't2_relaxation_ao': None, 
        't2s_relaxation_ao': None, 
        't1_relaxation_li': 'lin',
        't2_relaxation_li': None, 
        't2s_relaxation_li': None, 
        'inflow': 'none',
        'sequence': 'ZTE-3D-IR-SPGR-SS',
        # 'sequence': '3D-IR-SPGR', 
        'magnitude': False,
        'calibrate': True,
    }
    _run_single_config(config) 


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


if __name__ == "__main__":
    test_single_config()
    # test_all_config()
    # test_api()
    
    print('All AortaLiver tests passed!!')

