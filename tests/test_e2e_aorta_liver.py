import os
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt
from dcmri import AortaLiver as Model
from dcmri.core.exceptions import InvalidConfiguration
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
    # if cnfg != ('single', 'pfcomp', '2cxm', '3D-SPGR-SSI'):
    #     return
    try:
        model = Model(**cnfg)
    except InvalidConfiguration as e:
        print(f"Invalid configuration error: {e}")
        return
    free = model.params('free')
    data = model.predict()
    model.train(data, verbose=VERBOSE, xtol=1e-3)
    model.plot(data, show=DEBUG)
    cost = model.cost(data)
    print(f"\n{cnfg}: {cost}")
    # assert cost < 50, f"Cost {cost} of model {cnfg} exceeded threshold!"
    return cost


def test_config_coverage():
    if DEBUG:
        return
    
    start = time.perf_counter()

    result = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfg)
        for cnfg in dc.AortaLiverModel.configurations()
    )
    # result = [
    #     _run_single_config(cnfg)
    #     for cnfg in Model.configurations()
    # ]
    result = [r for r in result if r is not None]
    cost = [r[1] for r in result]
    cnfg = result[cost.index(max(cost))][0]

    end = time.perf_counter()
    print(f'Configuration coverage completed!')
    print(f'--> Number of configurations: {np.prod([len(v) for v in dc.AortaLiverModel.configs.values()])}')
    print(f'--> Total computation time: {(end - start) / 60:.1f} mins')
    print(f'--> Maximum cost: {np.max(cost)} %')
    print(f'--> Config with maximum cost: {cnfg}')


def test_code_coverage(): 
    config = {
        'bolus': 'single', 
        'heartlung': 'pfcomp',
        'lagut': 'plucom',
        'liver': '1I-IC',
        'non_stationary': 'UE', 
        'organs': 'comp', 
        'water_exchange': 'R',
        't2s_relaxation': 'lin', 
        'sequence': '3D-IR-SPGR', 
        'magnitude': False,
        'calibrate': False,
    }
    _run_single_config(config) 

    model = Model()

    # params()
    assert 'Thl' in model.params()
    assert np.isscalar(model.state['Thl']) 
    
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
    test_code_coverage()
    # test_config_coverage()
    
    print('All AortaLiver tests passed!!')

