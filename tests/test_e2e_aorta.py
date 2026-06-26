import os
import itertools
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt

from dcmri import Aorta as Model

DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def _run_single_config(cnfg):
    # if cnfg != ('comp', '2cxm', '3D-IR-SPGR'):
    #     return
    # print(cnfg)
    model = Model(*cnfg)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, verbose=VERBOSE, xtol=1e-6)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(f"\n{cnfg}: {cost}")
    model.conc()
    model.relax()
    model.signal()
    assert cost < 10, f"Cost {cost} of model {cnfg} exceeded threshold!"
    return cost


def test_config_coverage():
    if DEBUG:
        return
    values = Model.configs.values()
    start = time.perf_counter()

    cost = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfgs)
        for cnfgs in itertools.product(*values)
    )
    # cost = [
    #     _run_single_config(cnfgs) 
    #     for cnfgs in itertools.product(*values)
    # ]

    end = time.perf_counter()
    print(f'Configuration coverage completed!')
    print(f'--> Number of configurations: {np.prod([len(v) for v in values])}')
    print(f'--> Total computation time: {(end - start) / 60:.1f} mins')
    print(f'--> Maximum cost: {np.max(cost)} %')


def test_code_coverage(): 
    # _run_single_config(('comp', 'comp', '2D-IR-SPGR'))
    _run_single_config(('comp', 'comp', 'ZTE-3D-IR-SPGR-SS')) 

    model = Model()

    # params()
    assert 'Thl' in model.params()
    assert np.isscalar(model.state('Thl')) 
    assert not np.isscalar(model.state('Thl', 'Dhl')) 
    
    # Test Forward API outputs
    t = model.time()
    S = model.signal()

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(t, S, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang your tests
        plt.ion() # Turn interactive mode on
        model.plot(t, S, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)

    # Exceptions

    # # SSI sequence model with fixed S0
    # try:
    #     model = Model(sequence='3D-SPGR-SSI')
    #     t, s = model.time(), model.signal()
    #     model.train(t, s, bounds={'S0_a': None})
    # except ValueError:
    #     pass
    # else:
    #     assert False


if __name__ == "__main__":
    test_code_coverage()
    test_config_coverage()
    
    print('All Aorta tests passed!!')

