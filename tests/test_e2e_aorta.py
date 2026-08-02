import os
import itertools
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt

from dcmri import Aorta as Model
import dcmri as dc
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
    # if cnfg != ('comp', '2cxm', '3D-IR-SPGR'):
    #     return
    # data = {'dose': 0.2, 'rate': 5, 'TE': 0.05, 'TE1': 0.03, 'TE2': 0.05, 'TR': 15.5, 'FA': 90}
    data = {}
    try:
        model = Model(data, **cnfg)
    except InvalidConfiguration:
        return
    # print(cnfg)
    free = model.params('free')
    model._model.inputs()
    pred = model.predict()
    tacq, signal = pred['tS'], pred['Sa']
    result = model.train(tacq, signal, verbose=VERBOSE, n0=5, n_bat=1, xtol=1e-6)
    # if result['cost'] > 1:
    #     result = model.train(tacq, signal, verbose=VERBOSE, n0=10, n_bat=10, xtol=1e-3)
    # if result['cost'] > 1:
    #     result = model.train(tacq, signal, verbose=VERBOSE, n0=10, n_bat=50, xtol=1e-3)
    model.plot(tacq, signal, show=DEBUG)
    cost = model.cost(tacq, signal)
    print(f"{cnfg}: {cost}")
    #assert cost < 100, f"Cost {cost} of model {cnfg} exceeded threshold!"
    return cnfg, cost


def test_config_coverage():
    if DEBUG:
        return
    
    start = time.perf_counter()

    result = Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfg)
        for cnfg in dc.AortaModel.configurations()
    )

    # result = [
    #     _run_single_config(cnfg)
    #     for cnfg in dc.AortaModel.configurations()
    # ]

    result = [r for r in result if r is not None]
    cost = [r[1] for r in result]
    cnfg = result[cost.index(max(cost))][0]

    end = time.perf_counter()
    print(f'Configuration coverage completed!')
    print(f'--> Number of configurations: {np.prod([len(v) for v in dc.AortaModel.configs.values()])}')
    print(f'--> Total computation time: {(end - start) / 60:.1f} mins')
    print(f'--> Maximum cost: {np.max(cost)} %')
    print(f'--> Config with maximum cost: {cnfg}')


def test_code_coverage(): 
    _run_single_config({'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 't2s_relaxation': 'lin', 'sequence': '3D-IR-SPGR', 'magnitude': False}) 

    model = Model()

    # params()
    assert 'Thl' in model.params()
    assert np.isscalar(model.state['Thl']) 
    
    # Test Forward API outputs
    S = model.predict()

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(S['tS'], S['Sa'], fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang the tests
        plt.ion() # Turn interactive mode on
        model.plot(S['tS'], S['Sa'], show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)


if __name__ == "__main__":
    test_code_coverage()
    # test_config_coverage()
    
    print('All Aorta tests passed!!')

