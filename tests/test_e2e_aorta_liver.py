import os
import itertools
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt
from dcmri import AortaLiver as Model


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
    # if cnfg != ('single', 'pfcomp', '2cxm', '3D-SPGR-SSI'):
    #     return
    #print(cnfg)
    # pars = {'DRPF':0.25, 'vp_rk':0.3, 'TR': 2, 'TE': 0.05}
    try:
        model = Model(*cnfg)
    except ValueError:
        return
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, verbose=VERBOSE, xtol=1e-3)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(f"\n{cnfg}: {cost}")
    model.conc()
    model.relax()
    model.signal()
    assert cost < 50, f"Cost {cost} of model {cnfg} exceeded threshold!"
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
    #_run_single_config(('pfcomp', 'comp', '2I-EC', None, '2D-GE-EPI')) 
    _run_single_config(('comp', 'comp', '2I-EC', None, '2D-SR-SPGR'))
    model = Model()
    
    # Test Forward API outputs
    t = model.time()
    S = model.signal()

    # TODO: Write to p - now p is not overwritten which means derived parameters are lost
    # # export_params()
    # model.export_params(deriv=True)

    # # print_params()
    # model.print_params('Thl', 'Dhl', 'TS', deriv=True, fixed_only=True)
    # model.print_params('Thl', 'Dhl', 'TS', deriv=True, free_only=True)

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(t, S, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        plt.ion() # Turn interactive mode on
        model.plot(t, S, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)


if __name__ == "__main__":
    test_code_coverage()
    test_config_coverage()
    
    print('All AortaLiver tests passed!!')

