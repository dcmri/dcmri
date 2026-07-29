import os
import itertools
from joblib import Parallel, delayed
import time

import numpy as np
import matplotlib.pyplot as plt
from dcmri import AortaKidneys as Model


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
    pars = {'DRPF':0.25, 'vp_rk':0.3}
    model = Model(*cnfg, **pars)
    time = model.time()
    signal = model.predict(time)
    # for roi, sig in signal.items():
    #     signal[roi] = utils.add_noise(sig, sig[0] / SNR)
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
    _run_single_config(('pfcomp', 'comp', '2PF', '2D-GE-EPI')) 
    #_run_single_config(('comp', 'comp', '2CF', '3D-SPGR-SS')) 

    # Create some asymmetry for testing
    pars = {'DRPF':0.25, 'vp_rk':0.3}

    # One time array
    model = Model(**pars)
    time = model.time()
    signal = model.predict(time['aorta'])
    model.train(time['aorta'], signal, verbose=VERBOSE, xtol=0.01)
    model.plot(time['aorta'], signal, show=DEBUG)
    cost = model.cost(time['aorta'], signal)
    print(cost)
    assert cost < 5

    # Test Forward API outputs
    model = Model()
    t = model.time()
    S = model.signal()

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
    
    print('All aorta_kidneys tests passed!!')

