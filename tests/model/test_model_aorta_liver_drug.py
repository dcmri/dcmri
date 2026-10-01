import os
from joblib import Parallel, delayed
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from dcmri import AortaLiverDrug as Model
from dcmri.core.module import InvalidConfig

DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def _test_config(cnfg):
    #state = {'tacq': 600}
    state = None
    try:
        model = Model(state, **cnfg)
    except InvalidConfig:
        return
    # print(cnfg)
    state = model.state()
    data = model.predict()
    model.train(data, nb=5, n_bat=4, verbose=VERBOSE, xtol=1e-3)
    model.plot(data, show=DEBUG)
    cost = model.cost(data)
    # print(f"{cnfg}: {cost}")
    print(cost)
    #assert cost < 1e-1, f"Cost {cost} of model {cnfg} exceeded threshold!"


def test_model_aorta_liver_drug_instance():
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 'lagut': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'lin'}
    _test_config(cnfg)


def test_model_aorta_liver_drug():
    configs = Model.all_configs(sample=1e4, seed=51)

    # [_test_config(cnfg) for cnfg in tqdm(configs, desc=f'Testing {Model.__name__}')]
    Parallel(n_jobs=-1)(delayed(_test_config)(cnfg) for cnfg in configs)

    print(f'Successfully covered {len(configs)} {Model.__name__} configurations!')


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
    # test_model_aorta_liver_drug_instance()
    test_model_aorta_liver_drug()
    # test_api()
    
    print('All AortaLiverDrug tests passed!!')

