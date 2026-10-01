import os
from joblib import Parallel, delayed
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from dcmri import Liver as Model
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

    state = model.state()
    data = model.predict()
    model.train(data, nb=5, verbose=VERBOSE, xtol=1e-3)
    model.plot(data, show=DEBUG)
    cost = model.cost(data)
    # print(f"{cnfg}: {cost}")
    print(cost)
    #assert cost < 1e-1, f"Cost {cost} of model {cnfg} exceeded threshold!"


def test_model_liver_instance():
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'inflow': 'none', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'kinetics': '2I-EC', 'non_stationary': None, 'input': True}
    _test_config(cnfg)


def test_model_liver():
    configs = Model.all_configs(sample=1e4, seed=51)

    # [_test_config(cnfg) for cnfg in tqdm(configs, desc=f'Testing {Model.__name__}')]
    Parallel(n_jobs=-1)(delayed(_test_config)(cnfg) for cnfg in configs)

    print(f'Successfully covered {len(configs)} {Model.__name__} configurations!')


if __name__ == "__main__":
    #test_model_liver_instance()
    test_model_liver()
    
    print('All Liver tests passed!!')

