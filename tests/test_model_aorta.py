# Prevent NumPy/SciPy/OpenBLAS from spawning thread pools for tiny array operations
import os

import cProfile
import pstats
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import AortaModel
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.bloch.functions_dynamic import Mz_dyn_se

# kernprof automatically injects 'profile' into builtins when run via command line
# kernprof -l -v tests/test_model_aorta.py
try:
    Mz_dyn_se = profile(Mz_dyn_se)
except NameError:
    # Fallback so the script doesn't crash if run with standard 'python' instead of 'kernprof'
    pass

def test_aorta(cls=AortaModel):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
    
        data = instance.dummy_data()

        # --- DIAGNOSTIC TIMING ---
        t0 = time.perf_counter()
        # print(cnfg)
        instance(data)

        elapsed = time.perf_counter() - t0
        # print(cnfg)
        # print(f"  [Total model execution time: {elapsed:.4f}s]")

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=1e5, seed=51)

    configs = cls.all_configs(sample=1e4, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')




def test_aorta_times():

    # Setup a profiler to aggregate timing across all iterations
    profiler = cProfile.Profile()

    def _test_config(cnfg):
        try:
            model = AortaModel(**cnfg)
        except InvalidConfiguration:
            return
        print(cnfg)
        data = model.dummy_data()
        # data |= {'dt': 2.0, 'tmax': 30}

        # --- DIAGNOSTIC TIMING ---
        t0 = time.perf_counter()

        # Profile the specific execution of model(data)
        profiler.enable()
        results = model(data)
        profiler.disable()

        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")

        assert results["S_a"].ndim == 3

    cnt = 0
    for cnfg in AortaModel.all_configs():
        cnt += 1
        _test_config(cnfg)
        if cnt == 100:
            break

    print(f"\nSuccessfully covered {cnt} aorta configurations!")

    # --- PRINT DETAILED SUBFUNCTION BREAKDOWN ---
    print("\n" + "=" * 60)
    print("TOP 15 SUBFUNCTIONS BY TOTAL TIME (cumtime):")
    print("=" * 60)
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(10)



def test_aorta_instance():
    cnfg = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'kidneys': None,
        'liver': None, 
        'lagut': None, 
        'bolus': 'dual', 
        't1_relaxation': 'lin', 
        't2_relaxation': None, 
        't2s_relaxation': None, 
        'inflow': 'none', 
        'sequence': 'ZTE-3D-SPGR-SS', 
        'magnitude': False, 
        'trigger': True, 
        'calibrate': False,
    }
    try:
        model = AortaModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data() 
    results = model(data)
    
    plt.plot(results['tS'], results['S'][0, 0, :], 'ro')
    plt.show()


if __name__ == '__main__':
    test_aorta()
    # test_aorta_instance()
    # test_aorta_times()

    print('All model coverage tests passed!!')