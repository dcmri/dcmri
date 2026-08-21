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


def test_aorta_times():

    # Setup a profiler to aggregate timing across all iterations
    profiler = cProfile.Profile()

    def _test_config(cnfg):
        try:
            model = AortaModel(**cnfg)
        except InvalidConfiguration:
            return
        print(cnfg)
        data = model.lexicon_data()
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
    for cnfg in AortaModel.configurations():
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



def test_aorta():
    def _test_config(cnfg):
        try:
           model = AortaModel(**cnfg)
        except InvalidConfiguration:
            return
    
        print(cnfg)
        data = model.lexicon_data()
        t0 = time.perf_counter()
        results = model(data)
        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")
        assert results['S_a'].ndim == 3

    configs = AortaModel.configurations()
    cnt = 0
    for cnfg in tqdm(list(configs)):
        cnt += 1
        _test_config(cnfg)
        # if cnt==100:
        #     break

    print(f'Successfully covered {cnt} Aorta configurations!')


def test_aorta_function():
    cnfg = {
        'heartlung': 'comp', 
        'organs': 'comp', 
        'kidneys': None, 
        'liver': None, 
        'lagut': None, 
        'bolus': 'single', 
        't1_relaxation': 'lin', 
        't2_relaxation': None, 
        't2s_relaxation': None, 
        'sequence': 'ZTE-3D-IR-SPGR-SS', 
        'inflow': True, 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = AortaModel(**cnfg)
    except InvalidConfiguration:
        return
    print(model.inputs())
    print(model.outputs())

    data = model.lexicon_data() 
    results = model(data)
    
    plt.plot(results['tS_a'], results['S_a'][0, 0, :], 'ro')
    plt.show()


if __name__ == '__main__':
    test_aorta()
    #test_aorta_times()
    #test_aorta_function()

    print('All model coverage tests passed!!')