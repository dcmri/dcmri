# Prevent NumPy/SciPy/OpenBLAS from spawning thread pools for tiny array operations
import os

import cProfile
import pstats
import time

import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration

# kernprof automatically injects 'profile' into builtins when run via command line
# kernprof -l -v tests/test_model_aorta.py
try:
    dc.bloch.functions_dynamic.Mz_dyn_se = profile(dc.bloch.functions_dynamic.Mz_dyn_se)
except NameError:
    # Fallback so the script doesn't crash if run with standard 'python' instead of 'kernprof'
    pass


def test_aorta_times():

    # Setup a profiler to aggregate timing across all iterations
    profiler = cProfile.Profile()

    def _test_config(cnfg):
        try:
            model = dc.AortaModel(**cnfg)
        except InvalidConfiguration:
            return
        print(cnfg)
        data = dc.QVALUES | model.map_lexicon(dc.QVALUES)
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
    for cnfg in dc.AortaModel.configurations():
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
        # if cnfg != {'bolus': 'dual', 'heartlung': 'comp', 'organs': 'comp', 't2s_relaxation': None, 'sequence': 'Eq-SE-EPI', 'magnitude': False}:
        #     return
        try:
           model = dc.AortaModel(**cnfg)
        except InvalidConfiguration:
            return
        print(cnfg)
        data = dc.QVALUES | model.map_lexicon(dc.QVALUES)
        t0 = time.perf_counter()
        results = model(data)
        elapsed = time.perf_counter() - t0
        print(f"  [Total model execution time: {elapsed:.4f}s]")
        assert results['S_a'].ndim == 3

    cnt = 0
    for cnfg in dc.AortaModel.configurations():
        cnt += 1
        _test_config(cnfg)
        # if cnt==100:
        #     break

    print(f'Successfully covered {cnt} aorta configurations!')


def test_aorta_function():
    cnfg = {
        'bolus': 'dual', 
        'heartlung': 'pfcomp', 
        'organs': '2cxm', 
        't2s_relaxation': None, 
        'sequence': 'ZTE-3D-IR-SPGR-SS', 
        'magnitude': False, 
        'calibrate': True,
    }
    try:
        model = dc.AortaModel(**cnfg)
    except InvalidConfiguration:
        return
    print(model.inputs())
    print(model.outputs())
    data = dc.QVALUES | model.map_lexicon(dc.QVALUES)
    results = model(data)
    
    plt.plot(results['tS_a'], results['S_a'][0, 0, :], 'ro')
    plt.show()


if __name__ == '__main__':
    test_aorta()
    #test_aorta_times()
    #test_aorta_function()

    print('All model coverage tests passed!!')