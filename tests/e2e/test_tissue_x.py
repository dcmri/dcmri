import os
import itertools
from joblib import parallel_config, Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
from dcmri.magnetization import Signal
from dcmri.core import Input
from dcmri import TissueX as Model
from dcmri import aif


DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def test_coverage():
    def run_single_config(kin, wex, seq, r2s):
        print(kin, wex, seq, r2s)
        model = Model(kin, wex, seq, r2s)
        time = model.time()
        signal = model.predict(time)
        _, sdev, _ = model.train(time, signal, xtol=0.1)
        model.plot(time, signal, sdev=sdev, round_to=3, show=DEBUG)
        cost = model.cost(time, signal)
        return cost

    # 1. Create the Cartesian product of all configurations
    values = Model.configs.values()

    # 2. Run in parallel
    if DEBUG:
        results = [
            run_single_config(*cnfgs) 
            for cnfgs in itertools.product(*values)
        ]
    else:
        results = [
            run_single_config(*cnfgs) 
            for cnfgs in itertools.product(*values)
        ]
        # results = Parallel(n_jobs=-1)(
        #     delayed(run_single_test)(*cnfgs) 
        #     for cnfgs in itertools.product(*values)
        # )

    # 3. Assertions (collectively)
    for cost in results:
        assert cost < 1e-9, f"Model cost {cost} exceeded threshold!"



def test_api():

    # Run some options
    Model(vb=np.zeros((5,5)))
    Model(vb=np.zeros((5,5)), shape=(5,5))

    model =Model()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    M = model.magn()
    S = model.signal()

    assert C.ndim in [1,2]
    assert R1.size == t.size
    assert S.size == t.size
    assert M.size == t.size

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

def test_exceptions():
    # Invalid Config
    try:
       Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
    try:
       Model(kinetics='X')
    except ValueError:
        pass 
    else:
        assert False
    try:
       Model(water_exchange='X')
    except ValueError:
        pass 
    else:
        assert False
    try:
       Model(shape=(10,10,10,10))
    except ValueError:
        pass 
    else:
        assert False
    try:
       Model(vb=np.zeros((5,5)), vi=np.zeros((6,5)),)
    except ValueError:
        pass 
    else:
        assert False
    try:
       Model(vb=np.zeros((5,5)), shape=(6,6))
    except ValueError:
        pass 
    else:
        assert False
    


def test_function():

    # Generate an AIF
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR, TE = 15, 0.005, 0.0 # Defaults

    rp = dc.const.r1(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = aif.tristan(aif_time, BAT=10)

    params = {
        'dt': dt, 
        'c_a': aif_conc, 
        'field_strength': B0,
        'agent': agent,
        'FA': FA, 
        'TR': TR,
        'TE': TE,
        'S0': 5,
    }

    seq = '3D-SPGR-SS'

    model =Model('2CX', 'FF', seq, **params)
    time = model.time()
    signal = model.predict(time)

    # Generate AIF with the same signal model and parameters
    aif_R1 = R10a + rp * aif_conc
    aif_signal = Signal(seq)(R1=aif_R1, S0=S0a, FA=FA, TR=TR, TE=TE, B1corr=B1a)

    # This is OK
    # ca_rec = dc.SignalToConc(seq)(aif_signal, FA=FA, TR=TR, R10=R10a, r1=rp, B1corr=B1a)
    # err = np.linalg.norm(aif_conc-ca_rec) / np.linalg.norm(aif_conc)

    # Fit with generated AIF signal
    aif_ = Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)
    model.train(time, signal, aif_)
    model.plot(time, signal)
    assert model.cost(time, signal) < 0.01

    # Test some training options
    model =Model(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'PS': [0,1], 'vb': None, 'S0':[0,5]})

    # Test model selection
    # Generate data with a simple model
    model =Model('HF', 'FR', dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)

    # Use model selection on a more complex model to find the optimal configuration
    configs = ['kinetics', 'water_exchange']
    model =Model('2CX', 'RR', dt=dt, c_a=aif_conc)
    vals, sdev, pcov, best_config = model.train(time, signal, n0=1, configs=configs, xtol=0.01) 
    assert best_config == ('HF', 'FR')  

    # Try BIC
    model =Model('2CX', 'RR', dt=dt, c_a=aif_conc)
    vals, sdev, pcov, best_config = model.train(time, signal, n0=1, configs=configs, select='BIC', xtol=0.01) 
    assert best_config == ('HF', 'FR')     

    # Run again with threading so all lines show up in the coverage
    with parallel_config(backend='threading'):
        model =Model('2CX', 'RR', dt=dt, c_a=aif_conc)
        model.train(time, signal, n0=1, configs=configs, xtol=0.1)


if __name__ == "__main__":

    # Coverage tests
    test_coverage()
    test_api()
    test_exceptions()
    
    # # Functional tests
    test_function()
    
    print('All ui_tissue tests passed!!')

