import os
import itertools
from joblib import parallel_config, Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np

import dcmri as dc
from dcmri import TissueX as Model


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
    # if cnfg != ('2CX', 'FF', 'lin', 'Eq-GE-EPI', True):
    #     return 0
    print(cnfg)
    model = Model(*cnfg)
    time = model.time()
    signal = model.predict(time)
    _, sdev, _ = model.train(time, signal, xtol=0.1)
    model.plot(time, signal, sdev=sdev, round_to=3, show=DEBUG)
    cost = model.cost(time, signal)
    # other API
    model.signal()
    model.relax()
    model.mz()
    model.conc()
    print(cnfg, cost)
    assert cost < 1e-9, f"Cost {cost} of model {cnfg} exceeded threshold!"
    return cost


def test_coverage():

    # Use generated data
    model = Model()
    time = model.time()
    signal = model.predict(time)
    _, sdev, _ = model.train(time, signal, xtol=0.1)
    model.plot(time, signal, sdev=sdev, round_to=3, show=DEBUG)
    cost = model.cost(time, signal)
    assert cost < 1e-9, f"Cost {cost} exceeded threshold!"
    # other API
    model.signal()
    model.relax()
    model.mz()
    model.conc()

    # Derive shape from data
    model = Model()
    time = np.arange(10)
    signal = np.ones((5,10))
    aif = {'signal': np.ones(20), 'time': np.arange(20)}
    model.train(time, signal, aif, xtol=0.1)
    model.cost(time, signal)


def test_api():

    # Run some options
    model = Model(vb=np.zeros((5,5)))
    model = Model(vb=np.zeros((5,5)), shape=(5,5))

    model.print_params()
    model.params()
    model.params('vb')
    model.params('vb', 'TS')
    try:
        model.params('XX')
    except:
        pass
    else:
        assert False

    model = Model()

    # print_params()
    model.print_params('vb', 'R1b', 'TS', fixed_only=True)
    model.print_params('vb', 'R1b', 'TS', free_only=True)

    
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

def test_exceptions():
    # # Invalid Config
    # try:
    #    Model(sequence='X')
    # except:
    #     pass 
    # else:
    #     assert False
    # try:
    #    Model(kinetics='X')
    # except:
    #     pass 
    # else:
    #     assert False
    # try:
    #    Model(water_exchange='X')
    # except:
    #     pass 
    # else:
    #     assert False
    try:
        Model(shape=(10,10,10,10))
    except:
        pass 
    else:
        assert False
    try:
        Model(vb=np.zeros((5,5)), vi=np.zeros((6,5)),)
    except:
        pass 
    else:
        assert False
    try:
        Model(vb=np.zeros((5,5)), shape=(6,6))
    except:
        pass 
    else:
        assert False

    try:
        model = Model(shape=(10,))
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, free={'TR':[0, 1]})
    except:
        pass 
    else:
        assert False

    try:
        model = Model(shape=(10,))
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, select='X')
    except:
        pass 
    else:
        assert False
    


def test_function():

    # Generate an AIF
    dt, tmax, B0, agent, R1ba, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR, TE = 15, 0.005, 0.0 # Defaults
    CONSTANTS = {'Fw': 0, 'v': 1, 'me': 1, 'noise_sdev':0}

    rp = dc.r1(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.tristan(aif_time, BAT=10)

    params = {
        'dt': dt, 
        'ca': aif_conc, 
        # 'field_strength': B0,
        # 'agent': agent,
        'FA': FA, 
        'TR': TR,
        'TE': TE,
        'S0': 5,
    }

    seq = '3D-SPGR-SS'

    model = Model('2CX', 'FF', sequence=seq, **params)
    time = model.time()
    signal = model.predict(time)

    # Generate AIF with the same signal model and parameters
    aif_R1 = R1ba + rp * aif_conc
    aif_R2s = np.zeros_like(aif_R1) # required for 
    aif_signal = dc.Signal(seq)(R1=aif_R1, R2s=aif_R2s, S0=S0a, FA=FA, TR=TR, TE=TE, B1corr=B1a, **CONSTANTS)

    # This is OK
    # ca_rec = dc.SignalToConc(seq)(aif_signal, FA=FA, TR=TR, R1b=R1ba, r1=rp, B1corr=B1a)
    # err = np.linalg.norm(aif_conc-ca_rec) / np.linalg.norm(aif_conc)

    # Fit with generated AIF signal
    aif = {'signal': aif_signal, 'time': aif_time, 'R1b': R1ba, 'B1corr': B1a}
    model.train(time, signal, aif)
    model.plot(time, signal)
    assert model.cost(time, signal) < 0.01

    # Test some training options
    model = Model(dt=dt, ca=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'PS': [0,1], 'vb': None, 'S0':[0,5]})

    # Test model selection
    # Generate data with a simple model
    model =Model('HF', 'FR', dt=dt, ca=aif_conc)
    time = model.time()
    signal = model.predict(time)

    # Use model selection on a more complex model to find the optimal configuration
    configs = ['kinetics', 'water_exchange']
    model = Model('2CX', 'RR', dt=dt, ca=aif_conc)
    vals, sdev, pcov, best_config = model.train(time, signal, n0=1, configs=configs, xtol=0.01) 
    assert best_config == ('HF', 'FR')  

    # Try BIC
    model =Model('2CX', 'RR', dt=dt, ca=aif_conc)
    vals, sdev, pcov, best_config = model.train(time, signal, n0=1, configs=configs, select='BIC', xtol=0.01) 
    assert best_config == ('HF', 'FR')     

    # Run again with threading so all lines show up in the coverage
    print('Running with threading for coverage..')
    with parallel_config(backend='threading'):
        model =Model('2CX', 'RR', dt=dt, ca=aif_conc)
        model.train(time, signal, n0=1, configs=configs, xtol=0.1)


def test_tutorial():
    tmax = 120
    dt = 1.5
    t = np.arange(0, tmax, dt)
    ca_A = dc.tristan(t, BAT=20, CO=150)
    gm = {'kinetics': 'NX', 'vb': 0.05, 'Fb': 0.01}

    gm_A = Model(ca=ca_A, dt=dt, **gm)
    time = gm_A.time()
    signal = gm_A.predict(time)

    gm_A = Model(ca=ca_A, dt=dt, kinetics='NX', vb=0.1, Fb=0.03)  
    gm_A.train(time, signal)
    print(gm_A.params('vb'), gm['vb'])
    print(gm_A.params('Fb'), gm['Fb'])
    print(gm_A.cost(time, signal))
    #gm_A.plot(time, signal, round_to=2)

    assert np.isclose(gm_A.params('vb'), gm['vb'], atol=0.01)
    assert np.isclose(gm_A.params('Fb'), gm['Fb'], atol=0.01)


def test_configs():
    values = Model.configs.values()

    Parallel(n_jobs=-1)(
        delayed(_run_single_config)(cnfgs)
        for cnfgs in itertools.product(*values)
    )

    # [
    #     _run_single_config(cnfgs) 
    #     for cnfgs in itertools.product(*values)
    # ]


if __name__ == "__main__":
    test_coverage()
    test_function()
    test_tutorial()
    test_api()
    test_exceptions()
    test_configs()

    print('All tissue_x tests passed!!')

