import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
from dcmri import CortMed as Model


DEBUG = False

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def test_configs():

    values = Model.configs.values()
    for cnfgs in itertools.product(*values):
        model = Model(*cnfgs)
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, verbose=VERBOSE, xtol=0.01)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(cnfgs, cost)
        model.conc()
        model.relax()
        model.signal()
        assert cost < 1e-6

    model = Model()
    time = model.time()
    time = time['cort']
    signal = model.predict(time)
    model.train(time, signal, verbose=VERBOSE, xtol=0.1)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(cost)
    assert cost < 5

def test_api():
    model = Model()
    
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
    #     Model(sequence='InversionRecovery')
    # except ValueError:
    #     pass 

    model = Model()
    time = model.time()
    signal = model.signal()
    time = np.append(time['cort'], time['cort'].max() * 2)
    try:
        model.predict(time)
    except:
        pass
    else:
        assert False
    try:
        model.train(time, signal)
    except:
        pass
    else:
        assert False


def test_function():

    # Simulation parameters
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R10a, R20sa, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 20, 3, 0.75
    FA, TR, TE = 15, 0.005, 0.002 # Defaults
    
    # Input signals
    rp = dc.r1(B0, 'blood', agent)
    r2s = dc.r2s(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.tristan(aif_time, BAT=10)
    aif_R1 = R10a + rp * aif_conc
    aif_R2s = R20sa + r2s * aif_conc
    aif_signal = dc.Signal(seq)(R1=aif_R1, R2s=aif_R2s, S0=S0a, FA=FA, TR=TR, TE=TE, B1corr=B1a)
    
    # Kidney signals
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

    # Tissue model
    model = Model(sequence=seq, **params)
    time = model.time()
    signal = model.predict(time) 
    
    # Fit with AIF signal
    aif = {'signal': aif_signal, 'time': aif_time, 'R10': R10a, 'B1corr':B1a}
    model.train(time, signal, aif) 
    model.plot(time, signal, show=DEBUG)
    assert model.cost(time, signal) < 1

    # Test some training options
    model = Model(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'Fp': [0,1], 'Tglom': None, 'S0':[0,5]})


if __name__ == "__main__":
    test_function()
    test_configs()
    test_api()
    test_exceptions()
    
    print('All ui_kidney_cort_med tests passed!!')

