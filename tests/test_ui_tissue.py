import os
from joblib import parallel_config

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc

from dcmri import Tissue


DEBUG = True

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def test_coverage():

    shape = None
    for kin in Tissue.configs['kinetics']:
        for wex in Tissue.configs['water_exchange']:
            for seq in Tissue.configs['sequence']:
                for r2s in Tissue.configs['transverse_relaxation']:
                    print(kin, wex, seq, r2s)
                    model = Tissue(shape, kin, wex, seq, r2s)
                    time = model.time()
                    signal = model.predict(time)
                    _, sdev, _ = model.train(time, signal, xtol=0.1)
                    # sdev = None
                    model.plot(time, signal, sdev=sdev, round_to=3, show=DEBUG)
                    cost = model.cost(time, signal)
                    # print('Cost', cost)
                    assert cost < 1e-9 


def test_api():
    model = dc.Tissue()
    
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
        dc.Tissue(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
    try:
        dc.Tissue(kinetics='X')
    except ValueError:
        pass 
    else:
        assert False
    try:
        dc.Tissue(water_exchange='X')
    except ValueError:
        pass 
    else:
        assert False

    time = (np.arange(1000), np.arange(1000))
    # # Predict out of AIF range
    # try:
    #     dc.Tissue().predict(time)
    # except ValueError:
    #     pass
    # else:
    #     assert False

    # Train out of AIF range
    signal = (np.arange(1000), np.arange(1000))
    # try:
    #     dc.Tissue().train(time, signal)
    # except ValueError:
    #     pass
    # else:
    #     assert False

def test_function():

    # Generate an AIF
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR, TE = 15, 0.005, 0.0 # Defaults

    rp = dc.relaxivity(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif_tristan(aif_time, BAT=10)

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

    model = dc.Tissue('2CX', 'FF', seq, **params)
    time = model.time()
    signal = model.predict(time)

    # Generate AIF with the same signal model and parameters
    aif_R1 = R10a + rp * aif_conc
    aif_signal = dc.Signal(seq)(R1=aif_R1, S0=S0a, FA=FA, TR=TR, TE=TE, B1corr=B1a)

    # This is OK
    # ca_rec = dc.SignalToConc(seq)(aif_signal, FA=FA, TR=TR, R10=R10a, r1=rp, B1corr=B1a)
    # err = np.linalg.norm(aif_conc-ca_rec) / np.linalg.norm(aif_conc)

    # Fit with generated AIF signal
    aif = dc.Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)
    model.train(time, signal, aif)
    model.plot(time, signal)
    assert model.cost(time, signal) < 0.01

    # Test some training options
    model = dc.Tissue(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'PS': [0,1], 'vb': None, 'S0':[0,5]})

    # Test model selection
    # Generate data with a simple model
    model = dc.Tissue('HF', 'FR', dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)

    # Use model selection to train a more complex model
    model = dc.Tissue('2CX', 'RR', dt=dt, c_a=aif_conc)
    vals, sdev, pcov, best_model = model.train(time, signal, n0=1, modsel='AIC', xtol=0.01) 
    assert best_model == ('HF', 'FR')  

    # Try BIC
    model = dc.Tissue('2CX', 'RR', dt=dt, c_a=aif_conc)
    vals, sdev, pcov, best_model = model.train(time, signal, n0=1, modsel='BIC', xtol=0.01) 
    assert best_model == ('HF', 'FR')     

    # Run again with threading so all lines show up in the coverage
    with parallel_config(backend='threading'):
        model = dc.Tissue('2CX', 'RR', dt=dt, c_a=aif_conc)
        model.train(time, signal, n0=1, modsel='AIC', xtol=0.1)


if __name__ == "__main__":

    # Coverage tests
    # test_coverage()
    # test_api()
    # test_exceptions()
    
    # # Functional tests
    test_function()
    
    print('All ui_tissue tests passed!!')

