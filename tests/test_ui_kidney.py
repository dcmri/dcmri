import os

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
from dcmri import Kidney


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

    for kin in Kidney.configs['kinetics']:
        for seq in Kidney.configs['sequence']:
            model = dc.Kidney(kin, seq)
            time = model.time()
            signal = model.predict(time)
            model.train(time, signal)
            model.plot(time, signal, show=DEBUG)
            cost = model.cost(time, signal)
            print(kin, seq, cost)
            assert cost < 1e-6

def test_api():
    model = dc.Kidney()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C.ndim == 2 # Should return [compartment, time]
    assert len(R1) == len(t)
    assert len(S) == len(t)

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
        dc.Kidney(sequence='InversionRecovery')
    except ValueError:
        pass 
    try:
        dc.Kidney(kinetics='Liver')
    except ValueError:
        pass 

    # # Predict out of AIF range
    # time = (np.arange(1000), np.arange(1000))
    # try:
    #     dc.Kidney().predict(time)
    # except ValueError:
    #     pass

    # # Train out of AIF range
    # signal = (np.arange(1000), np.arange(1000))
    # try:
    #     dc.Kidney().train(time, signal)
    # except ValueError:
    #     pass

def test_function():

    # Simulation parameters
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR = 15, 0.005 # Defaults
    
    # Input signals
    rp = dc.relaxivity(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif_tristan(aif_time, BAT=10)
    aif_R1 = R10a + rp * aif_conc
    aif_signal = dc.Signal(seq)(R1=aif_R1, S0=S0a, FA=FA, TR=TR, TE=0, B1corr=B1a)
    aif = dc.Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)

    # Kidney signals
    params = {
        'dt': dt, 
        'c_a': aif_conc,
        'field_strength': B0,
        'agent': agent,
        'FA': FA, 
        'TR': TR,
        'TE': 0,
        'S0': 5,
    }

    # Tissue model
    model = dc.Kidney(sequence=seq, **params)
    time = model.time()
    signal = model.predict(time)
    
    # Fit with AIF and VIF signal
    model.train(time, signal, aif)
    model.plot(time, signal)
    assert model.cost(time, signal) < 1

    # Test some training options
    model = dc.Kidney(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'Fp': [0,1], 'Ft': None, 'S0':[0,5]})


if __name__ == "__main__":

    # Coverage tests
    test_configs()
    test_api()
    test_exceptions()
    
    # Functional tests
    test_function()
    
    print('All ui_kidney tests passed!!')

