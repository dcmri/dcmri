import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
from dcmri import Liver as Model
from dcmri import pk
from dcmri.magnetization import Signal
from dcmri.core import Input
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


def test_configs():

    values = Model.configs.values()
    for cnfgs in itertools.product(*values):
        kin, ns = cnfgs[0], cnfgs[1]
        if 'EC' in kin and ns is not None:
            continue
        elif 'U' in kin and ns is not None:
            if 'E' in ns:
                continue
        model = Model(*cnfgs)
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, verbose=VERBOSE, xtol=0.01)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(cnfgs, cost)
        assert cost < 5

def test_api():
    model = Model()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C.ndim in [1,2] 
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
        Model(sequence='InversionRecovery')
    except ValueError:
        pass 
    else:
        assert False
    try:
        Model(kinetics='Liver')
    except ValueError:
        pass 
    else:
        assert False

    # Different length inputs
    try:
        Model(c_a=np.arange(10))
    except:
        pass
    else:
        assert False

def test_function():

    # Simulation parameters
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR = 15, 0.005 # Defaults
    
    # Input signals
    rp = dc.const.r1(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = aif.tristan(aif_time, BAT=10)
    vif_conc = pk.flux_chain(aif_conc, 10, 0.5, dt=dt)
    aif_R1 = R10a + rp * aif_conc
    vif_R1 = R10a + rp * vif_conc
    aif_signal = Signal(seq)(R1=aif_R1, S0=S0a, FA=FA, TR=TR, TE=0, B1corr=B1a)
    vif_signal = Signal(seq)(R1=vif_R1, S0=S0a, FA=FA, TR=TR, TE=0, B1corr=B1a)
    aif_ = Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)
    vif = Input(vif_signal, aif_time, R10=R10a, B1corr=B1a)

    # Liver signals
    params = {
        'dt': dt, 
        'c_a': aif_conc,
        'c_v': vif_conc, 
        'field_strength': B0,
        'agent': agent,
        'FA': FA, 
        'TR': TR,
        'TE': 0,
        'S0': 5,
    }

    # Tissue model
    model = Model(sequence=seq, **params)
    time = model.time()
    signal = model.predict(time)
    
    # Fit with AIF and VIF signal
    model.train(time, signal, aif_, vif)
    model.plot(time, signal)
    assert model.cost(time, signal) < 1

    # Test some training options
    model = Model(dt=dt, c_a=aif_conc, c_v=vif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'Fp': [0,1], 'fa': None, 'S0':[0,5]})


if __name__ == "__main__":

    # # Coverage tests
    test_configs()
    test_api()
    test_exceptions()
    
    # Functional tests
    test_function()
    
    print('All ui_liver tests passed!!')

