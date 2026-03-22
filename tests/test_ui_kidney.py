import os

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc


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

    for kin in ['2CF', 'HF']:
        for seq in ['SR', 'SS', 'lin']:
            model = dc.Kidney(kinetics=kin, sequence=seq)
            time = model.time()
            signal = model.predict(time)
            model.train(time, signal)
            model.plot(time, signal)
            assert model.cost(time, signal) < 1e-6

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

    # 2. Invalid Parameter
    try:
        dc.Kidney(fake_parameter=99)
    except ValueError:
        pass

    # Predict out of AIF range
    time = (np.arange(1000), np.arange(1000))
    try:
        dc.Kidney().predict(time)
    except ValueError:
        pass

    # Train out of AIF range
    signal = (np.arange(1000), np.arange(1000))
    try:
        dc.Kidney().train(time, signal)
    except ValueError:
        pass

def test_function():

    # Generate an AIF
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR, TC, TP = 15, 0.005, 0.2, 0.05 # Defaults

    rp = dc.relaxivity(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif_tristan(aif_time)
    aif_R1 = R10a + rp * aif_conc
    params = {
        'SR': {'FA': FA, 'TR': TR, 'TC': TC, 'TP': TP},
        'SS': {'FA': FA, 'TR': TR},
        'lin': {},
    }
    aif_signal = {
        'SR': dc.signal_spgr(S0a, aif_R1, TC, TR, B1a * FA, TP),
        'SS': dc.signal_ss(S0a, aif_R1, TR, B1a * FA),
        'lin': dc.signal_lin(S0a, aif_R1)
    }
    aif = {'time': aif_time, 'R10': R10a, 'B1corr': B1a}

    for seq in ['SS', 'lin', 'SR']:
        model = dc.Kidney(
            dt=dt, 
            c_a=aif_conc, 
            field_strength=B0,
            agent=agent,
            sequence=seq,
            **params[seq],
        )
        time = model.time()
        signal = model.predict(time)

        # Generate AIF
        aif['signal'] = aif_signal[seq]
        
        # Fit with generated AIF signal
        model.train(time, signal, aif)
        model.plot(time, signal)
        assert model.cost(time, signal) < 1e-6

    # Test some training options
    model = dc.Kidney(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10, bounds={'Fp': [0,1], 'FF': None, 'S0':[0,5]})


if __name__ == "__main__":

    # Coverage tests
    test_configs()
    test_api()
    test_exceptions()
    
    # Functional tests
    test_function()
    
    print('All ui_kidney tests passed!!')

