import os

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc

from joblib import parallel_config


DEBUG = True

if DEBUG:
    # Debugging mode
    VERBOSE = 2
else:
    VERBOSE = 0
    # Allow coverage of plot functions without actually plotting
    import matplotlib
    matplotlib.use('Agg')


def test_configs():

    # for kin in ['HF', 'U', 'FX', 'NX', 'NXP', 'WV', 'HFU', '2CU', '2CX']:
    #     for wex in ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']:
    #         for seq in ['SR', 'SS', 'IR-SS']:
    for kin in ['HF']:
        for wex in ['FF']:
            for seq in ['IR-SS']:
                print(kin, wex, seq)
                model = dc.Tissue(kinetics=kin, water_exchange=wex, sequence=seq)
                time = model.time()
                signal = model.predict(time)
                #_, sdev, _ = model.train(time, signal, xtol=0.1)
                sdev = None
                model.plot(time, signal, sdev=sdev, round_to=3)
                cost = model.cost(time, signal)
                print(kin, wex, seq, cost)
                #S0 assert cost < 100 # liberal for debugging

def test_api():
    model = dc.Tissue()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    M = model.magn()
    S = model.signal()

    assert C.ndim in [1,2]
    assert len(R1) == len(t)
    assert len(S) == len(t)
    assert len(M) == len(t)

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

    # 2. Invalid Parameter
    try:
        dc.Tissue(fake_parameter=99)
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
    FA, TR, TC, TP = 15, 0.005, 0.2, 0.05 # Defaults

    rp = dc.relaxivity(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif_tristan(aif_time)
    aif_R1 = R10a + rp * aif_conc
    params = {
        'SR': {'FA': FA, 'TR': TR, 'TC': TC, 'TP': TP},
        'SS': {'FA': FA, 'TR': TR},
    }
    aif_mz = {
        'SR': dc.Mz('PR', aif_R1, TC=TC, TR=TR, FA=B1a * FA, TP=TP),
        'SS': dc.Mz('SS', aif_R1, TR=TR, FA=B1a * FA),
    }

    for seq in ['SS', 'SR']:
        model = dc.Tissue(
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
        aif_signal = dc.signal(aif_mz[seq], S0=S0a, FA=FA)
        aif = dc.Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)

        # Fit with generated AIF signal
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
    test_configs()
    # test_api()
    # test_exceptions()
    
    # # Functional tests
    # test_function()
    
    print('All ui_tissue tests passed!!')

