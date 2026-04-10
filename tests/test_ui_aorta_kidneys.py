import os
import itertools

import matplotlib.pyplot as plt
from dcmri import AortaKidneys as Model


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

    # Create some asymmetry for testing
    pars = {'DRF':0.25, 'vp_rk':0.3}
    SNR = 10

    values = Model.configs.values()
    for cnfgs in itertools.product(*values):
        model = Model(*cnfgs, **pars)
        time = model.time()
        signal = model.predict(time)
        # for roi, sig in signal.items():
        #     signal[roi] = utils.add_noise(sig, sig[0] / SNR)
        model.train(time, signal, verbose=VERBOSE, xtol=1e-2)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(cnfgs, cost)
        assert cost < 5

    # Staged Training
    model = Model(**pars)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, verbose=VERBOSE, xtol=0.01)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print('staged', cost)
    assert cost < 5

def test_api():
    model = Model()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C['aorta'].ndim in [1,2] 
    assert C['kidney_left'].ndim == 2
    assert len(R1['kidney_left']) == len(t['kidney_left'])
    assert len(S['kidney_left']) == len(t['kidney_left'])

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(t, S, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        plt.ion() # Turn interactive mode on
        model.plot(t, S, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)

def test_exceptions():
    # Invalid Config
    try:
        Model(organs='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        Model(heartlung='X')
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(kidneys='X')
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = Model(sequence='3D-SPGR-SSI')
        t, s = model.time(), model.signal()
        model.train(t, s, bounds={'S0_a': None})
    except ValueError:
        pass
    else:
        assert False

if __name__ == "__main__":

    test_configs()
    test_api()
    test_exceptions()
    
    print('All ui_aorta_portal_liver tests passed!!')

