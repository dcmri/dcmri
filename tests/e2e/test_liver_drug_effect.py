import os
import itertools

import matplotlib.pyplot as plt
from dcmri import LiverDrugEffect as Model


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

    values = Model.configs.values()
    for cnfgs in itertools.product(*values):
        seq = cnfgs[0]
        model = Model(*cnfgs)
        time = model.time()
        signal = model.predict(time)
        bounds = {'c_S0_a': [0, 5], 'd_S0_a': [0, 5]} if seq=='3D-SPGR-SSI' else None
        model.train(time, signal, bounds=bounds, verbose=2, xtol=0.1)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(cnfgs, cost)
        assert cost < 5

    # Test Variations (override parameter and staged training)
    model = Model(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(cost)
    assert cost < 5

def test_api():

    # Test Forward API outputs
    model = Model()
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C['ctrl', 'aorta'].ndim == 1
    assert C['ctrl', 'liver'].ndim == 2
    assert len(R1['ctrl', 'liver']) == len(t['ctrl', 'liver'])
    assert len(S['ctrl', 'liver']) == len(t['ctrl', 'liver'])

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
        Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    model = Model(sequence='3D-SPGR-SSI')
    t, s = model.time(), model.signal()
    try:
        model.train(t, s, bounds={'c_S0_a': None})
    except ValueError:
        pass
    else:
        assert False

if __name__ == "__main__":

    test_configs()
    test_api()
    test_exceptions()
    
    print('All liver_drug_effect tests passed!!')

