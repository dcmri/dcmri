import os
import itertools

import numpy as np
import matplotlib.pyplot as plt

from dcmri import AortaLiverDrug as Model
from dcmri.e2e.aorta_liver_drug import _div


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
        model.train(time, signal, verbose=2, xtol=0.1)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(cnfgs, cost)
        model.conc()
        model.relax()
        model.signal()
        assert cost < 5

    # Test Variations (override parameter and 2 runs)
    model = Model(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n_runs=2, verbose=2, xtol=0.1)
    model.plot(time, signal, show=DEBUG, clim=[0,1])
    cost = model.cost(time, signal)
    print(cost)
    assert cost < 6
    model.export_params(desc=True)

    model = Model(CO=50)
    time = model.time()
    signal = model.predict(time['ctrl', 'aorta'])
    vals, sdev, pcov = model.train(time['ctrl', 'aorta'], signal, n_runs=2, verbose=2, xtol=0.1)
    model.plot(time['ctrl', 'aorta'], signal, show=DEBUG, clim=[0,1])
    cost = model.cost(time['ctrl', 'aorta'], signal)
    print(cost)
    assert cost < 6
    model.export_params(sdev=sdev, desc=True)

def test_api():

    # Test Forward API outputs
    model = Model()
    t = model.time()
    S = model.signal()

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


    # Tests a standard division case
    result = _div(6, 0)
    assert np.isinf(result)  # 1/0 in numpy results in infinity (inf)

# def test_exceptions():
#     # Invalid Config
#     try:
#         Model(sequence='X')
#     except ValueError:
#         pass 
#     else:
#         assert False

if __name__ == "__main__":

    test_configs()
    test_api()
    # test_exceptions()
    
    print('All liver_drug_effect tests passed!!')

