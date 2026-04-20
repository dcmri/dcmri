import os
import itertools

import matplotlib.pyplot as plt
from dcmri import Aorta as Model


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
        # if cnfgs != ('comp', 'comp', '3D-SPGR-SSI'):
        #     continue
        print(cnfgs)
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
        assert cost < 5

    # Variations
    model = Model(CO=50)

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
    # Invalid Config
    try:
        Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        Model(organs='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(heartlung='Z')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = Model(sequence='3D-SPGR-SSI', CO=50)
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
    
    print('All ui_aorta tests passed!!')

