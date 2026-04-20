import os
import itertools

import matplotlib.pyplot as plt
from dcmri import AortaLiverDynamic as Model


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
        kin, ns, seq = cnfgs[0], cnfgs[1], cnfgs[2]
        if 'EC' in kin and ns is not None:
            continue
        elif 'U' in kin and ns is not None:
            if 'E' in ns:
                continue
        # if not ((kin=='2I-IC') and (ns=='E') and (seq=='3D-SPGR-SS')):
        #     continue
        model = Model(*cnfgs)
        time = model.time()
        signal = model.predict(time)
        R1, R2s = model.relax()
        R102a=R1['aorta', 2][0]
        R102l=R1['liver', 2][0]
        model.train(time, signal, R102a=R102a, R102l=R102l, xtol=0.01)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(kin, ns, seq, cost)
        model.conc()
        model.relax()
        model.signal()
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

    # Concentration with single compartment
    Model('1I-EC').conc()

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

def test_exceptions():
    # Invalid Config
    try:
        Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        Model(kinetics='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(kinetics='2I-EC')
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(non_stationary='Z')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = Model(sequence='SSI')
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
    
    print('All ui_aorta_liver_2scan tests passed!!')

