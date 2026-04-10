import os

import matplotlib.pyplot as plt
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

    for seq in ['3D-SPGR-SS', '3D-SPGR-SSI']:
        model = dc.LiverDynamicDrugEffect(seq)
        time = model.time()
        signal = model.predict(time)
        R1 = model.relax()
        bounds = {'c_S0_1_a': [0, 5], 'd_S0_1_a': [0, 5]} if seq=='3D-SPGR-SSI' else None
        R102a = [R1['ctrl', 'aorta', 2][0], R1['drug', 'aorta', 2][0]]
        R102l = [R1['ctrl', 'liver', 2][0], R1['drug', 'liver', 2][0]]
        model.train(time, signal, R102a=R102a, R102l=R102l, bounds=bounds, verbose=2, xtol=0.1)
        model.plot(time, signal, show=DEBUG)
        cost = model.cost(time, signal)
        print(seq, cost)
        assert cost < 10

    # Test Variations (override parameter and staged training)
    model = dc.LiverDynamicDrugEffect(CO=99)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, verbose=2, xtol=0.1)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(cost)
    assert cost < 10

def test_api():

    # Test Forward API outputs
    model = dc.LiverDynamicDrugEffect()
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C['ctrl', 'aorta', 1].ndim == 1
    assert C['ctrl', 'liver', 1].ndim == 2
    assert len(R1['ctrl', 'liver', 1]) == len(t['ctrl', 'liver', 1])
    assert len(S['ctrl', 'liver', 1]) == len(t['ctrl', 'liver', 1])

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot(t, S, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        plt.ion() # Turn interactive mode on
        model.plot(t, S, show=DEBUG)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)

def test_exceptions():
    # Invalid Config
    try:
        dc.LiverDynamicDrugEffect(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    model = dc.LiverDynamicDrugEffect(sequence='3D-SPGR-SSI')
    t, s = model.time(), model.signal()
    try:
        model.train(t, s, bounds={'c_S0_1_a': None})
    except ValueError:
        pass
    else:
        assert False

if __name__ == "__main__":
    test_configs()
    test_api()
    test_exceptions()
    
    print('All ui_liver_dynamic_drug_effect tests passed!!')

