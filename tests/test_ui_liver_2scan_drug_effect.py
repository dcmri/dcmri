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

    for seq in ['SR', 'SS', 'SSI', 'lin']:
        model = dc.Liver2scanDrugEffect(sequence=seq)
        time = model.time()
        signal = model.predict(time)
        R1 = model.relax()
        bounds = {'c_S0_1_a': [0, 5], 'd_S0_1_a': [0, 5]} if seq=='SSI' else None
        R102a = [R1[1][0], R1[5][0]]
        R102l = [R1[3][0], R1[7][0]]
        model.train(time, signal, R102a=R102a, R102l=R102l, bounds=bounds, verbose=2, xtol=0.1)
        model.plot(time, signal)
        cost = model.cost(time, signal)
        print(seq, cost)
        assert cost < 15

    # Test Variations (override parameter and staged training)
    model = dc.Liver2scanDrugEffect(CO=99)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, verbose=2, xtol=0.1)
    model.plot(time, signal)
    cost = model.cost(time, signal)
    print(cost)
    assert cost < 15

def test_api():

    # Test Forward API outputs
    model = dc.Liver2scanDrugEffect()
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C[0].ndim == 1
    assert C[2].ndim == 2
    assert len(R1[0]) == len(t[0])
    assert len(S[0]) == len(t[0])

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
        dc.Liver2scanDrugEffect(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # 2. Invalid Parameter
    try:
        dc.Liver2scanDrugEffect(fake_parameter=99)
    except ValueError:
        pass
    else:
        assert False

    # SSI sequence model with fixed S0
    model = dc.Liver2scanDrugEffect(sequence='SSI')
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
    
    print('All ui_liver_2scan_drug_effects tests passed!!')

