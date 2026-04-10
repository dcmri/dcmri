import os

import matplotlib.pyplot as plt
import dcmri as dc
from dcmri import AortaLiver2scan


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

    for kin in AortaLiver2scan.configs['kinetics']:
        for seq in AortaLiver2scan.configs['sequence']:
            for ns in AortaLiver2scan.configs['non_stationary']:
                if 'EC' in kin and ns is not None:
                    continue
                if 'U' in kin and ns is not None:
                    if 'E' in ns:
                        continue
                model = AortaLiver2scan(kinetics=kin, sequence=seq, S02_l=2, S02_a=3)
                time = model.time()
                signal = model.predict(time)
                R1 = model.relax()
                R102a=R1['aorta', 2][0]
                R102l=R1['liver', 2][0]
                model.train(time, signal, R102a=R102a, R102l=R102l, xtol=0.01)
                model.plot(time, signal, show=DEBUG)
                cost = model.cost(time, signal)
                print(kin, ns, seq, cost)
                assert cost < 5

    # Test Variations (override parameter and staged training)
    model = AortaLiver2scan(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(kin, ns, seq, cost)
    assert cost < 5

    # Concentration with single compartment
    AortaLiver2scan('1I-EC').conc()

def test_api():

    # Test Forward API outputs
    model = AortaLiver2scan()
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C['aorta', 1].ndim == 1
    assert C['liver', 1].ndim == 2
    assert len(R1['liver', 1]) == len(t['liver', 1])
    assert len(S['liver', 1]) == len(t['liver', 1])

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
        dc.AortaLiver2scan(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        dc.AortaLiver2scan(kinetics='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaLiver2scan(kinetics='2I-EC')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaLiver2scan(non_stationary='Z')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = dc.AortaLiver2scan(sequence='SSI')
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

