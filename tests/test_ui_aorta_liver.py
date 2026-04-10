import os

import matplotlib.pyplot as plt
import dcmri as dc
from dcmri import AortaLiver


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

    # kin = '1I-IC-HF'
    # ns = 'UE'
    # seq = '3D-SPGR-SSI'

    # model = dc.AortaLiver(kin, ns, seq)
    # time = model.time()
    # signal = model.predict(time)
    # model.train(time, signal, staged=True, verbose=2, xtol=0.001)
    # model.plot(time, signal, show=DEBUG)
    # cost = model.cost(time, signal)
    # print(kin, ns, seq, cost)
    # #assert cost < 5

    # return

    for kin in AortaLiver.configs['kinetics']:
        for seq in AortaLiver.configs['sequence']:
            for ns in AortaLiver.configs['non_stationary']:
                if 'EC' in kin and ns is not None:
                    continue
                elif 'U' in kin and ns is not None:
                    if 'E' in ns:
                        continue
                model = dc.AortaLiver(kin, ns, seq)
                time = model.time()
                signal = model.predict(time)
                #bounds = {'S0_a': [0, 5]} if seq=='3D-SPGR-SSI' else None
                model.train(time, signal, verbose=0, xtol=0.01)
                model.plot(time, signal, show=DEBUG)
                cost = model.cost(time, signal)
                print(kin, ns, seq, cost)
                assert cost < 5

    # Test Variations (override parameter and staged training)
    model = dc.AortaLiver(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, verbose=VERBOSE, xtol=0.1)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print(kin, ns, seq, cost)
    #assert cost < 5

def test_api():
    model = dc.AortaLiver()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C['aorta'].ndim == 1
    assert C['liver'].ndim == 2
    assert len(R1['liver']) == len(t['liver'])
    assert len(S['liver']) == len(t['liver'])

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
        dc.AortaLiver(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        dc.AortaLiver(kinetics='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaLiver(kinetics='2I-EC')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaLiver(non_stationary='Z')
    except ValueError:
        pass 
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = dc.AortaLiver(sequence='3D-SPGR-SSI', CO=50)
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
    
    print('All ui_aorta_liver tests passed!!')

