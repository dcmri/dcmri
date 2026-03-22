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

    ns_opts = {
        '2I-EC-HF': [None],
        '2I-EC': [None],
        '2I-IC-U': [None, 'U'],
        '2I-IC-HF': [None, 'U', 'E', 'UE'],
        '2I-IC': [None, 'U', 'E', 'UE'],
    }

    for seq in ['SR', 'SS', 'SSI', 'lin']:
        for kin in ns_opts.keys():
            for ns in ns_opts[kin]:
                model = dc.AortaPortalLiver(kinetics=kin, sequence=seq)
                time = model.time()
                signal = model.predict(time)
                model.train(time, signal)
                model.plot(time, signal)
                cost = model.cost(time, signal)
                print(kin, ns, seq, cost)
                assert cost < 5

    # Test Variations
    model = dc.AortaPortalLiver(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True)
    model.plot(time, signal)
    cost = model.cost(time, signal)
    print(kin, ns, seq, cost)
    assert cost < 5

def test_api():
    model = dc.AortaPortalLiver()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C[0].ndim in [1,2] 
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
        dc.AortaPortalLiver(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        dc.AortaPortalLiver(kinetics='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaPortalLiver(kinetics='1I-EC')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaPortalLiver(non_stationary='Z')
    except ValueError:
        pass 
    else:
        assert False

    # 2. Invalid Parameter
    try:
        dc.AortaPortalLiver(fake_parameter=99)
    except ValueError:
        pass
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = dc.AortaPortalLiver(sequence='SSI')
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

