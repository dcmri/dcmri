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

    for org in ['comp', '2cxm']:
        for hl in ['pfcomp', 'chain']:
            for seq in ['SR', 'SS', 'SSI', 'lin']:
                model = dc.Aorta(organs=org, heartlung=hl, sequence=seq)
                time = model.time()
                signal = model.predict(time)
                model.train(time, signal)
                model.plot(time, signal)
                cost = model.cost(time, signal)
                # print(org, hl, seq, cost)
                assert cost < 2

    # Variations
    model = dc.Aorta(CO=50)

def test_api():
    model = dc.Aorta()
    
    # Test Forward API outputs
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C.ndim in [1,2] 
    assert len(R1) == len(t)
    assert len(S) == len(t)

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
        dc.Aorta(sequence='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        dc.Aorta(organs='Y')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.Aorta(heartlung='Z')
    except ValueError:
        pass 
    else:
        assert False

    # 2. Invalid Parameter
    try:
        dc.Aorta(fake_parameter=99)
    except ValueError:
        pass
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = dc.Aorta(sequence='SSI', CO=50)
        t, s = model.time(), model.signal()
        model.train(t, s, bounds={'S0': None})
    except ValueError:
        pass
    else:
        assert False

if __name__ == "__main__":

    test_configs()
    test_api()
    test_exceptions()
    
    print('All ui_aorta tests passed!!')

