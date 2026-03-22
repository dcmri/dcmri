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

    # Create some asymmetry for testing
    pars = {'DRF':0.25, 'vp_rk':0.3}

    # All configs
    for org in ['comp','2cxm']:
        for hl in ['comp', 'pfcomp', 'chain']:
            for kid in ['2CF', 'HF']:
                for seq in ['SR', 'SS', 'SSI', 'lin']:
                    for agent in ['gadoterate', 'gadoxetate']:
                        model = dc.AortaKidneys(org, hl, kid, seq, agent, **pars)
                        time = model.time()
                        signal = model.predict(time)
                        bnds = {'S0_a': [0,5]} if seq=='SSI' else None
                        model.train(time, signal, bounds=bnds, xtol=0.01)
                        model.plot(time, signal)
                        cost = model.cost(time, signal)
                        print(org, hl, kid, seq, agent, cost)
                        assert cost < 5

    # Staged Training
    model = dc.AortaKidneys(**pars)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, xtol=0.1)
    model.plot(time, signal)
    cost = model.cost(time, signal)
    print('staged', cost)
    assert cost < 5

def test_api():
    model = dc.AortaKidneys()
    
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
        dc.AortaKidneys(organs='X')
    except ValueError:
        pass 
    else:
        assert False
        
    try:
        dc.AortaKidneys(heartlung='X')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaKidneys(kidneys='X')
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.AortaKidneys(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # 2. Invalid Parameter
    try:
        dc.AortaKidneys(fake_parameter=99)
    except ValueError:
        pass
    else:
        assert False

    # SSI sequence model with fixed S0
    try:
        model = dc.AortaKidneys(sequence='SSI')
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

