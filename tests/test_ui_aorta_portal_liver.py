import os

import matplotlib.pyplot as plt
import dcmri as dc
from dcmri import AortaPortalLiver


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

    # kin = '2I-IC-U'
    # ns = 'UE'
    # seq = '3D-SPGR-SSI'
    # model = AortaPortalLiver(kin, ns, seq)
    # time = model.time()
    # signal = model.predict(time)
    # model.train(time, signal, staged=False, verbose=2, xtol=0.01)
    # model.plot(time, signal, show=DEBUG)
    # cost = model.cost(time, signal)
    # print(kin, ns, seq, cost)
    # #assert cost < 5

    for kin in AortaPortalLiver.configs['kinetics']:
        for seq in AortaPortalLiver.configs['sequence']:
            for ns in AortaPortalLiver.configs['non_stationary']:
                if 'EC' in kin and ns is not None:
                    continue
                if 'U' in kin and ns is not None:
                    if 'E' in ns:
                        continue
                model = dc.AortaPortalLiver(kinetics=kin, sequence=seq)
                time = model.time()
                signal = model.predict(time)
                model.train(time, signal, verbose=0, xtol=0.01)
                model.plot(time, signal, show=DEBUG)
                cost = model.cost(time, signal)
                print(kin, ns, seq, cost)
                assert cost < 5

    # Test Variations
    model = AortaPortalLiver(CO=50)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, staged=True, verbose=VERBOSE, xtol=0.01)
    model.plot(time, signal, show=DEBUG)
    cost = model.cost(time, signal)
    print('staged', cost)
    assert cost < 5

def test_api():
    model = dc.AortaPortalLiver()
    
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

