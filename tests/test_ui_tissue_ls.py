import os

import matplotlib.pyplot as plt
import numpy as np
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


def test_coverage():

    for seq in dc.TissueLS.configs['sequence']:
        model = dc.TissueLS(sequence=seq)
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, tol=0.01)
        model.plot(time, signal, round_to=3)
        cost = model.cost(time, signal)
        assert cost < 10

    # Test Forward API outputs
    model = dc.TissueLS()
    t = model.time()
    C = model.conc()
    R1 = model.relax()
    S = model.signal()

    assert C.ndim in [1,2]
    assert len(R1) == len(t)
    assert len(S) == len(t)

    # Coverage config options
    dc.TissueLS(irf=np.ones((128, 300)))


def test_io():

    # 1D plot
    model = dc.TissueLS()
    t, S = model.time(), model.signal()

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

    # 2D plot
    model = dc.TissueLS((10, 10))
    t, S = model.time(), model.signal()
    pars = model.params()
    c = model.conc()
    r = model.relax()
    s = model.predict(t)
    model.plot(t, s)

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot_2d(t, S, fname=test_plot_file, show=False)
        assert os.path.exists(test_plot_file)
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang your tests
        plt.ion() # Turn interactive mode on
        model.plot_2d(t, S, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        if os.path.exists(test_plot_file):
            os.remove(test_plot_file)

    # 3D plot
    model = dc.TissueLS((10, 10, 10))
    t, S = model.time(), model.signal()

    test_plot_file = "test_plot_output.png"
    try:
        # This hits plt.savefig(fname)
        model.plot_3d(t, S, fname=test_plot_file, show=False)
        assert os.path.exists("Fp_test_plot_output.png")
        
        # This hits plt.show()
        # We wrap this in a check to ensure it doesn't hang your tests
        plt.ion() # Turn interactive mode on
        model.plot_3d(t, S, show=True)
        plt.ioff() # Turn interactive mode off
    finally:
        for par in ['Fp', 've', 'Te', 'S0']:
            test_file = f"{par}_test_plot_output.png"
            if os.path.exists(test_file):
                os.remove(test_file)

def test_exceptions():
    # Invalid Config

    try:
        dc.TissueLS(shape=(1,2,3,4))
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS(irf=np.ones(300), c_a=np.ones(200))
        model.predict(np.arange(30))
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS()
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS((10, ))
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS((10, ))
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS()
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS((10, 10, 10))
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = dc.TissueLS((10, 10))
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        dc.TissueLS(sequence='X')
    except ValueError:
        pass 
    else:
        assert False

    # 2. Invalid Parameter
    try:
        dc.TissueLS(fake_parameter=99)
    except ValueError:
        pass
    else:
        assert False


def test_array_1d():

    # Generate an AIF
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR, TC, TP = 15, 0.005, 0.2, 0.05 # Defaults

    rp = dc.relaxivity(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif_tristan(aif_time)
    aif_R1 = R10a + rp * aif_conc
    params = {
        'SR': {'FA': FA, 'TR': TR, 'TC': TC, 'TP': TP},
        'SS': {'FA': FA, 'TR': TR},
        'lin': {},
    }
    aif_signal = {
        'SR': dc.signal_spgr(S0a, aif_R1, TC, TR, B1a * FA, TP),
        'SS': dc.signal_ss(S0a, aif_R1, TR, B1a * FA),
        'lin': dc.signal_lin(S0a, aif_R1)
    }

    for seq in ['SS', 'SR', 'lin']:
        model = dc.TissueLS(
            dt=dt, 
            c_a=aif_conc, 
            field_strength=B0,
            agent=agent,
            sequence=seq,
            **params[seq],
        )
        time = model.time()
        signal = model.predict(time)

        # Generate AIF
        aif = dc.Input(aif_signal[seq], aif_time, R10=R10a, B1corr=B1a)

        # Fit with generated AIF signal
        model.train(time, signal, aif)
        model.plot(time, signal, round_to=3)
        cost = model.cost(time, signal)
        print(seq, cost)
        print(model.params(iv=True))
        assert cost < 5

    # Test some training options
    model = dc.TissueLS(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10)


def test_array_2d():
    # Generated with a 2CX model
    time, signal, aif, gt = dc.fake_brain(n=64)

    # Compute R10
    R10 = np.zeros_like(gt['T1'], dtype=float)
    np.divide(1, gt['T1'], out=R10, where=gt['T1'] != 0)

    # Match paramaters to the fake_brain simulation
    tissue = dc.TissueLS(
        dt = 1.5,
        sequence = 'SS',
        agent = 'gadodiamide',
        field_strength = 3,
        TR = 0.005,
        FA = 15,
        R10 = R10,
    )

    aif = dc.Input(aif, time, R10=1/dc.T1(3.0, 'blood'))
    tissue.train(time, signal, aif, n0=10, tol=0.01)

    vmin = {'Fp':0, 've':0, 'Te':0}
    vmax = {'Fp':0.02, 've':0.2, 'Te':15}
    truth = {'S0': gt['S0'], 'Fp': gt['Fp'], 've': gt['ve']}
    truth['Te'] = np.zeros_like(truth['ve'], dtype=float)
    np.divide(truth['ve'], truth['Fp'], out=truth['Te'], where=truth['Fp'] != 0)
    tissue.plot_2d(time, signal, vmin=vmin, vmax=vmax, truth=truth)

def test_array_3d():
    n, nz = 64, 12
    time, signal, aif, gt = dc.fake_brain(n)

    # Compute R10
    R10 = np.zeros_like(gt['T1'], dtype=float)
    np.divide(1, gt['T1'], out=R10, where=gt['T1'] != 0)

    # Tile into 3D arrays
    R10_3d = R10[:, :, np.newaxis]
    R10 = np.tile(R10_3d, (1, 1, nz))
    signal_4d = signal[:, :, np.newaxis, :]
    signal = np.tile(signal_4d, (1, 1, nz, 1))

    tissue = dc.TissueLS(
        dt = 1.5,
        sequence = 'SS',
        agent = 'gadodiamide',
        field_strength = 3,
        TR = 0.005,
        FA = 15,
        R10 = R10,
    )

    aif = dc.Input(aif, time, R10=1/dc.T1(3.0, 'blood'))
    tissue.train(time, signal, aif, n0=10, tol=0.01)

    vmin = {'Fp':0, 've':0, 'Te':0}
    vmax = {'Fp':0.02, 've':0.2, 'Te':15}
    truth = {'S0': gt['S0'], 'Fp': gt['Fp'], 've': gt['ve']}
    truth['Te'] = np.zeros_like(truth['ve'], dtype=float)
    np.divide(truth['ve'], truth['Fp'], out=truth['Te'], where=truth['Fp'] != 0)

    tissue.plot_3d(time, signal, vmin=vmin, vmax=vmax)


if __name__ == "__main__":

    # Coverage tests
    test_coverage()
    test_io()
    test_exceptions()
    
    # Functional tests
    test_array_1d()
    test_array_2d()
    test_array_3d()
    
    print('All ui_tissue_ls tests passed!!')

