import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
from dcmri.magnetization import Signal
from dcmri.core import Input
from dcmri import TissueLS as Model
#from dcmri.fake import fake_brain


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

    values = Model.configs.values()
    for cnfgs in itertools.product(*values):
        # if cnfgs[0] != 'DE-EPI':
        #     continue
        model = Model(*cnfgs)
        time = model.time()
        signal = model.predict(time)
        model.train(time, signal, tol=0.01)
        model.plot(time, signal, round_to=3, show=DEBUG)
        cost = model.cost(time, signal)
        assert cost < 10

    # Test Forward API outputs
    model = Model()
    t = model.time()
    C = model.conc()
    S = model.signal()

    assert C.ndim in [1,2]
    assert len(S) == len(t)

    # Coverage config options
    Model(irf=np.ones((128, 480)))


def test_io():

    # 1D plot
    model = Model()
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
    model = Model(shape=(10, 10))
    t, S = model.time(), model.signal()
    pars = model.params()
    c = model.conc()
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
    model = Model(shape=(10, 10, 10))
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
        Model(shape=(1,2,3,4))
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model(irf=np.ones(300), c_a=np.ones(200))
        model.predict(np.arange(30))
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model()
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model((10, ))
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model((10, ))
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model()
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model((10, 10, 10))
        model.plot_2d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        model = Model((10, 10))
        model.plot_3d(model.time(), model.signal())
    except ValueError:
        pass 
    else:
        assert False

    try:
        Model(sequence='X')
    except ValueError:
        pass 
    else:
        assert False


def test_array_1d():

    # Generate an AIF
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R10a, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 3, 0.75
    FA, TR = 15, 0.005

    # Input signals
    rp = dc.const.r1(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.aif.tristan(aif_time, BAT=10)
    aif_R1 = R10a + rp * aif_conc
    aif_signal = Signal(seq)(R1=aif_R1, S0=S0a, FA=FA, TR=TR, TE=0, B1corr=B1a)
    aif_ = Input(aif_signal, aif_time, R10=R10a, B1corr=B1a)

    # Tissue
    params = {
        'dt': dt, 
        'c_a': aif_conc,
        'field_strength': B0,
        'agent': agent,
        'FA': FA, 
        'TR': TR,
        'TE': 0,
        'S0': 5,
    }
    model = Model(sequence=seq, **params)
    time = model.time()
    signal = model.predict(time)

    # Fit with generated AIF signal
    model.train(time, signal, aif_)
    model.plot(time, signal, round_to=3, show=DEBUG)
    cost = model.cost(time, signal)
    print(seq, cost)
    print(model.parameters(iv=True))
    assert cost < 5

    # Test some training options
    model = Model(dt=dt, c_a=aif_conc)
    time = model.time()
    signal = model.predict(time)
    model.train(time, signal, n0=10)

def test_array_1d_brain():
    # Generated with a 2CX model
    npix = 64
    time, signal, aif, gt = dc.fake.brain(n=npix)

    # ca = SignalToConc('3D-SPGR-SS')(aif, S0=None, R10=1/dc.const.T1(3.0, 'blood'), n0=1, B1corr=1, TR=0.005, FA=15, TE=0, r1=const.r1(3, 'plasma', 'gadodiamide'))
    # plt.plot(time, ca, 'bo')
    # plt.plot(gt['t'], gt['cb'], 'r-')
    # plt.show()
    # return

    # Compute R10
    R10 = np.zeros_like(gt['T1'], dtype=float)
    np.divide(1, gt['T1'], out=R10, where=gt['T1'] != 0)

    x = npix // 2
    R10 = R10[x, x]
    signal = signal[x, x, :]

    # Match paramaters to the fake brain simulation
    params = {
        'dt': 1.5,
        'sequence': '3D-SPGR-SS',
        'agent': 'gadodiamide',
        'field_strength': 3,
        'TR': 0.005,
        'FA': 15,
        'R10': R10,
        'TE': 0,
    }

    model = Model(**params)

    aif = Input(aif, time, R10=1/dc.const.T1(3.0, 'blood'))

    # Fit with generated AIF signal
    model.train(time, signal, aif)
    model.plot(time, signal, round_to=3, show=DEBUG)
    cost = model.cost(time, signal)
    print(cost)
    print(model.parameters(iv=True))
    print(gt['Fp'][x, x], model.parameters()['Fp'])
    assert cost < 5


def test_array_2d():
    # Generated with a 2CX model
    npix = 64
    time, signal, aif, gt = dc.fake.brain(n=npix)

    # Compute R10
    R10 = np.zeros_like(gt['T1'], dtype=float)
    np.divide(1, gt['T1'], out=R10, where=gt['T1'] != 0)

    # Match paramaters to the fake_brain simulation
    tissue = Model(
        dt = 1.5,
        sequence = '3D-SPGR-SS',
        agent = 'gadodiamide',
        field_strength = 3,
        TR = 0.005,
        FA = 15,
        R10 = R10,
    )

    aif = Input(aif, time, R10=1/dc.const.T1(3.0, 'blood'))
    tissue.train(time, signal, aif, n0=10, tol=0.01)

    vmin = {'Fp':0, 've':0, 'Te':0}
    vmax = {'Fp':0.02, 've':0.2, 'Te':15}
    truth = {'S0': gt['S0'], 'Fp': gt['Fp'], 've': gt['ve']}
    truth['Te'] = np.zeros_like(truth['ve'], dtype=float)
    np.divide(truth['ve'], truth['Fp'], out=truth['Te'], where=truth['Fp'] != 0)
    tissue.plot_2d(time, signal, vmin=vmin, vmax=vmax, truth=truth, show=DEBUG)

def test_array_3d():
    n, nz = 64, 12
    time, signal, aif, gt = dc.fake.brain(n)

    # Compute R10
    R10 = np.zeros_like(gt['T1'], dtype=float)
    np.divide(1, gt['T1'], out=R10, where=gt['T1'] != 0)

    # Tile into 3D arrays
    R10_3d = R10[:, :, np.newaxis]
    R10 = np.tile(R10_3d, (1, 1, nz))
    signal_4d = signal[:, :, np.newaxis, :]
    signal = np.tile(signal_4d, (1, 1, nz, 1))

    tissue = Model(
        dt = 1.5,
        sequence = '3D-SPGR-SS',
        agent = 'gadodiamide',
        field_strength = 3,
        TR = 0.005,
        FA = 15,
        R10 = R10,
    )

    aif = Input(aif, time, R10=1/dc.const.T1(3.0, 'blood'))
    tissue.train(time, signal, aif, n0=10, tol=0.01)

    vmin = {'Fp':0, 've':0, 'Te':0}
    vmax = {'Fp':0.02, 've':0.2, 'Te':15}
    truth = {'S0': gt['S0'], 'Fp': gt['Fp'], 've': gt['ve']}
    truth['Te'] = np.zeros_like(truth['ve'], dtype=float)
    np.divide(truth['ve'], truth['Fp'], out=truth['Te'], where=truth['Fp'] != 0)

    tissue.plot_3d(time, signal, vmin=vmin, vmax=vmax, show=DEBUG)


if __name__ == "__main__":

    # Coverage tests
    test_coverage()
    test_io()
    test_exceptions()
    
    # Functional tests
    test_array_1d()
    test_array_1d_brain()
    test_array_2d()
    test_array_3d()
    
    print('All tissue_ls tests passed!!')

