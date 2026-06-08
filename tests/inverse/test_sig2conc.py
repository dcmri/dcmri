import numpy as np
import matplotlib.pyplot as plt

import dcmri as dc
from dcmri.lexicon.dicts import SEQUENCES
from dcmri.core.types import Input



def test_coverage():

    # TODO include 'lin in sequences

    # 1D signals (n_times) or 2D (n_channels, n_times)
    for seq in dc.SignalToConc.configs['sequence']:
        print('scalar: ',seq)
        signal = 5 + np.arange(10) # n_times
        if seq in ['Eq-DE-EPI', 'DE-EPI']:
            signal = np.stack((signal, signal)) # n_channels, n_times
        sig2conc = dc.SignalToConc(seq)
        sig2conc.params()
        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(signal)

    # 3D signals (n_samples, n_channels, n_times)
    n_samples = 3
    for seq in dc.SignalToConc.configs['sequence']:
        print('array: ', seq)
        signal = 5 + np.arange(10)
        if seq in ['Eq-DE-EPI', 'DE-EPI']:
            n_channels = 2 
        else:
            n_channels = 1
        signal = np.tile(signal[None, None, :], (n_samples, n_channels, 1)) # n_samples, n_channels, n_times
        sig2conc = dc.SignalToConc(seq)
        sig2conc.params()
        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(signal)

        # Options
        if seq == 'lin':
            sig2conc(signal, S0=np.full(n_samples, 1))
            sig2conc(signal, R10=None, S0=1)
        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']:
            sig2conc(signal, R10=None, S0=np.full(n_samples, 1))

        if seq in ['Eq-GE-EPI', 'Eq-SE-EPI', 'Eq-DE-EPI', 'GE-EPI', 'SE-EPI', 'DE-EPI']:
            sig2conc(signal, R10=np.full(n_samples, 1))
            sig2conc(signal, S0=np.full(n_samples, 1))


def test_exceptions():

    try:
        dc.SignalToConc('3D-SPGR')
    except:
        pass
    else:
        assert False
    try:
        dc.SignalToConc()(S=[1])
    except:
        pass
    else:
        assert False
    try:
        dc.SignalToConc()(S=[1,2], R10=[1,2])
    except:
        pass
    else:
        assert False
    try:
        dc.SignalToConc()(S=[1,2], S0=[1,2])
    except:
        pass
    else:
        assert False

def test_function():

    # Generate concentrations
    dt, tmax = 0.5, 180
    time = np.arange(0, tmax, dt)
    ca = dc.tristan(time, BAT=10)

    params_dce = {
        'FA': 45,
        'PA': 120,
        'TR': 0.005,
        'TC': 0.250, # 250ms
        'TP': 0.100,
        'TA': 0.400,
        'TE': 0.000,
    }
    params_dsc = {
        'TE': 0.050, 
        'TE1': 0.005,
        'TE2': 0.050, 
        'FA': 90,
        'TR': np.inf,
    }

    seqs_dsc = ['GE-EPI', 'SE-EPI', 'DE-EPI', 'Eq-GE-EPI', 'Eq-SE-EPI', 'Eq-DE-EPI'] 
    seqs_dce = [s for s in SEQUENCES.keys() if s not in seqs_dsc]
    seqs_dce = [s for s in seqs_dce if SEQUENCES[s]['steady-state']]

    # Define signal parameters
    R10a, S0a, B1a = 0.7, 3, 0.75
    R20s, R20 = 1.5, 2.0
    rp = dc.r1(3, 'blood', 'gadoterate')
    r2 = 10000
    r2s = 15000

    # DCE 
    for sequence in seqs_dce:
        # Generate signal and reconstruct concentrations
        R1a = R10a + rp * ca
        R2s = R20s + r2s * ca
        S = dc.Signal(sequence, **params_dce)(
            R1=R1a, R2s=R2s, S0=S0a, B1corr=B1a, TE=0,
        )
        ca_rec = dc.SignalToConc(sequence, **params_dce)(
            S, R10=R10a, r1=rp, B1corr=B1a, 
        )
        
        # Determine reconstruction error
        err = np.linalg.norm(ca-ca_rec) / np.linalg.norm(ca)
        assert err < 1e-6
        # try:
        #     assert err < 1e-2
        # except:
        #     print(sequence, err)
        #     plt.plot(time, ca, 'r-')
        #     plt.plot(time, ca_rec, 'bx')
        #     plt.show()


    # DSC without T1-weighting
    for sequence in seqs_dsc:

        # Generate signal and reconstruct concentrations
        R1a = R10a + rp * ca
        R2s = R20s + r2s * ca
        R2 = R20 + r2 * ca
        S = dc.Signal(sequence, **params_dsc)(
            R1=R1a, R2=R2, R2s=R2s, S0=S0a, B1corr=B1a, TR=np.inf,
        )

        ca_rec = dc.SignalToConc(sequence, **params_dsc)(
            S, B1corr=B1a, r2s=r2s, r2=r2
        )
        if sequence in ['DE-EPI', 'Eq-DE-EPI']:
            ca_rec = np.mean(ca_rec, axis=0)
        
        # Determine reconstruction error
        err = np.linalg.norm(ca-ca_rec) / np.linalg.norm(ca)
        assert err < 1e-9

        # print(sequence, err)
        # plt.plot(time, ca, 'r-')
        # plt.plot(time, ca_rec, 'bx')
        # plt.show()

    # linear
    # Generate signal and reconstruct concentrations
    S = R10a + rp * ca

    ca_rec = dc.SignalToConc('lin')(
        S, R10=R10a, r1=rp, 
    )
    
    # Determine reconstruction error
    err = np.linalg.norm(ca-ca_rec) / np.linalg.norm(ca)
    assert err < 1e-2
    # try:
    #     assert err < 1e-2
    # except:
    #     print(sequence, err)
    #     plt.plot(time, ca, 'r-')
    #     plt.plot(time, ca_rec, 'bx')
    #     plt.show()

def test_brain():
    # Settings
    dt=1.5
    tacq=180.0
    dt_sim=0.1
    BAT=20
    field_strength=3.0
    agent='gadodiamide'
    R10a=1/dc.T1(3.0, 'blood')
    S0=150
    TR=0.005
    FA=15
    H=0.45
    seq = 'ZTE-3D-SPGR-SS'
    rp = dc.r1(field_strength, 'plasma', agent)

    # Simulated arterial concentration
    tsim = np.arange(0, tacq+dt, dt_sim)
    cp = dc.parker(tsim, BAT)
    cb = cp * (1-H)

    # Sampled arterial signal
    R1b = R10a + rp * cb
    aif_ = dc.Signal(seq)(S0=S0, R1=R1b, TR=TR, FA=FA)
    time = np.arange(0, tacq, dt)
    aif_acq = dc.sample(time, tsim, aif_, dt)

    # Invert
    cb_rec = dc.SignalToConc(seq)(aif_acq, R10=R10a, n0=1, TR=TR, FA=FA, r1=rp)
    # plt.plot(time, cb_rec, 'bo')
    # plt.plot(tsim, cb, 'r-')
    # plt.show()

    cb_acq = dc.sample(time, tsim, cb, dt)
    err = np.linalg.norm(cb_acq-cb_rec) / np.linalg.norm(cb)
    assert err < 1e-2


def test_aif():

    # Simulation parameters
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R10a, R20sa, S0a, B1a = 0.5, 180, 3, 'gadoterate', 0.7, 20, 3, 0.75
    FA, TR, TE = 15, 0.005, 0.002 # Defaults
    
    # Input signals
    rp = dc.r1(B0, 'blood', agent)
    r2s = dc.r2s(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.tristan(aif_time, BAT=10)
    aif_R1 = R10a + rp * aif_conc
    # aif_R2s = R20sa + r2s * aif_conc
    aif_R2s = np.full_like(aif_conc, R20sa)
    aif_signal = dc.Signal(seq)(R1=aif_R1, R2s=aif_R2s, S0=S0a, FA=FA, TR=TR, TE=TE, B1corr=B1a)

    # Invert
    p = {
        'FA': FA, 
        'TE': TE,
        'TR': TR,
        'r2s': r2s,    
    }
    ca = dc.SignalToConc(seq, **p)(
        aif_signal, R10=R10a, n0=1, S0=None,
        B1corr=B1a, r1=rp,
    )
    assert np.linalg.norm(ca - aif_conc) < 1e-9

    # plt.plot(aif_time, aif_conc)
    # plt.plot(aif_time, ca)
    # plt.show()



if __name__ == "__main__":

    test_aif()
    test_coverage()
    test_exceptions()
    test_function()
    test_brain()
    print('All signal_to_conc tests passing!')