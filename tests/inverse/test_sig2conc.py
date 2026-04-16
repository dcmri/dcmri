import numpy as np
import matplotlib.pyplot as plt

from dcmri.inverse import SignalToConc  
from dcmri.lexicon import SEQUENCES
from dcmri import aif, const
from dcmri.bloch import Signal
from dcmri.utils.misc import sample



def test_coverage():

    # TODO include 'lin in sequences

    # 1D signals (n_times)
    for seq in list(SEQUENCES.keys()) + ['lin']:
        if seq != 'lin':
            if not SEQUENCES[seq]['steady-state']: # Not invertible
                continue 
        print('scalar: ',seq)
        signal = 5 + np.arange(10)
        if seq == 'DE-EPI':
            signal = np.stack((signal, signal)) # n_channels, n_times
        sig2conc = SignalToConc(seq)
        sig2conc.params()
        if seq in ['3D-IR-SPGR-SS', '2D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(signal)

    # 2D signals (n_samples, n_times)
    n_samples = 3
    for seq in list(SEQUENCES.keys()) + ['lin']:
        if seq != 'lin':
            if not SEQUENCES[seq]['steady-state']: # Not invertible
                continue
        print('array: ', seq)
        signal = 5 + np.arange(10)
        if seq == 'DE-EPI':
            signal = np.tile(signal[None, None, :], (n_samples, 2, 1)) # n_samples, n_signals, n_times
        else:
            signal = np.tile(signal, (n_samples,1)) # n_samples, n_times
        sig2conc = SignalToConc(seq)
        sig2conc.params()
        if seq in ['3D-IR-SPGR-SS', '2D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(signal)

        # Options
        if seq == 'lin':
            sig2conc(signal, S0=np.full(n_samples, 1))
            sig2conc(signal, R10=None, S0=1)
        if seq == '3D-SPGR-SS':
            sig2conc(signal, R10=None, S0=np.full(n_samples, 1))
        if seq == '3D-SR-SPGR':
            sig2conc(signal, R10=None, S0=np.full(n_samples, 1))

        if seq in ['GE-EPI', 'SE-EPI', 'DE-EPI']:
            sig2conc(signal, R10=np.full(n_samples, 1))
            sig2conc(signal, S0=np.full(n_samples, 1))


def test_exceptions():
    try:
        SignalToConc('XX')
    except:
        pass
    else:
        assert False
    try:
        SignalToConc('3D-IR-SPGR')(S=[1,2])
    except:
        pass
    else:
        assert False
    try:
        SignalToConc()(1)
    except:
        pass
    else:
        assert False
    try:
        SignalToConc()(S=[1,2], R10=[1,2])
    except:
        pass
    else:
        assert False
    try:
        SignalToConc()(S=[1,2], S0=[1,2])
    except:
        pass
    else:
        assert False

    # try: # non-monotonous signal
    #     S = np.ones(3)
    #     dc.SignalToConc('3D-PR-SPGR', PA=179, TE=0)(S, R10=0.7, r1=3500)
    # except:
    #     pass
    # else:
    #     assert False

def test_function():

    # Generate concentrations
    dt, tmax = 0.5, 180
    time = np.arange(0, tmax, dt)
    ca = aif.tristan(time, BAT=10)

    params_dce = {
        'FA': 45,
        'PA': 120,
        'TR': 0.005,
        'TC': 0.250, # 250ms
        'TP': 0.100,
        'TA': 0.400,
        'TE': 0.002,
    }
    params_ssi = {
        'FA': 45,
        'SA': 120,
        'TR': 0.005,
        'TF': 0.250, 
        'TE': 0,
    }
    params_dsc = {
        'TE': 0.050, 
        'TE2': 0.050, 
        'FA': 90,
        'TR': 1.5,
    }

    seqs_dce = [s for s, v in SEQUENCES.items() if v['type']=='DCE' and s!='Eq' and s!='SSI' and v['steady-state']]
    seqs_ssi = ['3D-SPGR-SSI']
    seqs_dsc = [s for s, v in SEQUENCES.items() if v['type']=='DSC' and v['steady-state']]

    # Define signal parameters
    R10a, S0a, B1a = 0.7, 3, 0.75
    R20s, R20 = 1.5, 2.0
    rp = const.r1(3, 'blood', 'gadoterate')
    r2 = 10000
    r2s = 15000

    # DCE without T2-weighting
    for sequence in seqs_dce:
        # Generate signal and reconstruct concentrations
        R1a = R10a + rp * ca
        S = Signal(sequence, **params_dce)(
            R1=R1a, S0=S0a, B1corr=B1a, TE=0,
        )
        ca_rec = SignalToConc(sequence, **params_dce)(
            S, R10=R10a, r1=rp, B1corr=B1a, TE=0, 
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

    # DCE with constant T2-weighting
    for sequence in seqs_dce:
        # Generate signal and reconstruct concentrations
        R1a = R10a + rp * ca
        R2s = R20s + 0 * ca
        S = Signal(sequence, **params_dce)(
            R1=R1a, S0=S0a, B1corr=B1a, R2s=R2s
        )
        ca_rec = SignalToConc(sequence, **params_dce)(
            S, R10=R10a, r1=rp, B1corr=B1a, R20s=R20s,
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


    # DSC without T1-weighting
    for sequence in seqs_dsc:

        # Generate signal and reconstruct concentrations
        R1a = R10a 
        R2s = R20s + r2s * ca
        R2 = R20 + r2 * ca
        S = Signal(sequence, **params_dsc)(
            R1=R1a, R2=R2, R2s=R2s, S0=S0a, B1corr=B1a,
        )
        ca_rec = SignalToConc(sequence, **params_dsc)(
            S, B1corr=B1a, r2s=r2s, r2=r2
        )
        if sequence == 'DE-EPI':
            ca_rec = np.mean(ca_rec, axis=0)
        
        # Determine reconstruction error
        err = np.linalg.norm(ca-ca_rec) / np.linalg.norm(ca)
        assert err < 1e-2

        # print(sequence, err)
        # plt.plot(time, ca, 'r-')
        # plt.plot(time, ca_rec, 'bx')
        # plt.show()

    # SSI without T2-weighting
    for sequence in seqs_ssi:
        # Generate signal and reconstruct concentrations
        R1a = R10a + rp * ca
        S = Signal(sequence, **params_ssi)(
            R1=R1a, S0=S0a, B1corr=B1a, TE=0,
        )
        ca_rec = SignalToConc(sequence, **params_ssi)(
            S, R10=R10a, r1=rp, B1corr=B1a, TE=0, 
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

    # linear
    # Generate signal and reconstruct concentrations
    S = R10a + rp * ca

    ca_rec = SignalToConc('lin')(
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
    R10a=1/const.T1(3.0, 'blood')
    S0=150
    TR=0.005
    FA=15
    H=0.45
    TE=0
    seq = '3D-SPGR-SS'
    rp = const.r1(field_strength, 'plasma', agent)

    # Simulated arterial concentration
    tsim = np.arange(0, tacq+dt, dt_sim)
    cp = aif.parker(tsim, BAT)
    cb = cp * (1-H)

    # Sampled arterial signal
    R1b = R10a + rp * cb
    aif_ = Signal(seq)(S0=S0, R1=R1b, TR=TR, FA=FA, TE=TE)
    time = np.arange(0, tacq, dt)
    aif_acq = sample(time, tsim, aif_, dt)

    # Invert
    cb_rec = SignalToConc(seq)(aif_acq, R10=R10a, n0=1, TR=TR, FA=FA, TE=TE, r1=rp)
    # plt.plot(time, cb_rec, 'bo')
    # plt.plot(tsim, cb, 'r-')
    # plt.show()

    cb_acq = sample(time, tsim, cb, dt)
    err = np.linalg.norm(cb_acq-cb_rec) / np.linalg.norm(cb)
    assert err < 1e-2

if __name__ == "__main__":

    test_coverage()
    test_exceptions()
    test_function()
    test_brain()
    print('All signal_to_conc tests passing!')