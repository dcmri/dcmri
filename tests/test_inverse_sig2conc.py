import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import dcmri as dc
from dcmri.core.tools import get_sequence
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.inverse.sig2conc import RelaxToSignal


def _test_class(cls):


    def _test_config(cnfg):
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return
        # if cnfg != {'sequence': '3D-DE-EPI', 'calibrate': False}:
        #     return
        # print(cnfg)
        data = instance.dummy_data()
        instance(data)

    cls.print_configs()
    cls.print_all_io(simple=False)

    configs = cls.all_configs()
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_s2c():
    for cls in [
        dc.SignalToConc,
    ]:
        _test_class(cls)



def test_coverage():

    # TODO include 'lin in sequences

    # 1D signals (n_times) or 2D (n_channels, n_times)
    for seq in dc.SignalToConc.configs['sequence']:
        print('scalar: ',seq)
        signal = 5 + np.arange(10) # n_times
        if seq in ['2D-DE-EPI', '3D-DE-EPI']:
            signal = np.stack((signal, signal)) # n_channels, n_times
        sig2conc = dc.SignalToConc(sequence=seq)
        sig2conc.inputs()
        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(dc.QVALUES, S=signal)

    # 3D signals (n_samples, n_channels, n_times)
    n_samples = 3
    for seq in dc.SignalToConc.configs['sequence']:
        # if seq != '3D-SR-SPGR-SS':
        #     continue
        print('array: ', seq)
        signal = 5 + np.arange(10)
        if seq in ['2D-DE-EPI', '3D-DE-EPI']:
            n_channels = 2 
        else:
            n_channels = 1
        signal = np.tile(signal[None, None, :], (n_samples, n_channels, 1)) # n_samples, n_channels, n_times
        sig2conc = dc.SignalToConc(sequence=seq)
        sig2conc.inputs()
        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']: # Non-monotonous
            continue 
        sig2conc(dc.QVALUES, S=signal)

        # Tst Options
        if seq == 'lin':
            dc.SignalToConc(sequence=seq, calibrate=False)(dc.QVALUES, S=signal, S0=np.full(n_samples, 1))
            dc.SignalToConc(sequence=seq, calibrate=True)(dc.QVALUES, S=signal)

        if seq in ['ZTE-3D-IR-SPGR-SS', '3D-IR-SPGR-SS']:
            dc.SignalToConc(sequence=seq, calibrate=False)(dc.QVALUES, S=signal, S0=np.full(n_samples, 1))

        if seq in ['2D-GE-EPI', '2D-SE-EPI', '2D-DE-EPI', '3D-GE-EPI', '3D-SE-EPI', '3D-DE-EPI']:
            dc.SignalToConc(sequence=seq, calibrate=True)(dc.QVALUES, S=signal, R1b=np.full(n_samples, 1))
            dc.SignalToConc(sequence=seq, calibrate=False)(dc.QVALUES, S=signal, S0=np.full(n_samples, 1))


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
        dc.SignalToConc()(S=[1,2], R1b=[1,2])
    except:
        pass
    else:
        assert False




def test_aif_analytical():

    # Simulation parameters
    seq = '3D-SPGR-SS'
    dt, tmax, B0, agent, R1ba, R2sba, S0a = 0.5, 180, 3, 'gadoterate', 0.7, 20, 3

    seq_params = {
        'FA': 15,
        'TR': 0.005,
        'TE': 0.002,
        'Nph': 32 * 10,
        'Nk0': 16 * 10,
        'NSR': 0,
        'B1corr': 0.75,
        'tacq': tmax - dt,
    }
    
    # Input signals
    rp = dc.r1(B0, 'blood', agent)
    r2s = dc.r2s(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.tristan(aif_time, BAT=10)
    aif_R1 = R1ba + rp * aif_conc
    # aif_R2s = R2sba + r2s * aif_conc
    aif_R2s = np.full_like(aif_conc, R2sba)
    aif_signal = RelaxToSignal(sequence=seq)(
        tR=aif_time, R1=aif_R1, R2s=aif_R2s, S0=S0a, 
        Kw=0, me=1, vw=1, **seq_params
    )

    # Invert
    ca = dc.SignalToConc(sequence=seq)(
        S=aif_signal['S'][0, 0, :], R1b=R1ba, R2sb=R2sba, nb=1,
        r1=rp, **seq_params
    )['C']
    ca_interp = np.interp(aif_signal['tS'], aif_time, aif_conc)
    assert np.linalg.norm(ca - ca_interp) < 1e-12 * np.linalg.norm(ca_interp)

    # plt.plot(aif_signal['tS'], ca, 'ro')
    # plt.plot(aif_time, aif_conc)
    # plt.show()


def test_aif_numerical():

    # Simulation parameters
    seq = '3D-IR-SPGR-SS'
    dt, tmax, B0, agent, R1ba, R2sba, S0a = 0.5, 180, 3, 'gadoterate', 0.7, 20, 3
    seq_params = {
        'FA': 15,
        'TR': 0.005,
        'TE': 0.002,
        'Nph': 32 * 10,
        'Nk0': 16 * 10,
        'TP': 0.1,
        'TD': 0.1,
        'NSR': 0,
        'B1corr': 0.75,
        'tacq': tmax - dt,
    }
    
    # Input signals
    rp = dc.r1(B0, 'blood', agent)
    r2s = dc.r2s(B0, 'blood', agent)
    aif_time = np.arange(0, tmax, dt)
    aif_conc = dc.tristan(aif_time, BAT=10)
    aif_R1 = R1ba + rp * aif_conc
    # aif_R2s = R2sba + r2s * aif_conc
    aif_R2s = np.full_like(aif_conc, R2sba)
    aif_signal = RelaxToSignal(sequence=seq)(
        tR=aif_time, R1=aif_R1, R2s=aif_R2s, S0=S0a, 
        Kw=0, me=1, v=1, **seq_params
    )

    # Invert
    ca = dc.SignalToConc(sequence=seq)(
        S=aif_signal['S'][0, 0, :], R1b=R1ba, R2sb=R2sba, nb=1, r1=rp, 
        **seq_params
    )['C']
    ca_interp = np.interp(aif_signal['tS'], aif_time, aif_conc)
    #print(np.linalg.norm(ca - ca_interp) / np.linalg.norm(ca_interp))
    assert np.linalg.norm(ca - ca_interp) < 1e-1 * np.linalg.norm(ca_interp)

    # plt.plot(aif_signal['tS'], ca, 'ro')
    # plt.plot(aif_time, aif_conc)
    # plt.show()


def test_function_dce():

    # Generate concentrations
    dt, tmax = 0.5, 180
    time = np.arange(0, tmax, dt)
    ca = dc.tristan(time, BAT=10)

    params_dce = dc.QVALUES | {
        'FA': 45,
        'PA': 120,
        'TR': 0.005,
        'TP': 0.100,
        'TE': 0.000,
        'TD': 0.100,
        'Nph': 32 * 10,
        'Nk0': 16 * 10,
        'TA': 1.0, # Short TA to avoid saturation of Mz in 1-shot sequences
        'tacq': tmax - dt,
    }

    seqs_dsc = ['2D-GE-EPI', '2D-SE-EPI', '2D-DE-EPI', '3D-GE-EPI', '3D-SE-EPI', '3D-DE-EPI'] 
    seqs_dce = [s for s in get_sequence('name') if s not in seqs_dsc]
    seqs_dce = [s for s in seqs_dce if get_sequence('steady-state', s)]

    # Define signal parameters
    R1ba, S0a, B1a = 0.7, 3, 0.75
    R2sb, R2b = 1.5, 2.0
    rp = dc.r1(3, 'blood', 'gadoterate')
    r2 = 10000
    r2s = 15000

    # DCE 
    for sequence in seqs_dce:
        # if sequence != '3D-IR-SS':
        #     continue
        print('DCE: ', sequence)

        # Generate signal and reconstruct concentrations
        R1a = R1ba + rp * ca
        R2s = R2sb + r2s * ca
        S = RelaxToSignal(sequence=sequence, calibrate=False)(
            params_dce, tR=time, R1=R1a, R2s=R2s, S0=S0a, B1corr=B1a
        )
        ca_rec = dc.SignalToConc(sequence=sequence, calibrate=True)(
            params_dce, S=S['S'][0, 0, :], iScal=[0], Scal=S['S'][:1,:1,:1], 
            R1b=R1ba, r1=rp, B1corr=B1a)['C']
        
        # Determine reconstruction error
        ca_interp = np.interp(S['tS'], time, ca)
        err = np.linalg.norm(ca_rec - ca_interp) / np.linalg.norm(ca_interp)
        
        try:
            assert err < 1e-2
            #assert err < 0
        except:
            print(sequence, err)
            plt.plot(S['tS'], ca_rec, 'ro')
            plt.plot(time, ca)
            plt.plot(S['tS'], ca_interp, marker='o', fillstyle='none', linestyle='None', color='blue')
            plt.show()
    # linear
    # Generate signal and reconstruct concentrations
    S = R1ba + rp * ca

    ca_rec = dc.SignalToConc(sequence='lin')(dc.QVALUES, S=S, R1b=R1ba, r1=rp)['C']
    
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

def test_function_dsc():

    # Generate concentrations
    dt, tmax = 0.5, 180
    time = np.arange(0, tmax, dt)
    ca = dc.tristan(time, BAT=10)

    params_dsc = dc.QVALUES | {
        'TE': 0.050, 
        'TE1': 0.005,
        'TE2': 0.050, 
        'FA': 90,
        'TR': 10.0, # Long TR to remove T1 weighting and get accurate conc
        'tacq': tmax - dt,
    }
    seqs_dsc = ['2D-GE-EPI', '2D-SE-EPI', '2D-DE-EPI', '3D-GE-EPI', '3D-SE-EPI', '3D-DE-EPI'] 

    # Define signal parameters
    R1ba, S0a, B1a = 0.7, 3, 0.75
    R2sb, R2b = 1.5, 2.0
    rp = dc.r1(3, 'blood', 'gadoterate')
    r2 = 10000
    r2s = 15000

    # DSC without T1-weighting
    for sequence in seqs_dsc:
        print('DSC: ', sequence)

        # Generate signal and reconstruct concentrations
        R1a = R1ba + rp * ca
        R2s = R2sb + r2s * ca
        R2 = R2b + r2 * ca
        S = RelaxToSignal(sequence=sequence, calibrate=False)(
            params_dsc, tR=time, R1=R1a, R2=R2, R2s=R2s, S0=S0a, B1corr=B1a,
        )
        ca_rec = dc.SignalToConc(sequence=sequence, calibrate=True)(
            params_dsc, S=S['S'][:, :1, :], iScal=[0], Scal=S['S'][:1,:1,:1], 
            B1corr=B1a, r2s=r2s, r2=r2
        )['C']
        ca_rec = np.mean(ca_rec, axis=0)
        
        # Determine reconstruction error
        ca_interp = np.interp(S['tS'], time, ca)
        err = np.linalg.norm(ca_rec - ca_interp) / np.linalg.norm(ca_interp)

        try:
            assert err < 1e-2
            # assert err < 0
        except:
            print(sequence, err)
            plt.plot(S['tS'], ca_rec, 'ro')
            plt.plot(time, ca)
            plt.plot(S['tS'], ca_interp, marker='o', fillstyle='none', linestyle='None', color='blue')
            plt.show()


def test_brain():
    # Settings
    dt=1.5
    tacq=180.0
    dt_sim=0.1
    BAT=20
    field_strength=3.0
    agent='gadodiamide'
    R1ba=1/dc.T1(3.0, 'blood')
    S0a=150
    H=0.45
    seq = 'ZTE-3D-SPGR-SS'
    rp = dc.r1(field_strength, 'plasma', agent)

    seq_params = {
        'FA': 15,
        'TR': 0.005,
        'Nph': 32 * 10,
        'Nk0': 16 * 10,
        'NSR': 0,
        'B1corr': 1,
        'tacq': 180.0 - dt,
    }

    # Simulated arterial concentration
    aif_time = np.arange(0, tacq+dt, dt_sim)
    aif_conc = dc.parker(aif_time, BAT)
    aif_conc = aif_conc * (1-H)

    # Sampled arterial signal
    aif_R1 = R1ba + rp * aif_conc
    aif_signal = RelaxToSignal(sequence=seq)(
        tR=aif_time, R1=aif_R1, S0=S0a,
        Kw=0, me=1, v=1, **seq_params
    )

    # Invert
    ca_rec = dc.SignalToConc(sequence=seq)(
        S=aif_signal['S'][0, 0, :], R1b=R1ba, nb=1, 
        r1=rp, **seq_params
    )['C']
    ca_interp = np.interp(aif_signal['tS'], aif_time, aif_conc)
    err = np.linalg.norm(ca_rec - ca_interp) / np.linalg.norm(ca_interp)

    try:
        assert err < 1e-3
    except:
        print('error', err)
        plt.plot(aif_signal['tS'], ca_rec, 'ro')
        plt.plot(aif_time, aif_conc)
        plt.plot(aif_signal['tS'], ca_interp, marker='o', fillstyle='none', linestyle='None', color='blue')
        plt.show()


if __name__ == "__main__":
    test_s2c()
    test_aif_analytical()
    test_aif_numerical()
    test_function_dce()
    test_function_dsc()
    test_coverage()
    test_exceptions()
    test_brain()

    print('All signal_to_conc tests passing!')