import numpy as np

from dcmri.bloch import Readout, Signal, Longitudinal
from dcmri.lexicon import SEQUENCES

params_dce = {
    'FA': 45,
    'PA': 120,
    'TR': 0.005,
    'TC': 0.250, # 250ms
    'TP': 0.100,
    'TA': 0.400,
}
params_ssi = {
    'FA': 45,
    'SA': 120,
    'TR': 0.005,
    'TF': 0.250, 
}
params_dsc = {
    'TE': 0.050, 
    'TE2': 0.050, 
    'FA': 75,
    'TR': 1.5,
}

seqs_dce = [s for s, v in SEQUENCES.items() if v['type']=='DCE' and s!='3D-SPGR-SSI']
seqs_ssi = ['3D-SPGR-SSI']
seqs_dsc = [s for s, v in SEQUENCES.items() if v['type']=='DSC'] + ['Eq']


def test_coverage_readout():
    # Cover configs
    for seq in Readout.configs['sequence']:
        print(seq)
        read = Readout(seq)
        read(Mz=1, R2=1, R2s=1)

    # Cover functionality
    read = Readout()
    read(Mz=np.ones(3), noise_sdev=1, TE=0)
    Readout().params()
    Readout()(Mz=1, TE=0)


def test_coverage_signal():

    for seq in Signal.configs['sequence']:
        print(seq)
        signal = Signal(seq)
        signal.params()
        signal(R1=1, R2=1, R2s=1)
        signal(R1=1, R2=1, R2s=1, R1i=1, Fi=1)
        signal(R1=np.ones((2,3)), R2=np.ones((2,3)), R2s=np.ones((2,3)), v=[0.1, 0.9], Fw=[[1,0],[0,0]], Fi=[1,0], R1i=np.ones((2,3)))

    # Alternative options
    Signal().params()
    Signal('3D-SPGR-SS')(R1=1, TR=0.002, TE=0)

    # No T1-weighting
    Signal('SE-EPI')(R2=1)
    Signal('GE-EPI')(R2s=1)
    Signal('DE-EPI')(R2=1, R2s=1)

def test_exceptions():
    try:
        Signal('XX')
    except:
        pass
    else:
        assert False
    try:
        Signal('3D-SPGR-SS')()
    except:
        pass
    else:
        assert False
    try:
        Signal('GE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Signal('SE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Signal('DE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Signal('DE-EPI')(R2=[1], R2s=[1,2])
    except:
        pass
    else:
        assert False
    try:
        Signal('3D-SPGR-SS')(R1=1, R1i=[1,2])
    except:
        pass
    else:
        assert False
    try:
        Signal('3D-SPGR-SS')(R1=1, R1i=1)
    except:
        pass
    else:
        assert False
    try:
        Signal('3D-SPGR-SS')(R1=1, R1i=1, Fi=[1,2])
    except:
        pass
    else:
        assert False

    try:
        Readout('XX')
    except:
        pass
    else:
        assert False
    try:
        Readout('GE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Readout('SE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Readout('DE-EPI')()
    except:
        pass
    else:
        assert False
    try:
        Readout('DE-EPI')(R2=1, R2s=[1,2])
    except:
        pass
    else:
        assert False
    try:
        Readout('3D-SPGR-SS')()
    except:
        pass
    else:
        assert False



def test_coverage_longitudinal():

    # scalar

    R1 = 1
    v = 0.3
    Fw = 0.01
    R1i = 0.75
    Fi = 0.008
    me = 2

    for seq in seqs_dce:
        Longitudinal(seq, **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        Longitudinal(seq, **params_ssi)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        Longitudinal(seq, **params_dsc)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # Variations
    Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=None, Fw=Fw, me=me)

    # nc

    R1 = [1,0.5]
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = [0.5, 0.75]
    Fi = [0.008, 0.004]
    me = 2

    for seq in seqs_dce:
        Longitudinal(seq, **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        Longitudinal(seq, **params_ssi)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        Longitudinal(seq, **params_dsc)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nt

    nt = 10
    R1 = np.full(nt, 1)
    v = 0.3
    Fw = 0.01
    R1i = np.full(nt, 0.06)
    Fi = 0.008
    me = 2

    for seq in seqs_dce:
        Longitudinal(seq, **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        Longitudinal(seq, **params_ssi)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        Longitudinal(seq, **params_dsc)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    for seq in seqs_dce:
        Longitudinal(seq, **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        Longitudinal(seq, **params_ssi)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        Longitudinal(seq, **params_dsc)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # Special case
    Fw = 0.03
    Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # Functions
    assert '3D-SPGR-SS' in Longitudinal.configs['sequence']
    Mz = Longitudinal('3D-SPGR-SS', **params_dce)
    assert 'TR' in Mz.params()

    # Variations
    Mz(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me, TR=0.01)
    Mz(R1=R1, R1i=None, Fi=Fi, v=v, Fw=Fw, me=me, TR=0.01)

def test_exceptions_longitudinal():
    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    try:
        Longitudinal('XX', **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        Longitudinal('SPGR-SS', **params_dce)()
    except:
        pass
    else:
        assert False

    try:
        Fw3 = [[1,2,3], [4, 5, 6], [7, 8, 9]]
        Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=v, Fw=Fw3, me=me)
    except:
        pass
    else:
        assert False

    try:
        Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1, R1i=R1i, Fi=Fi, v=None, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.full(nt, 1)
        Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1_1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.stack([np.full(nt, 1), np.full(nt, 1), np.full(nt, 1)])
        Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1_1, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        R1i_1 = np.full(nt, 0.6)
        Longitudinal('3D-SPGR-SS', **params_dce)(R1=R1, R1i=R1i_1, Fi=Fi, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False




if __name__ == "__main__":
    test_coverage_readout()
    test_coverage_signal()
    test_exceptions()
    test_coverage_longitudinal()
    test_exceptions_longitudinal()
    print('All sig tests passing!')