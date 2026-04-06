import numpy as np
from dcmri import Readout, Signal


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



if __name__ == "__main__":
    test_coverage_readout()
    test_coverage_signal()
    test_exceptions()
    print('All sig tests passing!')