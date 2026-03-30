import numpy as np
import dcmri as dc


def test_coverage():
    try:
        dc.Signal('SS', 'XX')
    except:
        pass
    else:
        assert False

    try:
        dc.Signal('XX', 'SS')
    except:
        pass
    else:
        assert False

    read = dc.Readout()
    read(np.ones(3), noise_sdev=1)
    dc.Readout().params()
    dc.Readout()(1)

    SEQS = ['SS', 'SR', 'IR-SS', 'PR-SS', 'PR', 'SSI', 'GE-EPI', 'SE-EPI', 'None']
    for seq in SEQS:
        for iseq in SEQS:
            print(seq, iseq)
            signal = dc.Signal(seq, iseq)
            signal.params()
            signal(R1=1)
            signal(R1=1, R1i=1, Fi=1)
            signal(R1=np.ones((2,3)), v=[0.1, 0.9], Fw=[[1,0],[0,0]], Fi=[1,0], R1i=np.ones((2,3)))

    # Alternative options
    dc.Signal().params()
    dc.Signal()(TR=0.002)
    dc.Signal()()

if __name__ == "__main__":
    test_coverage()
    print('All sig tests passing!')