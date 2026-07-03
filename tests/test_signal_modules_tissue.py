import itertools
import numpy as np

from dcmri import Signal, QVALUES


def test_coverage_signal():
    # scalar
    R1 = 1
    R2, R2s = R1, R1
    v = 0.3
    Fw = 0.01
    R1i = 0.75
    Fi = 0.008
    me = 2
    values = Signal.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(Signal.configs)}
        print('scalar', cnfgs)

        sig = Signal(**config)
        sig.inputs()

        q = sig.map_lexicon(QVALUES)
        sig(q, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)
        sig(q, R1=[R1], R2=R2, R2s=R2s, R1i=[R1i], Fi=[Fi], v=[v], Fw=[Fw], me=me)
        sig(q, R1=[R1], R2=R2, R2s=R2s, R1i=R1i, Fi=[Fi], v=[v], Fw=Fw, me=me)

    # nc
    R1 = [1,0.5]
    R2, R2s = R1, R1
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = [0.5, 0.75]
    Fi = [0.008, 0.004]
    me = 2

    values = Signal.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(Signal.configs)}
        # if cnfgs[0] != '3D-IR-SPGR-SS':
        #     continue
        print('nc', cnfgs)

        sig = Signal(**config)
        q = sig.map_lexicon(QVALUES)
        sig(q, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nt
    nt = 10
    R1 = np.full(nt, 1)
    R2, R2s = R1, R1
    v = 0.3
    Fw = 0.01
    R1i = np.full(nt, 0.06)
    Fi = 0.008
    me = 2

    values = Signal.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(Signal.configs)}
        print('nt', cnfgs)
        # if 'Eq-DE-EPI' != cnfgs[0]:
        #     continue

        sig = Signal(**config)
        sig.inputs()

        q = sig.map_lexicon(QVALUES)
        sig(q, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    R2, R2s = R1, R1
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    R1i = np.stack([np.full(nt, 0.6), np.full(nt, 0.8)])
    Fi = [0.008, 0.005]
    me = 2

    values = Signal.configs.values()
    for cnfgs in itertools.product(*values):
        config = {k: cnfgs[i] for i, k in enumerate(Signal.configs)}
        print('(nc, nt)', cnfgs)

        sig = Signal(**config)
        sig.inputs()


        q = sig.map_lexicon(QVALUES)
        sig(q, R1=R1, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    sig = Signal(sequence='Eq-DE-EPI')
    q = sig.map_lexicon(QVALUES)
    sig(q, R2=R2, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    sig = Signal(sequence='3D-SPGR-SS')
    q = sig.map_lexicon(QVALUES)
    sig(q, R1=R1, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)

    sig=Signal(sequence='Eq-GE-EPI')
    q = sig.map_lexicon(QVALUES)
    sig(q, R2s=R2s, R1i=R1i, Fi=Fi, v=v, Fw=Fw, me=me)


def test_exceptions_signal():
    pass


if __name__ == "__main__":
    test_coverage_signal()
    test_exceptions_signal()
    
    print('All magnetization tests passing!')