import numpy as np
import dcmri as dc

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

seqs_dce = [s for s, v in dc.MZ_PREP.items() if v['type']=='DCE' and s!='SSI']
seqs_ssi = ['SSI']
seqs_dsc = [s for s, v in dc.MZ_PREP.items() if v['type']=='DSC'] + ['Eq']

def test_coverage():

    # scalar

    R1 = 1
    v = 0.3
    Fw = 0.01
    j = 0.06
    me = 2

    for seq in seqs_dce:
        dc.Mz(seq, **params_dce)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        dc.Mz(seq, **params_ssi)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        dc.Mz(seq, **params_dsc)(R1, j, v=v, Fw=Fw, me=me)

    # Variations
    dc.Mz('SPGR-SS', **params_dce)(R1, j, v=None, Fw=Fw, me=me)

    # nc

    R1 = [1,0.5]
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = [0.06, 0.08]
    me = 2

    for seq in seqs_dce:
        dc.Mz(seq, **params_dce)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        dc.Mz(seq, **params_ssi)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        dc.Mz(seq, **params_dsc)(R1, j, v=v, Fw=Fw, me=me)

    # nt

    nt = 10
    R1 = np.full(nt, 1)
    v = 0.3
    Fw = 0.01
    j = np.full(nt, 0.06)
    me = 2

    for seq in seqs_dce:
        dc.Mz(seq, **params_dce)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        dc.Mz(seq, **params_ssi)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        dc.Mz(seq, **params_dsc)(R1, j, v=v, Fw=Fw, me=me)

    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = np.stack([np.full(nt, 0.06), np.full(nt, 0.08)])
    me = 2

    for seq in seqs_dce:
        dc.Mz(seq, **params_dce)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_ssi:
        dc.Mz(seq, **params_ssi)(R1, j, v=v, Fw=Fw, me=me)
    for seq in seqs_dsc:
        dc.Mz(seq, **params_dsc)(R1, j, v=v, Fw=Fw, me=me)

    # Special case
    Fw = 0.03
    dc.Mz('SPGR-SS', **params_dce)(R1, j, v=v, Fw=Fw, me=me)

    # Functions
    assert 'SPGR-SS' in dc.Mz.configs['sequence']
    Mz = dc.Mz('SPGR-SS', **params_dce)
    assert 'TR' in Mz.params()

    # Variations
    Mz(R1, j, v=v, Fw=Fw, me=me, TR=0.01)
    Mz(R1, None, v=v, Fw=Fw, me=me, TR=0.01)

def test_exceptions():
    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = np.stack([np.full(nt, 0.06), np.full(nt, 0.08)])
    me = 2

    try:
        dc.Mz('XX', **params_dce)(R1, j, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        dc.Mz('SPGR-SS', **params_dce)()
    except:
        pass
    else:
        assert False

    try:
        Fw3 = [[1,2,3], [4, 5, 6], [7, 8, 9]]
        dc.Mz('SPGR-SS', **params_dce)(R1, j, v=v, Fw=Fw3, me=me)
    except:
        pass
    else:
        assert False

    try:
        dc.Mz('SPGR-SS', **params_dce)(R1, j, v=None, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.full(nt, 1)
        dc.Mz('SPGR-SS', **params_dce)(R1_1, j, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.stack([np.full(nt, 1), np.full(nt, 1), np.full(nt, 1)])
        dc.Mz('SPGR-SS', **params_dce)(R1_1, j, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False

    try:
        j_1 = np.full(nt, 0.06)
        dc.Mz('SPGR-SS', **params_dce)(R1, j_1, v=v, Fw=Fw, me=me)
    except:
        pass
    else:
        assert False


if __name__ == "__main__":
    test_coverage()
    test_exceptions()

    print('All mz tests passing!')