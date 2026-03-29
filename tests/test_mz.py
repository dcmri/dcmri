import numpy as np
import dcmri as dc

import dcmri.lexicon_utils as lexicon

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
    'FA': 75,
    'TR': 1.5,
}


def test_coverage():

    # scalar

    R1 = 1
    v = 0.3
    Fw = 0.01
    j = 0.06
    me = 2

    dc.Mz('SS', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('IR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('PR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SPGR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SSI', **params_ssi)(R1, v, Fw, j, me)
    dc.Mz('GE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('SE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('None', **params_dsc)(R1, v, Fw, j, me)

    # Variations
    dc.Mz('SS', **params_dce)(R1, None, Fw, j, me)

    # nc

    R1 = [1,0.5]
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = [0.06, 0.08]
    me = 2

    dc.Mz('SS', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('IR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('PR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SPGR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SSI', **params_ssi)(R1, v, Fw, j, me)
    dc.Mz('GE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('SE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('None', **params_dsc)(R1, v, Fw, j, me)

    # nt

    nt = 10
    R1 = np.full(nt, 1)
    v = 0.3
    Fw = 0.01
    j = np.full(nt, 0.06)
    me = 2

    dc.Mz('SS', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('IR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('PR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SPGR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SSI', **params_ssi)(R1, v, Fw, j, me)
    dc.Mz('GE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('SE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('None', **params_dsc)(R1, v, Fw, j, me)

    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = np.stack([np.full(nt, 0.06), np.full(nt, 0.08)])
    me = 2

    dc.Mz('SS', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('IR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('PR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SPGR', **params_dce)(R1, v, Fw, j, me)
    dc.Mz('SSI', **params_ssi)(R1, v, Fw, j, me)
    dc.Mz('GE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('SE-EPI', **params_dsc)(R1, v, Fw, j, me)
    dc.Mz('None', **params_dsc)(R1, v, Fw, j, me)

    # Special case
    Fw = 0.03
    dc.Mz('SS', **params_dce)(R1, v, Fw, j, me)

    # Functions
    Mz = dc.Mz('SS', **params_dce)
    assert 'TR' in Mz.params()

    # Variations
    Mz(R1, v, Fw, j, me, TR=0.01)
    Mz(R1, v, None, j, me, TR=0.01)
    Mz(R1, v, Fw, None, me, TR=0.01)

def test_exceptions():
    # nc, nt
    nt = 10
    R1 = np.stack([np.full(nt, 1), np.full(nt, 0.5)])
    v = [0.1, 0.4]
    Fw = [[0.01, 0.02], [0.03, 0.04]]
    j = np.stack([np.full(nt, 0.06), np.full(nt, 0.08)])
    me = 2

    try:
        dc.Mz('XX', **params_dce)(R1, v, Fw, j, me)
    except:
        pass
    else:
        assert False

    try:
        Fw3 = [[1,2,3], [4, 5, 6], [7, 8, 9]]
        dc.Mz('SS', **params_dce)(R1, v, Fw3, j, me)
    except:
        pass
    else:
        assert False

    try:
        dc.Mz('SS', **params_dce)(R1, None, Fw, j, me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.full(nt, 1)
        dc.Mz('SS', **params_dce)(R1_1, v, Fw, j, me)
    except:
        pass
    else:
        assert False

    try:
        R1_1 = np.stack([np.full(nt, 1), np.full(nt, 1), np.full(nt, 1)])
        dc.Mz('SS', **params_dce)(R1_1, v, Fw, j, me)
    except:
        pass
    else:
        assert False

    try:
        j_1 = np.full(nt, 0.06)
        dc.Mz('SS', **params_dce)(R1, v, Fw, j_1, me)
    except:
        pass
    else:
        assert False





if __name__ == "__main__":

    test_coverage()
    test_exceptions()

    print('All mz tests passing!')