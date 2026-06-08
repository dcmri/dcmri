from dcmri.core.types import Input

def test_input():

    aif = Input({'signal': [1,2,3], 'time': [0,1,2], 'R10': 1, 'B1corr': 1})
    assert aif.signal[-1] == 3

    aif = Input({'signal': [1,2,3], 'time': [0,1,2]})
    assert aif.signal[-1] == 3

    aif = Input({'signal': [1,2,3], 'dt': 2.0})
    assert aif.time[-1] == 4

    try:
        aif = Input({'signal': [1,2,3]})
    except:
        pass
    else:
        assert False

    try:
        aif = Input({})
    except:
        pass
    else:
        assert False

if __name__=='__main__':
    test_input()