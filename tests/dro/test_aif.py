import numpy as np
from dcmri import aif


def test_parker():

    t = np.arange(0, 6*60, 1)
    ca = aif.parker(t)

    # Test that this generates values in the right range
    assert np.round(1000*np.amax(ca)) == 6

    # Add a delay and check that this produces the same maximum
    ca = aif.parker(t, BAT=60)
    assert np.round(1000*np.amax(ca)) == 6

    # Try with list as input
    ca = aif.parker([50, 100, 150])
    assert np.array_equal(np.round(1000*ca), [1, 1, 1]) 

    # Or just a single variable
    ca = aif.parker(100)
    assert 1000*ca == 0.7929118932243691

    # Check that an error message is generated if BAT is not a scalar
    try:
        ca = aif.parker(t, BAT=[60,120])
    except: 
        assert True
    else:
        assert False


def test_tristan_rat():
    
    t = np.arange(0, 6*60, 1)
    ca = aif.tristan_rat(t)
    assert np.round(1000*np.amax(ca), 1) == 0.3


def test_tristan():

    t = np.arange(0, 6*60, 1)
    ca = aif.tristan(t)
    assert round(max(ca), 4) == 0.0042



if __name__ == "__main__":

    test_parker()
    test_tristan_rat()
    test_tristan()

    print('All aif tests passed!!')
