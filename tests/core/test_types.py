from dcmri.core import Input

def test_input():
    aif = Input([1,2,3])
    assert aif.signal[-1] == 3

if __name__=='__main__':
    test_input()