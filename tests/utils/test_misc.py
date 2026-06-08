import numpy as np

import dcmri as dc
from dcmri.utils import misc



# Helper
def tfib(n, tmax=1.0):
    t = np.empty(n)
    t[0] = 0
    t[1] = 1
    t[2] = 2
    k=3
    while k<n:
        t[k]=t[k-1]+t[k-2]
        k+=1
    return tmax*t/t.max()


def test_tarray():
    n = 4
    J = np.zeros(n)
    t = misc.tarray(len(J))
    assert np.array_equal(t, [0,1,2,3])
    t = misc.tarray(len(J), dt=2)
    assert np.array_equal(t, [0,2,4,6])
    t = misc.tarray(len(J), [1,2,3,9])
    assert np.array_equal(t, [1,2,3,9])
    try:
        t = misc.tarray(len(J), [1,2,3])
    except:
        assert True
    else:
        assert False


def test_interp():
    x = np.arange(3)
    assert np.array_equal(misc.interp(3, x, pos=False, floor=False), [3,3,3])
    assert np.array_equal(misc.interp([3], x, pos=False, floor=False), [3,3,3])
    assert np.array_equal(misc.interp([3,4], x, pos=False, floor=False), [3,3.5,4])
    assert np.array_equal(misc.interp([3,4,5], x, pos=False, floor=False), [3,4,5])
    assert np.array_equal(misc.interp(np.arange(5), x, pos=False, floor=False), [0,2,4])
    assert np.array_equal(misc.interp(np.arange(5), x, pos=True, floor=True), [0,2,4])
    assert np.array_equal(misc.interp(np.arange(5), x, pos=True, floor=True, lower=-1), [0,2,4])
    assert np.array_equal(misc.interp(np.arange(5), x, pos=True, floor=True, upper=5), [0,2,4])


def test_sample():

    tp = np.array([2,4,5,7])
    Sp = np.array([1,2,5,9])

    S = dc.sample(np.array([3]), tp, Sp)
    assert np.array_equal(S, [1.5])
    S = dc.sample(np.array([3]), tp, Sp, dt=1)
    assert np.array_equal(S, [1.5])
    S = dc.sample(np.array([3]), tp, Sp, dt=0.1)
    assert np.array_equal(S.astype(np.float32), [1.5])
    S = dc.sample(np.array([3]), tp, Sp, dt=2.0)
    assert np.array_equal(S.astype(np.float32), [1.5])
    S = dc.sample(np.array([3,6]), tp, Sp, dt=1)
    assert np.array_equal(S, [1.5,7])
    S = dc.sample([], tp, Sp, dt=1)
    assert np.array_equal(S, [])
    S = dc.sample(np.array([1,6]), tp, Sp, dt=1)
    assert np.array_equal(S, [1,7])

def test_add_noise():
    s0 = [1,2,3,4]
    s1 = dc.add_noise(s0, 0)
    assert np.array_equal(s0, s1)

def test_trapz():
    t = np.arange(0, 60, 10)
    ca = (t/np.amax(t))**2
    c = misc.trapz(ca, t)
    assert c[1] == 0.20000000000000004

def test_mle_rice():
    data = np.arange(10)
    mle = dc.mle_rice(data, fit_loc=False)
    assert round(mle['nu'], 1) == 3.9
    mle = dc.mle_rice(data, fit_loc=True)
    assert round(mle['nu'], 1) == 4.1
    try:
        data[0] = -1
        mle = dc.mle_rice(data, fit_loc=False)
    except:
        pass
    else:
        assert False

def test_describe():
    arr = np.ones((2,3))
    desc = misc.describe(arr)
    assert np.array_equal(desc['Sb'], np.ones(arr.shape[0]))
    
    try:
        arr = np.ones((2,3))
        desc = misc.describe(arr, rician=True)
    except:
        pass
    else:
        assert False

    arr = np.ones((4,5))
    desc = misc.describe(arr, n0=3, rician=True)
    assert np.array_equal(desc['Sb'], np.ones(arr.shape[0]))



if __name__ == "__main__":
    test_trapz()
    test_interp()
    test_tarray()
    test_sample()
    test_add_noise()
    test_mle_rice()
    test_describe()

    print('All misc tests passed!!')
