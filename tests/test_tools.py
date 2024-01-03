import numpy as np
import dcmri.tools as tools

def test_tarray():
    n = 4
    J = np.zeros(n)
    t = tools.tarray(J)
    assert np.array_equal(t, [0,1,2,3])
    t = tools.tarray(J, dt=2)
    assert np.array_equal(t, [0,2,4,6])
    t = tools.tarray(J, [1,2,3,9])
    assert np.array_equal(t, [1,2,3,9])
    try:
        t = tools.tarray(J, [1,2,3])
    except:
        assert True
    else:
        assert False

def test_prop_ddelta():
    t = [0,1,2,3]
    h = tools.prop_ddelta(t)
    assert np.array_equal(h, [2,0,0,0])
    assert np.trapz(h,t) == 1
    t = [0,2,3,4]
    h = tools.prop_ddelta(t)
    assert np.array_equal(h, [1,0,0,0])
    assert np.trapz(h,t) == 1
    t = [1,2,3,4]
    h = tools.prop_ddelta(t)
    assert np.array_equal(h, [0,0,0,0])
    assert np.trapz(h,t) == 0

    # Check that this is a unit for the convolution
    t = np.array([0,2.5,3,5,8,10,15,20])
    #t = np.arange(10)
    f = np.exp(-t/30)/30
    h = tools.prop_ddelta(t)
    g = tools.conv(f, h, t)
    assert np.linalg.norm(g[1:]-f[1:])/np.linalg.norm(f[1:]) < 1e-9


def test_res_ddelta():
    t = [0,1,2,3]
    r = tools.res_ddelta(t)
    assert np.array_equal(r, [1,0,0,0])
    t = [0,2,3,4]
    r = tools.res_ddelta(t)
    assert np.array_equal(r, [1,0,0,0])
    t = [1,2,3,4]
    r = tools.res_ddelta(t)
    assert np.array_equal(r, [0,0,0,0])


def test_trapz():
    t = np.arange(0, 60, 10)
    ca = (t/np.amax(t))**2
    c = tools.trapz(ca, t)
    assert c[1] == 0.20000000000000004


def test_expconv():
    t = np.arange(0, 60, 10)
    ca = (t/np.amax(t))**2
    c = tools.expconv(ca, 30, t)
    assert c[1] == 0.005983757268854711


def test_uconv():
    dt = 0.1
    t = np.arange(0,30,dt)
    f = np.exp(-t/20)/20
    h = np.exp(-t/30)/30
    g = tools.uconv(f, h, dt)
    g0 = tools.expconv(f, 30, t)
    assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < 1e-5

    dt = 0.01
    t = np.arange(0,30,dt)
    f = np.exp(-t/20)/20
    h = np.exp(-t/30)/30
    g = tools.uconv(f, h, dt)
    g0 = tools.expconv(f, 30, t)
    assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < 1e-7


def test_conv():
    t = np.array([0,1,3,5,8,10,15,20])
    f = np.exp(-t/20)/20
    h = np.exp(-t/30)/30
    g = tools.conv(f, h, t)
    g0 = tools.expconv(f, 30, t)
    assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < 1e-3

    t = np.arange(0,30,0.01)
    f = np.exp(-t/20)/20
    h = np.exp(-t/30)/30
    g = tools.conv(f, h, t)
    g0 = tools.expconv(f, 30, t)
    assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < 1e-6

    g0 = tools.expconv(f, 30, dt=0.01)
    assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < 1e-6

    try:
        g0 = tools.expconv(f, 30, [1,2])
    except:
        assert True
    else:
        assert False



if __name__ == "__main__":

    test_tarray()
    test_res_ddelta()
    test_prop_ddelta()
    test_trapz()
    test_expconv()
    test_uconv()
    test_conv()

    print('All tools tests passed!!')
