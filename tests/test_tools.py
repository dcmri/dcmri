import numpy as np
import dcmri.tools as tools

def test_tarray():
    n = 4
    J = np.zeros(n)
    t = tools.tarray(len(J))
    assert np.array_equal(t, [0,1,2,3])
    t = tools.tarray(len(J), dt=2)
    assert np.array_equal(t, [0,2,4,6])
    t = tools.tarray(len(J), [1,2,3,9])
    assert np.array_equal(t, [1,2,3,9])
    try:
        t = tools.tarray(len(J), [1,2,3])
    except:
        assert True
    else:
        assert False


def test_trapz():
    t = np.arange(0, 60, 10)
    ca = (t/np.amax(t))**2
    c = tools.trapz(ca, t)
    assert c[1] == 0.20000000000000004


def test_expconv():

    # Uniform time grid with increasing precision: compare against analytical convolution.

    Tf = 20
    Th = 30
    tmax = 30

    prec = [1e-1, 1e-3, 1e-5]
    for i, dt in enumerate([10,1,0.1]):
        t = np.arange(0,tmax,dt)
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = tools.expconv(f, Th, dt=dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    # Non-uniform time grid with increasing precision: check against analytical convolution.
    t0 = np.array([0,1,2,3,5,8,13,21,34])
    prec = [0.05, 1e-2, 1e-3, 1e-5]
    for i, dt0 in enumerate([10,1,0.1,0.01]):
        t = dt0*t0
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = tools.expconv(f, Th, t)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

def test_inttrap():
    t=np.array([0,1,2,3])
    f=[1,1,1,1]
    assert tools.inttrap(f,t,0.5,1.5) == 1

def test_stepconv():
    T = 3.5
    D = 0.5
    T0 = T-D*T
    T1 = T+D*T
    # Check against conv at high res
    prec = [0.04, 0.02, 0.002]
    for k, n in enumerate([10,100,1000]):
        t = np.linspace(0,10,n) 
        h = np.zeros(n)
        h[(t>=T0)*(t<=T1)] = 1/(T1-T0)
        f = np.sqrt(t)
        g = tools.stepconv(f, T, D, dt=t[1])
        g0 = tools.conv(f, h, dt=t[1])
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[k]

def test_intprod():
    # Non-uniform time interval: compare to numerical integration.
    t = [0,2,6]
    f = [1,10,3]
    h = [5,1,7]
    i = tools.intprod(f, h, t)
    n = 1000
    t1 = np.linspace(t[0],t[1],n)
    f1 = np.interp(t1, t[0:2], f[0:2])
    h1 = np.interp(t1, t[0:2], h[0:2])
    i1 = np.trapz(f1*h1, t1)
    t2 = np.linspace(t[1],t[2],n)
    f2 = np.interp(t2, t[1:3], f[1:3])
    h2 = np.interp(t2, t[1:3], h[1:3])
    i2 = np.trapz(f2*h2, t2)
    assert (i-(i1+i2))**2/(i1+i2)**2 < 1e-12

    # Uniform time interval: compare to numerical integration.
    dt = 2
    t = dt*np.arange(3)
    f = [1,10,3]
    h = [5,1,7]
    i = tools.intprod(f, h, dt=dt)
    n = 1000
    t1 = np.linspace(t[0],t[1],n)
    f1 = np.interp(t1, t[0:2], f[0:2])
    h1 = np.interp(t1, t[0:2], h[0:2])
    i1 = np.trapz(f1*h1, t1)
    t2 = np.linspace(t[1],t[2],n)
    f2 = np.interp(t2, t[1:3], f[1:3])
    h2 = np.interp(t2, t[1:3], h[1:3])
    i2 = np.trapz(f2*h2, t2)
    assert (i-(i1+i2))**2/(i1+i2)**2 < 1e-12


def test_uconv():
    # Compare against analytical convolution for 3 time intervals

    Tf = 20
    Th = 30
    tmax = 30

    prec = [1e-1, 1e-3, 1e-5]

    for i, dt in enumerate([10,1,0.1]):
        t = np.arange(0,tmax,dt)
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = tools.uconv(f, h, dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]


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

def test_conv():

    # Uniform time grid with increasing precision: compare against analytical convolution.

    Tf = 20
    Th = 30
    tmax = 30

    prec = [1e-1, 1e-3, 1e-5]
    for i, dt in enumerate([10,1,0.1]):
        t = np.arange(0,tmax,dt)
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = tools.conv(f, h, dt=dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    # Non-uniform time grid with increasing precision: check against analytical convolution.
    t0 = np.array([0,1,2,3,5,8,13,21,34])
    prec = [0.05, 0.02, 1e-3, 1e-5]
    for i, dt0 in enumerate([10,1,0.1,0.01]):
        t = dt0*t0
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = tools.conv(f, h, t)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    # Uniform time grid: check area preserving and symmetric at any time resolution
    nt = [5,10,100]
    tmax = 150
    prec_area = [1e-4, 1e-5, 1e-9]
    prec_symm = 1e-15
    for i, n in enumerate(nt):
        t = np.linspace(0,tmax,n)
        dt = t[1]-t[0]
        f = np.exp(-t/10)
        h = np.exp(-((t-30)/15)**2)
        area = np.trapz(f,t)*np.trapz(h,t)
        g0 = tools.conv(f, h, dt=dt)
        g1 = tools.conv(h, f, dt=dt)
        assert (np.trapz(g0,t)-area)**2/area**2 < prec_area[i]
        assert np.linalg.norm(g0-g1)/np.linalg.norm(g0)  < prec_symm

    # Non-uniform time grid: check area preserving and symmetric
    nt = [5,10,50,100,500]
    tmax = 150
    prec_symm = 1e-14
    prec_area = 0.002
    for i, n in enumerate(nt):
        t = tfib(n, tmax)
        f = np.exp(-t/10)
        h = np.exp(-((t-30)/15)**2)
        area = np.trapz(f,t)*np.trapz(h,t)
        g0 = tools.conv(f, h, t)
        g1 = tools.conv(h, f, t)
        assert (np.trapz(g0,t)-area)**2/area**2 < prec_area
        assert np.linalg.norm(g0-g1)/np.linalg.norm(g0)  < prec_symm

    # Check error handling
    try:
        tools.conv([1,2,3], [1,2])
    except:
        assert True
    else:
        assert False

def test_biexpconv():
    Tf = 20
    Th = 30
    t = np.array([0,1,2,3,5,8,13,21,34])
    g = tools.biexpconv(Tf, Th, t)
    g0 = (np.exp(-t/Tf)-np.exp(-t/Th))/(Tf-Th)
    assert np.linalg.norm(g-g0) == 0#

def test_nexpconv():
    MTT = 20

    # High-res pseudocontinuous
    # Uniform time interval
    t = np.linspace(0, 10*MTT, 1000)

    n=2
    T=MTT/n
    g = tools.nexpconv(n, T, t)
    # Check against analytical
    g0 = (t/T) * np.exp(-t/T)/T
    assert np.linalg.norm(g-g0) == 0
    # Check against expconv
    g0 = tools.expconv(np.exp(-t/T)/T, T, t)
    assert np.linalg.norm(g-g0) < 1e-4
    # Check area = 1
    assert np.abs(np.trapz(g,t)-1) < 1e-3
    # Check MTT
    assert np.abs(np.trapz(t*g,t)-MTT) < 1e-5
    
    n=20
    T=MTT/n
    g = tools.nexpconv(n, T, t)
    # Check area = 1
    assert np.abs(np.trapz(g,t)-1) < 1e-12
    # Check MTT
    assert np.abs(np.trapz(t*g,t)-MTT) < 1e-12

    # In this case the numerical approximation is used
    n=200
    T=MTT/n
    g = tools.nexpconv(n, T, t)
    # Check area = 1
    assert np.abs(np.trapz(g,t)-1) < 1e-12
    # Check MTT
    assert np.abs(np.trapz(t*g,t)-MTT) < 0.1

def test_ddelta():
    t = [0,2,3,4]
    h = tools.ddelta(-1, t)
    assert np.array_equal(h, [0,0,0,0])
    h = tools.ddelta(0, t)
    assert np.array_equal(h, [1,0,0,0])
    assert np.trapz(h,t) == 1
    h = tools.ddelta(1, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = tools.ddelta(2, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = tools.ddelta(3.5, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12
    h = tools.ddelta(4, t)
    assert np.abs(np.trapz(h,t)-1) < 1e-12

    # Check that this is a unit for the convolution.
    t = tfib(10, 30)
    h = tools.ddelta(0,t)
    f = np.exp(-t/30)/30
    g = tools.conv(f, h, t)
    assert np.linalg.norm(g[1:]-f[1:])/np.linalg.norm(f[1:]) < 1e-2


if __name__ == "__main__":

    test_trapz()
    test_expconv()
    test_inttrap()
    test_stepconv()
    test_intprod()
    test_uconv()
    test_conv()
    test_tarray()
    test_ddelta()
    test_nexpconv()
    test_biexpconv()

    print('All tools tests passed!!')
