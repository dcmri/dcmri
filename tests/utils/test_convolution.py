import numpy as np
from scipy.integrate import trapezoid

import dcmri as dc
from dcmri.utils import convolution

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
        g = dc.expconv(f, Th, dt=dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    # Non-uniform time grid with increasing precision: check against analytical convolution.
    t0 = np.array([0,1,2,3,5,8,13,21,34])
    prec = [0.05, 1e-2, 1e-3, 1e-5]
    for i, dt0 in enumerate([10,1,0.1,0.01]):
        t = dt0*t0
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = dc.expconv(f, Th, t)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    #Special case: T=0
    t = np.arange(0,tmax,dt)
    f = np.exp(-t/Tf)/Tf
    assert np.array_equal(f, dc.expconv(f, 0, t))


def test_inttrap():
    t=np.array([0,1,2,3])
    f=[1,1,1,1]
    assert convolution.inttrap(f,t,0.5,1.5) == 1

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
        g = dc.stepconv(f, T, D, dt=t[1])
        g0 = dc.conv(f, h, dt=t[1])
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[k]
    try:
        dc.stepconv(f, T, 2, dt=t[1])
    except:
        assert True


def test_intprod():
    # Non-uniform time interval: compare to numerical integration.
    t = [0,2,6]
    f = [1,10,3]
    h = [5,1,7]
    i = convolution.intprod(f, h, t)
    n = 1000
    t1 = np.linspace(t[0],t[1],n)
    f1 = np.interp(t1, t[0:2], f[0:2])
    h1 = np.interp(t1, t[0:2], h[0:2])
    i1 = trapezoid(f1*h1, t1)
    t2 = np.linspace(t[1],t[2],n)
    f2 = np.interp(t2, t[1:3], f[1:3])
    h2 = np.interp(t2, t[1:3],h[1:3])
    i2 = trapezoid(f2*h2, t2)
    assert (i-(i1+i2))**2/(i1+i2)**2 < 1e-12

    # Uniform time interval: compare to numerical integration.
    dt = 2
    t = dt*np.arange(3)
    f = [1,10,3]
    h = [5,1,7]
    i = convolution.intprod(f, h, dt=dt)
    n = 1000
    t1 = np.linspace(t[0],t[1],n)
    f1 = np.interp(t1, t[0:2], f[0:2])
    h1 = np.interp(t1, t[0:2], h[0:2])
    i1 = trapezoid(f1*h1, t1)
    t2 = np.linspace(t[1],t[2],n)
    f2 = np.interp(t2, t[1:3], f[1:3])
    h2 = np.interp(t2, t[1:3], h[1:3])
    i2 = trapezoid(f2*h2, t2)
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
        g = convolution.uconv(f, h, dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]


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
        g = dc.conv(f, h, dt=dt)
        g0 = (Tf*f-Th*h)/(Tf-Th)
        assert np.linalg.norm(g-g0)/np.linalg.norm(g0) < prec[i]

    # Non-uniform time grid with increasing precision: check against analytical convolution.
    t0 = np.array([0,1,2,3,5,8,13,21,34])
    prec = [0.05, 0.02, 1e-3, 1e-5]
    for i, dt0 in enumerate([10,1,0.1,0.01]):
        t = dt0*t0
        f = np.exp(-t/Tf)/Tf
        h = np.exp(-t/Th)/Th
        g = dc.conv(f, h, t)
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
        area = trapezoid(f,t)*trapezoid(h,t)
        g0 = dc.conv(f, h, dt=dt)
        g1 = dc.conv(h, f, dt=dt)
        assert (trapezoid(g0,t)-area)**2/area**2 < 5*prec_area[i]
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
        area = trapezoid(f,t)*trapezoid(h,t)
        g0 = dc.conv(f, h, t)
        g1 = dc.conv(h, f, t)
        assert (trapezoid(g0,t)-area)**2/area**2 < prec_area
        assert np.linalg.norm(g0-g1)/np.linalg.norm(g0)  < prec_symm

    # compare trap and step solvers - should be identical at high temporal resolution
    dt = 0.1
    t = np.arange(0,tmax,dt)
    f = np.exp(-t/Tf)/Tf
    h = np.exp(-t/Th)/Th
    g0 = (Tf*f-Th*h)/(Tf-Th)
    g = dc.conv(f, h, dt=dt)
    assert np.linalg.norm(g-g0) < 1e-3*np.linalg.norm(g0)
    g = dc.conv(f, h, dt=dt, solver='trap')
    assert np.linalg.norm(g-g0) < 1e-3*np.linalg.norm(g0)
    g = dc.conv(f, h, t)
    assert np.linalg.norm(g-g0) < 1e-3*np.linalg.norm(g0)
    g = dc.conv(f, h, t, solver='trap')
    assert np.linalg.norm(g-g0) < 1e-3*np.linalg.norm(g0)

    # Check error handling
    try:
        dc.conv([1,2,3], [1,2])
    except:
        assert True
    else:
        assert False

def test_biexpconv():
    Tf = 20
    Th = 30
    t = np.array([0,1,2,3,5,8,13,21,34])
    g = dc.biexpconv(Tf, Th, t)
    g0 = (np.exp(-t/Tf)-np.exp(-t/Th))/(Tf-Th)
    assert np.linalg.norm(g-g0) == 0
    g = dc.biexpconv(Th, Th, t)
    g0 = (t/Th) * np.exp(-t/Th)/Th
    assert np.linalg.norm(g-g0) == 0


def test_nexpconv():
    MTT = 20

    # High-res pseudocontinuous
    # Uniform time interval
    t = np.linspace(0, 10*MTT, 1000)

    n=2
    T=MTT/n
    g = dc.nexpconv(n, T, t)
    # Check against analytical
    g0 = (t/T) * np.exp(-t/T)/T
    assert np.linalg.norm(g-g0) == 0
    # Check against expconv
    g0 = dc.expconv(np.exp(-t/T)/T, T, t)
    assert np.linalg.norm(g-g0) < 1e-4
    # Check area = 1
    assert np.abs(trapezoid(g,t)-1) < 1e-3
    # Check MTT
    assert np.abs(trapezoid(t*g,t)-MTT) < 1e-5
    
    n=20
    T=MTT/n
    g = dc.nexpconv(n, T, t)
    # Check area = 1
    assert np.abs(trapezoid(g,t)-1) < 1e-12
    # Check MTT
    assert np.abs(trapezoid(t*g,t)-MTT) < 1e-12

    # In this case the numerical approximation is used
    n=200
    T=MTT/n
    g = dc.nexpconv(n, T, t)
    # Check area = 1
    assert np.abs(trapezoid(g,t)-1) < 1e-12
    # Check MTT
    assert np.abs(trapezoid(t*g,t)-MTT) < 0.1
    # Check case of non-integer n
    g = dc.nexpconv(200.5, T, t)
    assert np.abs(trapezoid(g,t)-1) < 1e-12

    # Test list input format
    g = dc.nexpconv(200.5, T, list(t))
    assert np.abs(trapezoid(g,t)-1) < 1e-12

    # Test exceptions
    try:
        dc.nexpconv(n, -1, t)
    except:
        assert True
    try:
        dc.nexpconv(0.5, T, t)
    except:
        assert True


if __name__ == "__main__":
    
    test_intprod()
    test_uconv()
    test_conv()
    test_inttrap()
    test_stepconv()
    test_expconv()
    test_biexpconv()
    test_nexpconv()

    print('All convolution tests passed!!')