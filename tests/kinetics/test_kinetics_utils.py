import numpy as np
from scipy.integrate import trapezoid

import dcmri as dc
from dcmri.kinetics import functions_utils


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



def test_ddelta():
    t = [0,2,3,4]
    h = functions_utils.ddelta(-1, t)
    assert np.array_equal(h, [0,0,0,0])
    h = functions_utils.ddelta(5, t)
    assert np.array_equal(h, [0,0,0,0])
    h = functions_utils.ddelta(0, t)
    assert np.array_equal(h, [1,0,0,0])
    assert trapezoid(h,t) == 1
    h = functions_utils.ddelta(1, t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    h = functions_utils.ddelta(2, t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    h = functions_utils.ddelta(3.5, t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    h = functions_utils.ddelta(4, t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12

    # Check that this is a unit for the dc.
    t = tfib(10, 30)
    h = functions_utils.ddelta(0,t)
    f = np.exp(-t/30)/30
    g = dc.conv(f, h, t)
    assert np.linalg.norm(g[1:]-f[1:])/np.linalg.norm(f[1:]) < 1e-2

def test_dstep():
    t = [0,2,3,4]
    h = functions_utils.dstep(0, 4, t)
    assert np.array_equal(h, [0.25,0.25,0.25,0.25])
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    h = functions_utils.dstep(0.5, 3.5, t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    t = [0,1,2,3]
    h = functions_utils.dstep(0.5, 2.5, t)
    assert np.array_equal(h, [0.2,0.4,0.4,0.2])
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    t = [-1,0,1,2]
    h = functions_utils.dstep(0.5, 2.5, t)
    assert np.array_equal(h, [0, 0.25, 0.5, 0.5])
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    t = [1,2,3,4]
    h = functions_utils.dstep(0.5, 2.5, t)
    assert np.array_equal(h, [0.5, 0.5, 0.25, 0])
    assert np.abs(trapezoid(h,t)-1) < 1e-12

def test_ddist():
    t = [0,2,3,4]
    h = functions_utils.ddist([1/3,1/3,1/3], [0,2,3,4], t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    t = [0,1,2,3]
    h = functions_utils.ddist([1/3,1/3,1/3], [0,1,2,3], t)
    assert np.array_equal(h, [1/3,1/3,1/3,1/3])
    assert np.abs(trapezoid(h,t)-1) < 1e-12
    h = functions_utils.ddist([0.25,0.5,0.25], [0,1,2,3], t)
    assert np.abs(trapezoid(h,t)-1) < 1e-12





if __name__ == "__main__":
    test_ddelta()
    test_dstep()
    test_ddist()

    print('All misc tests passed!!')
