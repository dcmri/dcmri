"""
Signals and measurement
"""
import math
import numpy as np
import scipy.ndimage as ndi

def signal_SPGRESS(TR, FA, R1, S0):
    """Signal aof a spoiled gradient echo sequence in the steady state"""
    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def sample(t, S, ts, dts): 
    """Sample the signal"""
    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 



def sample_1d(C, shape, loc=None):
    if loc is None:
        loc = sample_loc_1d(C.shape, shape)
    # Sample array
    C = ndi.map_coordinates(C, loc)
    C = np.reshape(C, shape)
    return C

def sample_loc_1d(Cshape, shape):
    # width of the array
    w = [Cshape[0]-1, Cshape[1]]
    # distance between sample points
    d = [w[0]/(shape[0]-1), w[1]/shape[1]]
    # Boundaries of the FOV
    t0, t1 = 0, w[0]
    x0, x1 = -0.5, w[1]-0.5
    # Locations of sample points
    ts0, ts1 = t0, t1
    xs0, xs1 = x0+0.5*d[1], x1-0.5*d[1]
    t = np.linspace(ts0, ts1, shape[0])
    x = np.linspace(xs0, xs1, shape[1])
    # Convert to meshgrid and stack
    t, x = np.meshgrid(t, x, indexing='ij')
    tx = np.column_stack((t.ravel(), x.ravel()))
    return tx.T

# TESTS


def test_sample_loc_1d():
# times
# 0 1 2 3       # original indices
# 0.0   3.0     # locations t0 - t1
# 0     1       # new indices at locations ts0 - ts1
# positions
#        0         1         2         3            # original indices
# -0.5 - 0 - 0.5 - 1 - 1.5 - 2 - 2.5 - 3 - 3.5      # locations x0 - x1
#             0                   1                 # new indices at locs xs0 - xs1

    system_shape = (4,4)
    sample_shape = (2,2)
    loc = sample_loc_1d(system_shape, sample_shape)
    print(loc)
    print(loc.shape)
    print(loc[:,0]) # all times for position = 0
    print(loc[0,:]) # all positions for time = 0

if __name__ == "__main__":
    v = np.array([109, 110, 125])
    print(100*np.std(v)/np.mean(v))
    v = np.array([104, 101, 101])
    print(100*np.std(v)/np.mean(v))
    #test_sample_loc_1d()