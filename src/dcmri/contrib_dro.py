"""
Digital reference objects
"""

import numpy as np
import matplotlib.pyplot as plt

import contrib_aux as aux
import contrib_syst as syst
import aif





def sample(t, S, ts, dts): 
    """Sample the signal"""
    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 


def step_inject(t, weight=70, conc=0.5, dose=0.2, rate=2, start=0):
    """
    Injected flux into the body as a step function

    weight (kg)
    conc (mmol/mL)
    dose mL/kg
    rate mL/sec

    returns dose injected per unit time (mmol/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > start) & (t < start+duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    return J

def step_input(t, weight=70, dose=0.1, conc=0.1, flow=100, start=0):
    """
    Injected flux into an organ

    weight (kg)
    dose (mmol/kg)
    conc (mM)
    flow (mL/sec) = cardiac output for AIF = 100mL/sec = 6L/min
    with these defaults: 
        the injected dose is 0.1 mmol (weight*dose)
        the injected volume is 1000mL (weight*dose/conc) # 0.1 mmol/(0.1 mmol/L) = 1L
        the injection duration is 10 sec (1000mL/100mL/sec = 10). 
    returns flux injected per unit time (mmol/sec)"""

    N = weight*dose # mmol - total amount of tracer injected.
    V = N/conc # mL - total volume injected
    T = V/flow # sec - duration of injection
    J = flow*conc # mmol/sec - flux

    t_inject = (t > start) & (t < start+T)
    Jt = np.zeros(t.size)
    Jt[np.nonzero(t_inject)[0]] = J
    return Jt


def step_conc(t, conc=1, start=0, duration=15):
    # 1 mM * 15 sec
    t_inject = (t > start) & (t < start+duration)
    ct = np.zeros(t.size)
    ct[np.nonzero(t_inject)[0]] = conc
    return ct


def flow_organ_1d(
        length = 25,    # cm
        nr = 20,        # Nr of voxels
        umin = 5,       # cm/sec
        umax = 20,      # cm/sec
        flow = 4,       # mL/sec/cm^2
        ):
    dx = length/nr
    # voxel centers
    c = np.arange(dx/2, length+dx/2, dx)
    # voxel boundaries
    b = np.arange(0, length+dx, dx)
    # velocity = f/v
    u = aux.quad(b, [umax,umin,umax])
    # volume fractions
    v = flow/u
    v = (v[1:]+v[:-1])/2
    return c, b, u, v


def organ_perf_1d(step=True, 
        tmax=30, # sec
    ):
    # area = volume/length/2
    # flow (mL/sec/cm^2) = perf*volume/area = perf*length/2
    # Flow (mL/sec) = perf*volume/2
    # Flux (mmol/sec) = conc*perf*volume /2
    # flux (mmol/sec/cm^2) = conc*perf*length/2
    # dflux (mmol/sec/mL) = conc*perf*length/2/dx
    # vel = flow/v
    # VC1(t+dt) = VC1(t) + dt*flux*A - dt*vel_r*A*C1(t) - dt*perf*V*C1(t)   # mmol
    # C1(t+dt) = C1(t) + dt*flux/dx - dt*vel_r*C1(t)/dx - dt*perf*C1(t)  # mmol
    # C2(t+dt) = C2(t) + dt*vel_l*C1(t)/dx - dt*vel_r*C2(t)/dx - dt*perf*C2(t)  # mmol
    # Geometry
    length = 30 #cm
    nvox = 128 # nr of measured voxels
    nt = 90 # nr of time points
    # Tissue paramaters
    perf = 0.04 # mL/sec/mL
    flow = 0 # total flow mL/sec/cm^2
    vamin = 0.02 # cm/sec
    vvmin = 0.04 # cm/sec
    vamax = 0.1 # cm/sec
    vvmax = 0.3 # cm/sec
    # Create inputs
    fa0 = perf*length/2  # mL/sec/cm^2 
    tb = np.linspace(0, tmax, nt)
    # AIF parameters
    start = 5 # sec
    if step:
        conc = 2 # mM - average in blood
        duration = 10 # sec
        cb = step_conc(tb, conc, start, duration)
    else:
        cb = 0.55*aif.aif_parker(tb, start)
    jpa = fa0*cb # flux in mmol/sec/cm^2
    jna = fa0*cb # flux in mmol/sec/cm^2
    # Create tissue
    xb = np.linspace(0, length, 1+nvox)
    F = aux.gaussian(xb, length/2, length/8)
    F = perf*length*F/np.trapz(F, xb) # normalize so that int(F)/2 = perf*length/2 = fa0
    fa = fa0 - aux.trapz(xb, F)
    fv = flow-fa
    va = aux.gaussian(xb, length/2, length/6)
    va = vamax-(vamax-vamin)*(va-np.amin(va))/(np.amax(va)-np.amin(va))
    vv = aux.gaussian(xb, length/2, length/6)
    vv = vvmax-(vvmax-vvmin)*(vv-np.amin(vv))/(np.amax(vv)-np.amin(vv))
    ua = fa/va
    uv = fv/vv
    Kva = F/va
    dim = [tmax, length]
    mat = [nt, nvox] # acquisition matrix
    system = syst.Perf1D(dim, mat, jna, jpa, ua, uv, Kva, nx=200)
    return system


def organ_perf_1d_fpic(step=True):
    # Geometry
    length = 30 #cm
    tmax = 25 # sec
    nvox = 128 # nr of measured voxels
    nt = 90 # nr of time points
    # Tissue paramaters
    perf = 0.04 # mL/sec/mL
    flow = 0 # total flow mL/sec/cm^2
    vamin = 0.02 # cm/sec
    vvmin = 0.04 # cm/sec
    vamax = 0.1 # cm/sec
    vvmax = 0.3 # cm/sec
    # Create inputs
    fa0 = perf*length/2  # mL/sec/cm^2 
    tb = np.linspace(0, tmax, nt)
    # AIF parameters
    start = 5 # sec
    if step:
        conc = 2 # mM - average in blood
        duration = 10 # sec
        cb = step_conc(tb, conc, start, duration)
    else:
        cb = 0.55*aif.aif_parker(tb, start)
    # Create tissue
    xb = np.linspace(0, length, 1+nvox)
    F = aux.gaussian(xb, length/2, length/8)
    va = aux.gaussian(xb, length/2, length/6)
    va = vamax-(vamax-vamin)*(va-np.amin(va))/(np.amax(va)-np.amin(va))
    vv = aux.gaussian(xb, length/2, length/6)
    vv = vvmax-(vvmax-vvmin)*(vv-np.amin(vv))/(np.amax(vv)-np.amin(vv))
    return syst.Perf1D_fpic(
        dim = [tmax, length], 
        mat = [nt, nvox], 
        Jna = fa0*cb, 
        Jpa = fa0*cb,
        af = va/(va+vv), 
        v = va+vv, 
        F = perf*length*F/np.trapz(F, xb), # normalize so that int(F)/2 = perf*length/2 = fa0, 
        faL = fa0, 
        f = flow, 
        nx = 200, 
        fmax = 30,          # maximum flow (mL/sec/cm^2) 
        Jmax = 10,          # maximum influx (mmol/sec/cm^2)
        Fmax = 1,           # maximum perfusion (mL/sec/mL)
        vmin = 0.01,         # minimum volume fraction
        vmax = 1.0,         # maximum volume fraction
        afmin = 0.05,         # minimum afterial volume fraction
        afmax = 0.95,         # minimum afterial volume fraction
    )




######### TESTS ############

tmax = 120 # sec
dt = 0.01 # sec
MTT = 20 # sec

weight = 70.0           # Patient weight in kg
conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
dose = 0.025            # mL per kg bodyweight (quarter dose)
rate = 1                # Injection rate (mL/sec)
start = 20.0        # sec
dispersion = 90    # %


def test_step_inject():
    t = np.arange(0, tmax, dt)
    J = step_inject(t, weight, conc, dose, rate, start)
    assert np.round(np.sum(J)*dt,1) == np.round(weight*dose*conc, 1)

if __name__ == "__main__":

    #test_step_inject()
    #organ_perf_1d()
    organ_perf_1d_fpic()