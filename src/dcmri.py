import math
import numpy as np
from scipy.special import gamma


def expconv(T, time, a):
    """Convolve a 1D-array with a normalised exponential.

    expconv() uses an efficient and accurate numerical formula to calculate the convolution,
    as detailed in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006.

    Note (1): by definition, expconv preserves the area under a(time)
    Note (2): if T=0, expconv returns a copy of a

    Arguments
    ---------
    a : numpy array
        the 1D array to be convolved.
    time : numpy array
        the time points where the values of ca are defined
        these do not have to to be equally spaced.
    T : float
        the characteristic time of the the exponential function.
        time and T must be in the same units.

    Returns
    -------
    a numpy array of the same shape as ca.

    Example
    -------
    coming soon..

    """
    if T==0: return a

    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

def convolve(u, tc, c, th, h):

#    co(t) = int_0^t du h(u) c(t-u) 

    co = np.zeros(len(u))
    h = np.interp(u, th, h, left=0, right=0)
    c = np.interp(u, tc, c, left=0, right=0)
    for k, t in enumerate(u):
        if k != 0:
            ct = np.interp(t-u, u, c, left=0, right=0)
            co[k] = np.trapz(h[:k]*ct[:k], u[:k])
    return co   

def compartment_propagator(t, MTT):
    return np.exp(-t/MTT)/MTT

def propagate_compartment(t, c, MTT):
    """Returns the average concentration at the outlet given the concentration at the inlet"""
    return expconv(MTT, t, c)

def residue_compartment(t, c, MTT):
    """Returns the concentration inside the system given the concentration at the inlet"""
    return propagate_compartment(t, c, MTT)

def propagate_dd(t, c, MTT, TTD):
    """
    Propagate concentration through a serial arrangement of a plug flow and a compartment.

    Arguments
    ---------
    TTD : Transit Time Dispersion of the system
        This is the mean transit time of the compartment
    MTT : Mean Transit Time of the system
        This is the sum of delay and MTT of the compartment

    Returns
    -------
    Concentration at the outlet
    """

    delay = MTT - TTD 
    c = expconv(TTD, t, c)
    c = np.interp(t-delay, t, c, left=0)
    return c

def chain_propagator(t, MTT, dispersion): # dispersion in %
    # Needs error handling for dispersion=0 case - not numerically feasible.
    n = 100/dispersion
    Tx = MTT/n
    return (np.exp(-t/Tx)/Tx) * (t/Tx)**(n-1)/gamma(n)

def propagate_chain(t, ci, MTT, dispersion): # dispersion in % 

    if MTT == 0:
        return ci
    if dispersion == 0:
        return np.interp(t-MTT, t, ci, left=0)
    H = chain_propagator(t, MTT, dispersion)
    return convolve(t, t, ci, t, H)

def residue_chain(t, ci, MTT, dispersion):
    """Returns the (average) concentration inside the system given the concentration at the inlet"""
    co = propagate_chain(t, ci, MTT, dispersion)
    return np.trapz(ci-co, t)/MTT

def propagate_delay(t, c, delay):

    return np.interp(t-delay, t, c, left=0) 

def propagate_2cxm(t, ca, KP, KE, KB):
    """Calculate the propagators for the individual compartments in the 2CXM 
    
    For details and notations see appendix of 
    Sourbron et al. Magn Reson Med 62:672–681 (2009)

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    ca : numpy array
        input function (mmol/mL)
    KP : float
        inverse plasma MTT (sec) = VP/(FP+PS)
    KE : float
        inverse extracellular MTT (sec) = VE/PS
    KB : float
        inverse blood MTT (sec) = VP/FP

    Returns
    -------
    cp : numpy array
        concentration in the plasma compartment (mmol/mL)
    ce : numpy array
        concentration in the extracellular compartment (mmol/mL)

    Examples
    --------
    coming soon..

    """

    KT = KP + KE
    sqrt = math.sqrt(KT**2-4*KE*KB)

    Kpos = 0.5*(KT + sqrt)
    Kneg = 0.5*(KT - sqrt)

    cpos = expconv(1/Kpos, t, ca)
    cneg = expconv(1/Kneg, t, ca)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    cp = (1-Eneg)*cpos + Eneg*cneg
    ce = (cneg*Kpos - cpos*Kneg) / (Kpos -  Kneg) 

    return cp, ce

def propagate_simple_body(t, c_vena_cava, 
    MTTlh, Eint, MTTe, MTTo, TTDo, Eext):
    """Propagation through a 2-site model of the body."""

    dose0 = np.trapz(c_vena_cava, t)
    dose = dose0
    min_dose = 10**(-3)*dose0

    c_vena_cava_total = 0*t
    c_aorta_total = 0*t

    while dose > min_dose:
        c_aorta = expconv(MTTlh, t, c_vena_cava)
        c_aorta_total += c_aorta
        c_vena_cava_total += c_vena_cava
        c = propagate_dd(t, c_aorta, MTTo, TTDo)
        c = (1-Eint)*c + Eint*expconv(MTTe, t, c) 
        c_vena_cava = c*(1-Eext)
        dose = np.trapz(c_vena_cava, t)

    return c_vena_cava_total, c_aorta_total

def residue_high_flow_ccf(t, ci, Ktrans, Te, De, FiTi):
    """Residue for a compartment i with high flow (Ti=0) and a chain e"""

    ni = FiTi*ci
    ne = (Te*Ktrans)*residue_chain(t, ci, Te, De)
    return ni, ne

def residue_high_flow_2cfm(t, ci, Ktrans, Te, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    ni = FiTi*ci
    ne = (Te*Ktrans)*propagate_compartment(t, ci, Te)
    return ni, ne

def residue_high_flow_2cfm_varK(t, ci, Ktrans1, Ktrans2, Ktrans3, Te, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    # ve dce/dt = Ktrans*ci - k*ce
    # dne/dt = Ktrans * ci - ne / Te
    # Analytical solution with constant Te:
    #   ne(t) = exp(-t/Te) * Ktrans ci(t) 
    #   ne(t) = Te Ktrans P(Te, t) * ci(t)
    # Numerical solution with variable Te
    #   (ne(t+dt)-ne(t))/dt = Ktrans ci(t) - ne(t) / Te(t)
    #   ne(t+dt) = ne(t) + dt Ktrans ci(t) - ne(t) dt/Te(t)
    #   ne(t+dt) = dt Ktrans ci(t) + (1-dt/Te(t)) * ne(t)

    # Build time-varying Te (step function)
    # Te = np.empty(len(t))
    # tmid = math.floor(len(t)/2)
    # Te[:tmid] = Te1
    # Te[tmid:] = Te2
    mid = math.floor(len(t)/2)
    Ktrans = quadratic(t, t[0], t[mid], t[-1], Ktrans1, Ktrans2, Ktrans3)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re * ne[k]
    return ni, ne, Ktrans

def residue_high_flow_2cfm_varT(t, ci, Ktrans, Te1, Te2, Te3, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    # ve dce/dt = Ktrans*ci - k*ce
    # dne/dt = Ktrans * ci - ne / Te
    # Analytical solution with constant Te:
    #   ne(t) = exp(-t/Te) * Ktrans ci(t) 
    #   ne(t) = Te Ktrans P(Te, t) * ci(t)
    # Numerical solution with variable Te
    #   (ne(t+dt)-ne(t))/dt = Ktrans ci(t) - ne(t) / Te(t)
    #   ne(t+dt) = ne(t) + dt Ktrans ci(t) - ne(t) dt/Te(t)
    #   ne(t+dt) = dt Ktrans ci(t) + (1-dt/Te(t)) * ne(t)

    # Build time-varying Te (step function)
    # Te = np.empty(len(t))
    # tmid = math.floor(len(t)/2)
    # Te[:tmid] = Te1
    # Te[tmid:] = Te2
    mid = math.floor(len(t)/2)
    Te = quadratic(t, t[0], t[mid], t[-1], Te1, Te2, Te3)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re[k] * ne[k]
    return ni, ne


def residue_high_flow_2cfm_varlinT(t, ci, Ktrans, Te1, Te2, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    # ve dce/dt = Ktrans*ci - k*ce
    # dne/dt = Ktrans * ci - ne / Te
    # Analytical solution with constant Te:
    #   ne(t) = exp(-t/Te) * Ktrans ci(t) 
    #   ne(t) = Te Ktrans P(Te, t) * ci(t)
    # Numerical solution with variable Te
    #   (ne(t+dt)-ne(t))/dt = Ktrans ci(t) - ne(t) / Te(t)
    #   ne(t+dt) = ne(t) + dt Ktrans ci(t) - ne(t) dt/Te(t)
    #   ne(t+dt) = dt Ktrans ci(t) + (1-dt/Te(t)) * ne(t)

    # Build time-varying Te (step function)
    # Te = np.empty(len(t))
    # tmid = math.floor(len(t)/2)
    # Te[:tmid] = Te1
    # Te[tmid:] = Te2
    Te = linear(t, t[0], t[-1], Te1, Te2)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re[k] * ne[k]
    return ni, ne


def injection(t, weight, conc, dose1, rate, start1, dose2=None, start2=None):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose1/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J1 = propagate_delay(t, J, start1)
    if start2 is None:
        return J1
    duration = weight*dose2/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J2 = propagate_delay(t, J, start2)
    return J1 + J2

def injection_gv(t, weight, conc, dose, rate, start1, start2=None, dispersion=0.5):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    amount = conc*weight*dose       # mmol = (mmol/ml) * kg * (ml/kg)
    J = amount * chain_propagator(t, duration, dispersion) # mmol/sec
    J1 = propagate_delay(t, J, start1)
    if start2 is None:
        return J1
    else:
        J2 = propagate_delay(t, J, start2)
        return J1 + J2

def signalSPGRESS(TR, FA, R1, S0):


    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def signal_genflash(TR, R1, S0, a, A):
    """Steady-state model of a spoiled gradient echo but
    parametrised with cos(FA) instead of FA and generalised to include rate.
    0<S0
    0<a
    -1<A<+1
    """
    E = np.exp(-a*TR*R1)
    return S0 * (1-E) / (1-A*E)

def signal_genflash_with_sat(TI, Tsat, TR, FA, R1, S0):
    """Flash pulse seq. with saturation pulse
    """
    T1 = 1/R1

    FA_rad = FA/360*(2*np.pi)
    M_afterSat = S0 * (1-np.exp(-Tsat*R1))
    T1_app = (T1*TR)/(TR-T1*np.log(np.cos(FA_rad)))
    M_apparent = S0 * (1-np.exp(-TR*R1))/(1-np.cos(FA_rad)*np.exp(-TR*R1))

    M = M_apparent * (1-np.exp(-(TI-Tsat)/T1_app)) + M_afterSat * np.exp(-(TI-Tsat)/T1_app)

    return M 

def signal_monoExp_aorta(TI, R1, S0):
    """Free Recovery
    """

    return S0*(1-np.exp(-TI*R1))


def signal_hyper(TR, R1, S0, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (e^(+ax) - e^(-bx)) / (e^(+ax) + e^(-bx))
    with x = TR*R1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(+a*x)
    Eb = np.exp(-b*x)
    return S0 * (Ea-Eb)/(Ea+Eb)

def signalBiExp(TR, R1, S0, A, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (1 - A e^(-ax) - (1-A) e^(-bx))
    with x = TR*R1
    0 < A < 1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(-a*x)
    Eb = np.exp(-b*x)
    return S0 * (1 - A*Ea - (1-A)*Eb)

def quadratic(x, x1, x2, x3, y1, y2, y3):
    """returns a quadratic function of x 
    that goes through the three points (xi, yi)"""

    a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
    a /= (x1-x2)*(x1-x3)*(x2-x3)
    b = (y2-y1)/(x2-x1) - a*(x1+x2)
    c = y1-a*x1**2-b*x1
    return a*x**2+b*x+c

def linear(x, x1, x2, y1, y2):
    """returns a linear function of x 
    that goes through the two points (xi, yi)"""

    b = (y2-y1)/(x2-x1)
    c = y1-b*x1
    return b*x+c

def concentrationSPGRESS(S, S0, T10, FA, TR, r1):
    """
    Calculates the tracer concentration from a spoiled gradient-echo signal.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        FA: Flip angle in degrees
        TR: Repetition time TR in msec (=time between two pulses)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    
    E = math.exp(-TR/T10)
    c = math.cos(FA*math.pi/180)
    Sn = (S/S0)*(1-E)/(1-c*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	#relaxation rate in 1/msec
    return (R1 - 1/T10)/r1

def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 
