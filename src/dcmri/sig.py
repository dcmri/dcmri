import math
from scipy.linalg import expm
import numpy as np


def signal_dsc(R1, R2, S0:float, TR, TE)->np.ndarray:
    """Signal model for a DSC scan with T2 and T2-weighting.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec. Must have the same size as R2.
        R2 (array-like): Transverse relaxation rate in 1/sec. Must have the same size as R1.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (array-like): Repetition time, or time between successive selective excitations, in sec. If TR is an array, it must have the same size as R1 and R2.
        TE (array-like): Echo time, in sec. If TE is an array, it must have the same size as R1 and R2.

    Returns:
        np.ndarray: Signal in arbitrary units, same length as R1 and R2.
    """
    return S0*np.exp(-np.multiply(TE,R2))*(1-np.exp(-np.multiply(TR,R1)))


def signal_t2w(R2, S0:float, TE)->np.ndarray:
    """Signal model for a DSC scan with T2-weighting.

    Args:
        R2 (array-like): Transverse relaxation rate in 1/sec. Must have the same size as R1.
        S0 (float): Signal scaling factor (arbitrary units).
        TE (array-like): Echo time, in sec. If TE is an array, it must have the same size as R1 and R2.

    Returns:
        np.ndarray: Signal in arbitrary units, same length as R1 and R2.
    """
    return S0*np.exp(-np.multiply(TE,R2))

def conc_t2w(S, TE:float, r2=0.5, n0=1)->np.ndarray:
    """Concentration for a DSC scan with T2-weighting.

    Args:
        S (array-like): Signal in arbitrary units.
        TE (float): Echo time in sec.
        r2 (float, optional): Transverse relaxivity in Hz/M. Defaults to 0.5.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.
    """
    # S/Sb = exp(-TE(R2-R2b))
    #   ln(S/Sb) = -TE(R2-R2b)
    #   R2-R2b = -ln(S/Sb)/TE
    # R2 = R2b + r2C
    #   C = (R2-R2b)/r2
    #   C = -ln(S/Sb)/TE/r2
    Sb = np.mean(S[:n0])
    C = -np.log(S/Sb)/TE/r2
    return C


def signal_ss(R1, S0, TR, FA)->np.ndarray:
    """Signal of a spoiled gradient echo sequence applied in steady state.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (array-like): Repetition time, or time between successive selective excitations, in sec. If TR is an array, it must have the same size as R1.
        FA (array-like): Flip angle in degrees.

    Returns:
        np.ndarray: Signal in arbitrary units.
    """
    E = np.exp(-np.multiply(TR,R1))
    cFA = np.cos(FA*np.pi/180)
    return S0 * (1-E) / (1-cFA*E)


def conc_ss(S, TR:float, FA:float, T10:float, r1=0.005, n0=1)->np.ndarray:
    """Concentration of a spoiled gradient echo sequence applied in steady state.

    Args:
        S (array-like): Signal in arbitrary units.
        TR (float): Repetition time, or time between successive selective excitations, in sec.
        FA (float): Flip angle in degrees.
        T10 (float): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.
    """
    Sb = np.mean(S[:n0])
    E = math.exp(-TR/T10)
    c = math.cos(FA*math.pi/180)
    Sn = (S/Sb)*(1-E)/(1-c*E)	        # normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	# relaxation rate in 1/msec
    return (R1 - 1/T10)/r1 


def signal_ss_fex(v, R1, S0:float, TR:float, FA:float):
    """Signal of a spoiled gradient echo sequence applied in steady state, for a multi-compartmental tissue wit fast water exchange.

    Args:
        v (array-like): Volume fractions of the compartments. v is a numpy array with size nc, where nc is the number of compartments.
        R1 (np.ndarray): Longitudinal relaxation rates of the compartments in 1/sec. R1 is a numpy array with dimensions (nc,) or (nc,nt), where nc is the number of compartments and nt is tne number of time points.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.

    Returns:
        np.ndarray: Signal in arbitrary units. If R1 is two-dimensional, this returns an array with a signal value for each time point.
    """
    if np.size(R1) == np.size(v):
        R1 = np.sum(np.multiply(v,R1))
        return signal_ss(R1, S0, TR, FA)
    nc, nt = R1.shape
    R1fex = np.zeros(nt)
    for c in range(nc):
        R1fex += v[c]*R1[c,:]
    return signal_ss(R1fex, S0, TR, FA)


def signal_ss_nex(v, R1:np.ndarray, S0:float, TR:float, FA:float):
    """Signal of a spoiled gradient echo sequence applied in steady state, for a multi-compartmental tissue without water exchange.

    Args:
        v (array-like): Volume fractions of the compartments. v is a numpy array with size nc, where nc is the number of compartments.
        R1 (np.ndarray): Longitudinal relaxation rates of the compartments in 1/sec. R1 is a numpy array with dimensions (nc,) or (nc,nt), where nc is the number of compartments and nt is tne number of time points.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.

    Returns:
        np.ndarray: Signal in arbitrary units. If R1 is two-dimensional, this returns an array with a signal value for each time point.
    """
    if np.size(R1) == np.size(v):
        S = signal_ss(R1, S0, TR, FA)
        return np.sum(np.multiply(v,S)) 
    nc, nt = R1.shape
    S = np.zeros(nt)
    for c in range(nc):
        S += v[c]*signal_ss(R1[c,:], S0, TR, FA)
    return S


def signal_ss_iex(PS, v, R1, S0, TR, FA):
    """Signal of a spoiled gradient echo sequence applied in steady state, for a multi-compartmental tissue with intermediate water exchange.

    Args:
        PS (np.ndarray): Water permeability-surface area product, in units of mL/sec/mL. PS is a numpy array with dimensions (nc, nc), where nc is the number of compartments. The off-diagnonal elements PS[j,i] represent the PS for the transfer from compartment i to j. The diagonal elements PS[i,i] quantify the rate of transfer from compartment i to the outside.
        v (array-like): Volume fractions of the compartments. v is a numpy array with size nc, where nc is the number of compartments.
        R1 (np.ndarray): Longitudinal relaxation rates of the compartments in 1/sec. R1 is a numpy array with dimensions (nc,) or (nc,nt), where nc is the number of compartments and nt is tne number of time points.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.

    Returns:
        np.ndarray: Signal in arbitrary units. If R1 is two-dimensional, this returns an array with a signal value for each time point.
    """

    # Mathematical notes on water exchange modelling
    # ---------------------------------------------
    # Longitudinal magnetization in a 2-compartment system
    # with kinetic transport and free relaxation

    # v1*dm1/dt = fi1*mi1 - fo1*m1 + f12*m2 - f21*m1 + R1_1*v1*(m01-m1) 
    # v2*dm2/dt = fi2*mi2 - fo2*m2 + f21*m1 - f12*m2 + R1_2*v2*(m02-m2) 

    # v1*dm1/dt = fi1*mi1 - (fo1+f21+R1_1*v1)*m1 + f12*m2 + R1_1*v1*m01
    # v2*dm2/dt = fi2*mi2 - (fo2+f12+R1_2*v2)*m2 + f21*m1 + R1_2*v2*m02

    # f1 = fo1 + f21 + R1_1*v1 > 0
    # f2 = fo2 + f12 + R1_2*v2 > 0

    # J1(t) = fi1*mi1(t) + R1_1*v1*m01 > 0
    # J2(t) = fi2*mi2(t) + R1_2*v2*m02 > 0

    # v1*dm1/dt = J1 - f1*m1 + f12*m2 
    # v2*dm2/dt = J2 - f2*m2 + f21*m1 

    # K1 = (fo1 + f21)/v1 + R1_1
    # K2 = (fo2 + f12)/v2 + R1_2
    # K12 = f12/v2
    # K21 = f21/v1

    # dM1/dt = J1 - K1*M1 + K12*M2 
    # dM2/dt = J2 - K2*M2 + K21*M1 

    # K = [[K1, -K12],[-K21, K2]
    # dM/dt = J - KM

    # Use generic solution for n-comp system to solve for M(t)
    # Note with R1(t) this is not a stationary system

    # Solution with K and J constant in time:

    # M(t) = exp(-tK)M(0) + exp(-tK)*J
    # M(t) = exp(-tK)M(0) + (1-exp(-tK)) K^-1 J

    # Check

    # dM/dt 
    # = - K exp(-tK)M(0) + exp(-tK) J
    # J - KM 
    # = J - Kexp(-tK)M(0) - (1-exp(-tK))J 
    # = - K exp(-tK)M(0) + exp(-tK) J

    # Spoiled gradient echo steady state:
    # M = exp(-TR*K) cosFA M + (1-exp(-TR*K)) K^-1 J
    # (1 - cosFA exp(-TR*K)) M = (1-exp(-TR*K)) K^-1 J
    # M = (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) K^-1 J
    # M = (1 - cosFA exp(-TR*K))^-1 K^-1 (1-exp(-TR*K))  J
    # M = [K(1 - cosFA exp(-TR*K))]^-1 (1-exp(-TR*K)) J

    # Inputs to the function:

    # f1 = fo1 + f21 + R1_1(t)*v1
    # f2 = fo2 + f12 + R1_2(t)*v2
    
    # J1 = fi1*S0*mi1(t) + R1_1(t)*v1*S0*m01
    # J2 = fi2*S0*mi2(t) + R1_2(t)*v2*S0*m02

    # K = [[f1/v1, -f21/v1], [-f12/v2, f2/v2]]
    # J = [J1, J2]

    # K dimensions = (ncomp, ncomp, nt)
    # J dimensions = (ncomp, nt)

    # Returns:

    # M = (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) K^-1 J
    # M = [(1 - cosFA exp(-TR*K))K]^-1 (1-exp(-TR*K)) J

    nc, nt = R1.shape
    J = np.empty((nc,nt))
    K = np.empty((nc,nc,nt))
    for c in range(nc):
        J[c,:] = S0*v[c]*R1[c,:]
        K[c,c,:] = R1[c,:] + np.sum(PS[:,c])/v[c]
        for d in range(nc):
            if d!=c:
                K[c,d,:] = -PS[c,d]/v[d]

    cFA = np.cos(FA*np.pi/180)
    Mag = np.empty(nt)
    Id = np.eye(nc)
    for t in range(nt):
        Et = expm(-TR*K[:,:,t])
        Mt = np.dot(K[:,:,t], Id-cFA*Et)
        Mt = np.linalg.inv(Mt)
        Vt = np.dot(Id-Et, J[:,t])
        Vt = np.dot(Mt, Vt)
        Mag[t] = np.sum(Vt)
    return Mag


def signal_sr(R1, S0:float, TR:float, FA:float, TC:float, TP=0.0)->np.ndarray:
    """Signal model for a saturation-recovery sequence with a FLASH readout.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.
        TP (float, optional): Time (sec) between the saturation pre-pulse and the first readout pulse. Defaults to 0.

    Raises:
        ValueError: If TP is larger than TC.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    if TP > TC:
        msg = 'Incorrect sequence parameters.'
        msg += 'Tsat must be smaller than TC.'
        raise ValueError(msg)
    cFA = np.cos(np.pi*FA/180)
    T1_app = TR/(TR*R1-np.log(cFA))

    ER = np.exp(-TR*R1)
    E_sat = np.exp(-TP*R1)
    E_center = np.exp(-(TC-TP)/T1_app)

    S_sat = S0 * (1-E_sat)
    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-E_center) + S_sat*E_center


def signal_sre(R1, S0:float, TR:float, FA:float, TC:float)->np.ndarray:
    """Signal model for a FLASH readout, starting from equilibrium (i.e. no preparation pulse).

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    #TI is the residence time in the slab
    cFA = np.cos(np.pi*FA/180)
    R1_app = R1 - np.log(cFA)/TR

    ER = np.exp(-TR*R1)
    EI = np.exp(-TC*R1_app)

    S_ss = S0 * (1-ER)/(1-cFA*ER)

    return S_ss*(1-EI) + S0*EI


def signal_src(R1, S0, TC):
    """Signal model for a saturation-recovery with a center-encoded readout.

    This can also be used with other encoding schemens whenever the effect of the readout pulses can be ignored, such as for fast flowing magnetization in arterial blood.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    E = np.exp(-TC*R1)
    return S0 * (1-E)

def conc_src(S, TC:float, T10:float, r1=0.005, n0=1)->np.ndarray:
    """Concentration of a saturation-recovery sequence with a center-encoded readout.

    Args:
        S (array-like): Signal in arbitrary units.
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.
        T10 (float): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.

    Example:

        We generate some signals from ground-truth concentrations, then reconstruct the concentrations and check against the ground truth:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        First define some constants:

        >>> T10 = 1         # sec
        >>> TC = 0.2        # sec
        >>> r1 = 0.005      # Hz/M

        Generate ground truth concentrations and signal data:

        >>> t = np.arange(0, 5*60, 0.1)     # sec
        >>> C = 0.003*(1-np.exp(-t/60))     # M
        >>> R1 = 1/T10 + r1*C               # Hz
        >>> S = dc.signal_src(R1, 100, TC)  # au

        Reconstruct the concentrations from the signal data:

        >>> Crec = dc.conc_src(S, TC, T10, r1)

        Check results by plotting ground truth against reconstruction:

        >>> plt.plot(t/60, 1000*C, 'ro', label='Ground truth')
        >>> plt.plot(t/60, 1000*Crec, 'b-', label='Reconstructed')
        >>> plt.title('SRC signal inverse')
        >>> plt.xlabel('Time (min)')
        >>> plt.ylabel('Concentration (mM)')
        >>> plt.legend()
        >>> plt.show()

    """
    # S = S0*(1-exp(-TC*R1))
    # S/Sb = (1-exp(-TC*R1))/(1-exp(-TC*R10))
    # (1-exp(-TC*R10))*S/Sb = 1-exp(-TC*R1)
    # 1-(1-exp(-TC*R10))*S/Sb = exp(-TC*R1)
    # ln(1-(1-exp(-TC*R10))*S/Sb) = -TC*R1
    # -ln(1-(1-exp(-TC*R10))*S/Sb)/TC = R1
    Sb = np.mean(S[:n0])
    E = math.exp(-TC/T10)
    R1 = -np.log(1-(1-E)*S/Sb)/TC	
    return (R1 - 1/T10)/r1 


def signal_lin(R1, S0:float)->np.ndarray:
    """Signal for any sequence operating in the linear regime.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    return S0 * R1


def conc_lin(S, T10, r1=0.005, n0=1):
    """Concentration for any sequence operating in the linear regime.

    Args:
        S (array-like): Signal in arbitrary units.
        T10 (float): baseline T1 value in sec.
        r1 (float, optional): Longitudinal relaxivity in Hz/M. Defaults to 0.005.
        n0 (int, optional): Baseline length. Defaults to 1.

    Returns:
        np.ndarray: Concentration in M, same length as S.
    """
    Sb = np.mean(S[:n0])
    R10 = 1/T10
    R1 = R10*S/Sb	#relaxation rate in 1/msec
    return 1000*(R1 - R10)/r1 


def sample(t, tp, Sp, dt=None)->np.ndarray: 
    """Sample a signal at given time points.

    Args:
        t (array-like): The time points at which to evaluate the signal.
        tp (array-like): the time points of the signal to be sampled.
        Sp (array-like): the values of the signal to be sampled. Values that are outside of the range are set to zero.
        dt (float, optional): sampling interval. If this is not provided, linear interpolation between the data points is used.  Defaults to None.

    Returns:
        np.ndarray: Signals sampled at times t.
    """
    if dt is None:#
        return np.interp(t, tp, Sp, left=0, right=0)
    Ss = np.zeros(len(t)) 
    for k, tk in enumerate(t):
        data = Sp[(tp >= tk-dt/2) & (tp < tk+dt/2)]
        if data.size > 0:
            Ss[k] = np.mean(data)
    return Ss 

def add_noise(signal, sdev:float)->np.ndarray:
    """Add noise to an MRI magnitude signal.

    Args:
        signal (array-like): Signal values.
        sdev (float): Standard deviation of the noise.

    Returns:
        np.ndarray: signal with noise added.
    """
    noise_x = np.random.normal(0, sdev, signal.size)
    noise_y = np.random.normal(0, sdev, signal.size)
    signal = np.sqrt((signal+noise_x)**2 + noise_y**2)
    return signal