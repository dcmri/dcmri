from scipy.linalg import expm

import numpy as np


def signal(sequence, R1, S0, TR=None, FA=None, TC=None):
    # Private for now - generalize to umbrella signal function
    if sequence == 'SRC':
        return signal_src(R1, S0, TC)
    if sequence == 'SR':
        return signal_sr(R1, S0, TR, FA, TC)
    elif sequence == 'SS':
        return signal_ss(R1, S0, TR, FA)
    else:
        raise ValueError(
            'Sequence ' + str(sequence) + ' is not recognised.')


def signal_dsc(R1, R2, S0: float, TR, TE) -> np.ndarray:
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
    return S0*np.exp(-np.multiply(TE, R2))*(1-np.exp(-np.multiply(TR, R1)))


def signal_t2w(R2, S0: float, TE) -> np.ndarray:
    """Signal model for a DSC scan with T2-weighting.

    Args:
        R2 (array-like): Transverse relaxation rate in 1/sec. Must have the same size as R1.
        S0 (float): Signal scaling factor (arbitrary units).
        TE (array-like): Echo time, in sec. If TE is an array, it must have the same size as R1 and R2.

    Returns:
        np.ndarray: Signal in arbitrary units, same length as R1 and R2.
    """
    return S0*np.exp(-np.multiply(TE, R2))


def conc_t2w(S, TE: float, r2=0.5, n0=1) -> np.ndarray:
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


def _signal_ss(R1, Sinf, TR, FA, MTT=None, ni=None) -> np.ndarray:

    # v*dm/dt = fi*mi - fo*m + R1*v*(m0-m)
    # v*dm/dt = (fi*mi + R1*v*m0) - (fo+R1*v)*m
    # dM/dt = (fi*mi + R1*v*m0) - (fo/v + R1)*M

    # K = R1 + f/v
    # J = f*mi + R1*v*m0
    # J = (f*ni + R1*v) * m0
    # J = (R1 + (f/v)*ni) * v m0
    # j = R1 + (f/v) * ni

    # dM/dt = v m0*j - KM

    # Solution with K constant in time:
    # M(t) = exp(-tK)M(0) + exp(-tK)*v*m0*j

    # If j also constant in time:
    # M(t) = exp(-tK)M(0) + (1-exp(-tK)) K^-1 v m0*j

    # Spoiled gradient echo steady state:
    # M = exp(-TR*K) cosFA M + (1-exp(-TR*K)) K^-1 v m0 j
    # (1 - cosFA exp(-TR*K)) M = (1-exp(-TR*K)) K^-1 v m0 j
    # M = v m0 (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) j/K

    # S = I * sinFA * M
    # S = I*v*m0 * sinFA * (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) j/K

    # ni must have the same shape as R1

    if MTT is None:
        K = R1
        jK = 1
    else:
        K = R1 + 1/MTT
        jK = (R1 + ni/MTT) / K
    FA = FA*np.pi/180
    E = np.exp(-np.multiply(TR, K))
    cFA = np.cos(FA)
    n = (1-E) / (1-cFA*E)
    return Sinf * np.sin(FA) * n * jK


def _signal_ss_nex(v, R1: np.ndarray, S0, TR: float, FA: float, 
                   fo=None, Ji=None, sum=True):
    if np.size(R1) == np.size(v):
        if Ji is None:
            S = [v[c]*_signal_ss(R1[c], S0, TR, FA) 
                for c in range(np.size(v))]
        else:
            S = []
            for c in range(np.size(v)):
                if fo[c]==0:
                    Sc = v[c]*_signal_ss(R1[c], S0, TR, FA)
                else:
                    MTT = v[c]/fo[c]
                    ni = Ji[c]/fo[c] 
                    Sc = v[c]*_signal_ss(R1[c], S0, TR, FA, MTT, ni) 
                S.append(Sc)
        if sum:
            return np.sum(S)
        else:
            return S
    else:
        S = np.zeros(R1.shape)
        for c in range(R1.shape[0]):
            if Ji is None:
                S[c, ...] = v[c]*_signal_ss(
                    R1[c, ...], S0, TR, FA)
            else:
                if fo[c]==0:
                    S[c, ...] = v[c]*_signal_ss(
                        R1[c, ...], S0, TR, FA)
                else:
                    MTT = v[c]/fo[c]
                    ni = Ji[c, :]/fo[c] 
                    S[c, ...] = v[c]*_signal_ss(
                        R1[c, ...], S0, TR, FA, MTT, ni)
        if sum:
            return np.sum(S, axis=0)
        else:
            return S


def _signal_ss_fex(v, R1, S0, TR: float, FA: float, 
                   fo=None, Ji=None, sum=True):
    fo = np.sum(fo)
    if np.size(R1) == np.size(v):
        R1fex = np.sum(np.multiply(v, R1))
        if Ji is None:
            S = _signal_ss(R1fex, S0, TR, FA)
        else:
            MTT = np.sum(v)/fo
            ni = np.sum(Ji)/fo
            S = _signal_ss(R1fex, S0, TR, FA, MTT, ni)
        if sum:
            return S
        else:
            return S.reshape((1, len(S)))
    else:
        nc = R1.shape[0]
        R1fex = np.zeros(R1.shape[1:])
        for c in range(nc):
            R1fex += v[c] * R1[c, ...]
        R1fex /= np.sum(v)
        if Ji is None:
            S = _signal_ss(R1fex, S0, TR, FA)
        else:
            MTT = np.sum(v)/fo
            ni = np.sum(Ji, axis=0)/fo
            S = _signal_ss(R1fex, S0, TR, FA, MTT, ni)
        if sum:
            return S
        else:
            return S.reshape((1,) + S.shape)


def _signal_ss_aex(v, R1, S0, TR, FA, f=None, Ji=None, sum=True):

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
    # J = [J1, J2]

    # dM/dt = J - KM

    # Assume: m01 = m02

    # J1(t) = (fi1*ni1(t) + R1_1*v1) * m0
    # J2(t) = (fi2*ni2(t) + R1_2*v2) * m0

    # J1(t) = (Fi1(t) + R1_1 * v1) * m0
    # J2(t) = (Fi2(t) + R1_2 * v2) * m0   

    # J = m0 * [j1, j2]^T

    # dM/dt = m0*j - KM

    # Use generic solution for n-comp system to solve for M(t)
    # Note with R1(t) this is not a stationary system

    # Solution with K constant in time:
    # M(t) = exp(-tK)M(0) + exp(-tK)*m0*j

    # If j also constant in time:
    # M(t) = exp(-tK)M(0) + (1-exp(-tK)) K^-1 m0*j

    # Spoiled gradient echo steady state:
    # M = exp(-TR*K) cosFA M + (1-exp(-TR*K)) K^-1 m0 j
    # (1 - cosFA exp(-TR*K)) M = (1-exp(-TR*K)) K^-1 m0 j
    # M = m0 * (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) K^-1 j
    # M = m0 * (1 - cosFA exp(-TR*K))^-1 K^-1 (1-exp(-TR*K)) j
    # M = m0 * [K(1 - cosFA exp(-TR*K))]^-1 (1-exp(-TR*K)) j

    # S = I * sinFA * M
    # S = I*m0 * sinFA * [K(1 - cosFA exp(-TR*K))]^-1 (1-exp(-TR*K)) j

    # Inputs to the function:
    # K dimensions = (ncomp, ncomp, nt)
    # J dimensions = (ncomp, nt)

    # reshape for convenience
    n = np.shape(R1)
    R1 = np.reshape(R1, (n[0], -1))
    f = np.array(f)

    nc, nt = R1.shape
    j = np.zeros((nc, nt))
    K = np.zeros((nc, nc, nt))
    F = np.sum(f, axis=0)

    # TODO handle the case where F of one or more compartments is infinite
    for c in range(nc):

        # Build K-matrix
        K[c, c, :] = R1[c, :] + F[c]/v[c]
        for d in range(nc):
            if d != c:
                K[c, d, :] = -f[c, d]/v[d] 

        # Build j-array
        j[c, :] = v[c] * R1[c, :]
        if Ji is not None:
            j[c, :] += Ji[c, :]

    FA = FA*np.pi/180
    cFA = np.cos(FA)
    Id = np.eye(nc)
    if sum:
        Mag = np.empty(nt)
    else:
        Mag = np.empty((nc, nt))
    for t in range(nt):
        Et = expm(-TR*K[:, :, t])
        Magt = np.dot(Id-Et, j[:, t])
        Mt = np.dot(K[:, :, t], Id-cFA*Et)
        Mtinv = np.linalg.inv(Mt)
        Magt = np.dot(Mtinv, Magt)
        if sum:
            Mag[t] = np.sum(Magt)
        else:
            Mag[:, t] = Magt

    # Return in original shape
    # R1 = R1.reshape(n)  
    if sum:
        Mag = Mag.reshape(n[1:])
    else:
        Mag = Mag.reshape(n)
    return S0*np.sin(FA)*Mag

# TODO API change: rename PSw to Fw??
def signal_ss(R1, S0, TR, FA, v=1, PSw=None, Ji=None, R10=None, 
              sum=True) -> np.ndarray:
    """Signal of a spoiled gradient echo sequence applied in steady state.

    Args:
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a 
          tissue with a single compartment, R1 can have any shape.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective 
          excitations, in sec. 
        FA (float): Flip angle in degrees.
        v (array-like, optional): volume fractions of each compartment. 
          v=1 for a 1-compartment tissue. For a tissue with multiple 
        compartments, the length of v must be same as the first dimension of 
          R1 and values must add up to 1. Defaults to 1.
        PSw (array-like, optional): Water permeability-surface area products 
          through the interfaces between the compartments, in units of 
          mL/sec/mL. With PSw=np.inf (default), water exchange is in the 
          fast-exchange limit. With PSw=0, there is no water exchange between 
          the compartments. For any intermediate level of water exchange, 
          PSw must be a nxn array, where n is the number of compartments, 
          and PSw[j,i] is the permeability for water moving from compartment 
          i into j. The diagonal elements PSw[i,i] quantify the flow of water 
          from compartment i to outside. Defaults to np.inf.
        R10 (float, optional): R1-value where S0 is defined. If not provided, 
          S0 is the scaling factor corresponding to infinite R10. Defaults 
          to None.
        sum (bool, optional): If set to True, this returns an array with the 
          total signal. Otherwise an array is returned with the signal of 
          each compartment separately. In that case the first dimension is 
          the number of compartments. Defaults to True.

    Returns:
        np.ndarray: Signal in the same units as S0, e
    """
    if R10 is None:
        Sinf = S0
    else:
        Sinf = S0/signal_ss(R10, 1, TR, FA, v=v, PSw=PSw, Ji=Ji)

    # One compartment
    if np.isscalar(v): 
        if Ji is None:
            sig = _signal_ss(R1, Sinf, TR, FA)
        else:
            fo = PSw # one-comp: PSw = outflow = inflow
            sig = _signal_ss(R1, Sinf, TR, FA, v/fo, Ji/fo)
        if sum:
            return sig
        else:
            return sig.reshape((1,) + sig.shape)
        
    # Multiple compartments
    if np.isscalar(R1):
        raise ValueError(
            'In a multi-compartment system, R1 must be an array ' + 
            'with at least 1 dimension..')
    if np.ndim(R1) == 1:
        if np.size(v) != np.size(R1):
            raise ValueError('v must have the same length as R1.')
    elif np.size(v) != np.shape(R1)[0]:
        raise ValueError(
            'v must have the same length as the first dimension of R1.')
    
    # No inflow, constant PSw
    if np.isscalar(PSw): 
        if PSw == np.inf: # FWX
            return _signal_ss_fex(v, R1, Sinf, TR, FA, sum=sum)
        elif PSw == 0: # NWX
            return _signal_ss_nex(v, R1, Sinf, TR, FA, sum=sum)
        else:
            nc = len(v)
            PSw = np.full((nc, nc), PSw) - np.diag(np.full(nc, PSw))
            return signal_ss(R1, Sinf, TR, FA, v=v, PSw=PSw, sum=sum)

    # variable PSw    
    if np.ndim(PSw) != 2:
        raise ValueError(
            "For intermediate water exchange, PSw must be a square array")
    if np.shape(PSw)[0] != np.size(v):
        raise ValueError("Dimensions of PSw and v do not match up.")
    nc = np.size(v)
    PSexch = PSw - np.diag(np.diag(PSw))
    nex = (np.linalg.norm(PSexch)==0)
    ninf = np.count_nonzero(np.isinf(PSexch))
    fex = (ninf==nc*nc-nc)
    if nex:
        if Ji is None:
            return _signal_ss_nex(
                v, R1, Sinf, TR, FA, sum=sum)
        else:
            fo = np.diag(PSw)
            return _signal_ss_nex(
                v, R1, Sinf, TR, FA, fo=fo, Ji=Ji, sum=sum)
    elif fex:
        if Ji is None:
            return _signal_ss_fex(
                v, R1, Sinf, TR, FA, sum=sum)
        else:
            fo = np.diag(PSw)
            return _signal_ss_fex(
                v, R1, Sinf, TR, FA, fo=fo, Ji=Ji, sum=sum)
    elif 0 < ninf < nc*nc-nc:
        # TODO: add this case
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite flow values '+
            'are currently not implemented.')
    else:
        if Ji is None:
            return _signal_ss_aex(
                v, R1, Sinf, TR, FA, f=PSexch, sum=sum)
        else:
            return _signal_ss_aex(
                v, R1, Sinf, TR, FA, f=PSw, Ji=Ji, sum=sum)


def conc_ss(S, TR: float, FA: float, T10: float, r1=0.005, n0=1) -> np.ndarray:
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
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R10)) / (1-cFA*exp(-TR*R10))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn
    Sb = np.mean(S[:n0])
    E0 = np.exp(-TR/T10)
    c = np.cos(FA*np.pi/180)
    Sn = (S/Sb)*(1-E0)/(1-c*E0)	        # normalized signal
    # Replace any Nan values by interpolating between nearest neighbours
    outrange = Sn >= 1
    if np.sum(outrange) > 0:
        inrange = Sn < 1
        x = np.arange(Sn.size)
        Sn[outrange] = np.interp(x[outrange], x[inrange], Sn[inrange])
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR  # relaxation rate in 1/msec
    return (R1 - 1/T10)/r1


def _signal_sr(R1, Sinf: float, TR: float, FA: float, TC: float, TP=0.0) -> np.ndarray:

    if TP > TC:
        msg = 'Incorrect sequence parameters.'
        msg += 'Tsat must be smaller than TC.'
        raise ValueError(msg)

    FA = np.pi*FA/180
    cFA = np.cos(FA)
    T1_app = TR/(np.multiply(TR, R1)-np.log(cFA))

    ER = np.exp(-np.multiply(TR, R1))
    E_sat = np.exp(-np.multiply(TP, R1))
    E_center = np.exp(-(TC-TP)/T1_app)

    S_sat = Sinf * (1-E_sat)
    S_ss = Sinf * (1-ER)/(1-cFA*ER)

    S_ss = S_ss*(1-E_center) + S_sat*E_center

    return np.sin(FA)*S_ss


def _signal_sr_fex(v, R1, S0: float, TR: float, FA: float, TC: float, TP=0.0):
    if np.size(R1) == np.size(v):
        R1 = np.sum(np.multiply(v, R1))
        return _signal_sr(R1, S0, TR, FA, TC, TP)
    nc, nt = R1.shape
    R1fex = np.zeros(nt)
    for c in range(nc):
        R1fex += v[c]*R1[c, :]
    return _signal_sr(R1fex, S0, TR, FA, TC, TP)


def _signal_sr_nex(v, R1, S0: float, TR: float, FA: float, TC: float, TP=0.0):
    if np.size(R1) == np.size(v):
        S = _signal_sr(R1, S0, TR, FA, TC, TP)
        return np.sum(np.multiply(v, S))
    nc, nt = R1.shape
    S = np.zeros(nt)
    for c in range(nc):
        S += v[c]*_signal_sr(R1[c, :], S0, TR, FA, TC, TP)
    return S


def signal_sr(R1, S0: float, TR: float, FA: float, TC: float, TP=0.0, v=1, PSw=np.inf, R10=None) -> np.ndarray:
    """Signal model for a saturation-recovery sequence with a FLASH readout.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TR (float): Repetition time, or time between successive selective excitations, in sec. 
        FA (array-like): Flip angle in degrees.
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.
        TP (float, optional): Time (sec) between the saturation pre-pulse and the first readout pulse. Defaults to 0.
        v (array-like, optional): volume fractions of each compartment. v=1 for a 1-compartment tissue. For a tissue with multiple compartments, the length of v must be same as the first dimension of R1 and values must add up to 1. Defaults to 1.
        PSw (array-like, optional): Water permeability-surface area products through the interfaces between the compartments, in units of mL/sec/mL. With PSw=np.inf (default), water exchange is in the fast-exchange limit. With PSw=0, there is no water exchange between the compartments. For any intermediate level of water exchange, PSw must be a nxn array, where n is the number of compartments, and PSw[j,i] is the permeability for water moving from compartment i into j. The diagonal elements PSw[i,i] quantify the flow of water from compartment i to outside. Defaults to np.inf.
        R10 (float, optional): R1-value where S0 is defined. If not provided, S0 is the scaling factor corresponding to infinite R10. Defaults to None.

    Raises:
        ValueError: If TP is larger than TC.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    if R10 is None:
        Sinf = S0
    else:
        if v != 1:
            R10 = np.full(len(v), R10)
        Sinf = S0/signal_sr(R10, 1, TR, FA, TC, TP, v=v)
    if v == 1:
        return _signal_sr(R1, Sinf, TR, FA, TC, TP)
    if np.isscalar(R1):
        raise ValueError(
            'In a multi-compartment system, R1 must be an array with at least 1 dimension..')
    if np.ndim(R1) == 1:
        if np.size(v) != np.size(R1):
            raise ValueError('v must have the same length as R1.')
    if np.size(v) != np.shape(R1)[0]:
        raise ValueError(
            'v must have the same length as the first dimension of R1.')
    if np.isscalar(PSw):
        if PSw == np.inf:
            return _signal_sr_fex(v, R1, Sinf, TR, FA, TC, TP)
        elif PSw == 0:
            return _signal_sr_nex(v, R1, Sinf, TR, FA, TC, TP)
    else:
        if np.ndim(PSw) != 2:
            raise ValueError(
                "For intermediate water exchange, PSw must be a square array")
        if np.shape(PSw)[0] != np.size(v):
            raise ValueError("Dimensions of PSw and v do not match up.")
        raise NotImplementedError(
            'Internediate water exchange is not yet available for SR sequences')
        # return _signal_sr_aex(PSw, v, R1, Sinf, TR, FA, TC, TP)


def signal_er(R1, S0: float, TR: float, FA: float, TC: float) -> np.ndarray:
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
    # TI is the residence time in the slab
    FA = np.pi*FA/180
    cFA = np.cos(FA)
    R1_app = R1 - np.log(cFA)/TR

    ER = np.exp(-TR*R1)
    EI = np.exp(-TC*R1_app)

    Sss = S0 * (1-ER)/(1-cFA*ER)
    Sss = Sss*(1-EI) + S0*EI

    return np.sin(FA)*Sss


def signal_src(R1, S0, TC, R10=None):
    """Signal model for a saturation-recovery with a center-encoded readout.

    This can also be used with other encoding schemens whenever the effect of the readout pulses can be ignored, such as for fast flowing magnetization in arterial blood.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).
        TC (float): Time (sec) between the saturation pulse and the acquisition of the k-space center.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    if R10 is None:
        Sinf = S0
    else:
        Sinf = S0/signal_src(R10, 1, TC)
    E = np.exp(-TC*R1)
    return Sinf * (1-E)


def conc_src(S, TC: float, T10: float, r1=0.005, n0=1) -> np.ndarray:
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
    E = np.exp(-TC/T10)
    R1 = -np.log(1-(1-E)*S/Sb)/TC
    return (R1 - 1/T10)/r1


def signal_lin(R1, S0: float) -> np.ndarray:
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
    R1 = R10*S/Sb  # relaxation rate in 1/msec
    return (R1 - R10)/r1
