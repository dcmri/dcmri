from scipy.linalg import expm

import numpy as np







# TODO handle the case where F of one or more compartments is infinite
def _Mz_K(R1, v, Fw):
    if np.isscalar(R1):
        return R1 + Fw/v
    nc = np.size(R1)
    K = np.zeros((nc, nc))
    if np.isscalar(Fw):
        Fw = np.full((nc, nc), Fw) - np.diag(np.full(nc, Fw))  
    elif isinstance(Fw, list):
        Fw = np.array(Fw)
    F = np.sum(Fw, axis=0)
    for c in range(nc):
        K[c, c] = R1[c] + F[c]/v[c]
        for d in range(nc):
            if d != c:
                K[c, d] = -Fw[c, d]/v[d]
    return K


def _Mz_J(R1, v, me, j=None):
    J = np.multiply(np.multiply(R1, v), me)
    J = J if j is None else J + np.multiply(j, me)
    return J


def Mz_free(R1, T, v=1, Fw=0, j=None, n0=0, me=1):
    """Free longitudinal magnetization.

    See section :ref:`basics-relaxation-T1` for more detail.

    Args:
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a
          single compartment, R1 can be scalar or a 1D time-array.
        T (array-like): duration of free recovery.
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        n0 (array-like, optional): initial relative magnetization at T=0. 
          If this is a scalar, all compartments are assumed to have the same 
          initial magnetization. Defaults to 0.
        me (array-like, optional): equilibrium magnetization of the tissue 
          compartments. If a scalar value is provided, all compartments are 
          assumed to have the same equilibrium magnetization. Defaults to 1.

    Returns:
        np.ndarray: Magnetization in the compartments after a time T.

    Example:

        Magnetization recovery after inversion.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Plot magnetization recovery for the first 10 seconds after an 
        inversion pulse, for a closed tissue with R1 = 1 sec, and for an open 
        tissue with equilibrium inflow and inverted inflow:

        >>> TI = 0.1*np.arange(100)
        >>> R1 = 1
        >>> f = 0.5

        >>> Mz = dc.Mz_free(R1, TI, n0=-1)
        >>> Mz_e = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=f)
        >>> Mz_i = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=-f)

        >>> plt.plot(TI, Mz, label='No flow', linewidth=3)
        >>> plt.plot(TI, Mz_e, label='Equilibrium inflow', linewidth=3)
        >>> plt.plot(TI, Mz_i, label='Inverted inflow', linewidth=3)
        >>> plt.xlabel('Inversion time (sec)')
        >>> plt.ylabel('Magnetization (A/cm)')
        >>> plt.legend()
        >>> plt.show()

        Now consider a two-compartment model, with a central compartment 
        that has in- and outflow, and a peripheral compartment that only 
        exchanges with the central compartment:

        >>> R1 = [1,2]
        >>> v = [0.3, 0.7]
        >>> PS = 0.1
        >>> Fw = [[f, PS], [PS, 0]]
        >>> Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=[f, 0])

        >>> plt.plot(TI, Mz[0,:], label='Central compartment', linewidth=3)
        >>> plt.plot(TI, Mz[1,:], label='Peripheral compartment', linewidth=3)
        >>> plt.xlabel('Inversion time (sec)')
        >>> plt.ylabel('Magnetization (A/cm)')
        >>> plt.legend()
        >>> plt.show()

        In DC-MRI the more usual situation is one where TI is fixed and the 
        relaxation rates are variable due to the effect of a contrast agent. 
        As an illustration, consider the previous result again at TI=500 msec 
        and an R1 that is linearly declining in the central compartment and 
        constant in the peripheral compartment:

        >>> TI = 0.5
        >>> nt = 1000
        >>> t = 0.1*np.arange(nt)
        >>> R1 = np.stack((1-t/np.amax(t), np.ones(nt)))
        >>> j = np.stack((f*np.ones(nt), np.zeros(nt)))
        >>> Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)

        >>> plt.plot(t, Mz[0,:], label='Central compartment', linewidth=3)
        >>> plt.plot(t, Mz[1,:], label='Peripheral compartment', linewidth=3)
        >>> plt.xlabel('Time (sec)')
        >>> plt.ylabel('Magnetization (A/cm)')
        >>> plt.legend()
        >>> plt.show()   

        The function allows for R1 and TI to be both variable. Computing the 
        result for 10 different TI values and extracting the result 
        corresponding to TI=0.5 gives again the same result:

        >>> TI = 0.1*np.arange(10)
        >>> Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)

        >>> plt.plot(t, Mz[0,:,5], label='Central compartment', linewidth=3)
        >>> plt.plot(t, Mz[1,:,5], label='Peripheral compartment', linewidth=3)
        >>> plt.xlabel('Time (sec)')
        >>> plt.ylabel('Magnetization (A/cm)')
        >>> plt.legend()
        >>> plt.show()      

    """
    if not np.isscalar(T):
        if np.isscalar(R1):
            Mz = np.empty(np.size(T))
            for k, Tk in enumerate(T):
                Mz[k] = Mz_free(R1, Tk, v, Fw, j, n0, me)
        else:
            Mz = np.empty(np.shape(R1) + (np.size(T), ))
            for k, Tk in enumerate(T):
                Mz[...,k] = Mz_free(R1, Tk, v, Fw, j, n0, me)
        return Mz

    if np.isscalar(R1):
        K = _Mz_K(R1, v, Fw)
        J = _Mz_J(R1, v, me, j=j)
        E = np.exp(-T*K)
        Kinv = 1/K
        M0 = n0*me*v
        return E*M0 + (1-E)*Kinv*J

    elif np.ndim(R1)==1:

        if np.isscalar(v):
            nt = np.size(R1)
            M = np.empty(nt)
            for t in range(nt):
                jt = None if j is None else j[t]
                n0t = n0 if np.isscalar(n0) else n0[t]
                M[t] = Mz_free(R1[t], T, v=v, Fw=Fw, j=jt, n0=n0t, me=me)
            return M
        
        K = _Mz_K(R1, v, Fw)
        J = _Mz_J(R1, v, me, j=j)
        E = expm(-T*K)
        Kinv = np.linalg.inv(K)
        I = np.eye(np.size(R1))
        M0 = np.multiply(np.multiply(n0, me), v)
        return np.dot(E, M0) + np.dot(I-E, np.dot(Kinv, J))

    if np.isscalar(v):
        raise ValueError(
            'In a tissue with n compartments, v must be an n-element array.')
    n = np.shape(R1)
    R1 = np.reshape(R1, (n[0], -1))
    if j is not None:
        j = np.reshape(j, (n[0], -1))
    M = np.empty(R1.shape)
    for t in range(R1.shape[1]):
        jt = None if j is None else j[:,t]
        if np.ndim(n0)==2:
            n0t = n0[:,t]
        else:
            n0t = n0
        M[:,t] = Mz_free(R1[:,t], T, v=v, Fw=Fw, j=jt, n0=n0t, me=me) 
    return M.reshape(n)  


def Mz_ss(R1, TR, FA, v=1, Fw=0, j=None, me=1) -> np.ndarray:
    """Steady-state longitudinal tissue magnetization.

    See section :ref:`basics-relaxation-T1` for more detail.

    Args:
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a
          single compartment, R1 can be scalar or a 1D time-array.
        TR (float): repetition time.
        FA (float): flip angle.
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        me (array-like, optional): equilibrium magnetization of the tissue 
          compartments. If a scalar value is provided, all compartments are 
          assumed to have the same equilibrium magnetization. Defaults to 1.

    Returns:
        np.ndarray: Magnetization in the compartments after a time T.

    Example:

        Compute steady-state magnetization with inflow magnetization (mi) 
        ranging from fully inverted to equilibrium. Compare against 
        magnetization of an isolated tissue without flow.

    .. plot::
        :include-source:
        :context: close-figs

        Import packages:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Define constants:

        >>> FA, TR = 12, 0.005
        >>> R1 = 1
        >>> f, v = 0.5, 0.7
        >>> mi = np.linspace(-1, 1, 100)

        Compute magnetization without/with inflow:

        >>> m_c = dc.Mz_ss(R1, TR, FA, v)/v
        >>> m_f = [dc.Mz_ss(R1, TR, FA, v, f, j=f*m)/v for m in mi]

        Plot the results:

        >>> plt.plot(mi, mi*0+m_c, label='No inflow', linewidth=3)
        >>> plt.plot(mi, m_f, label='Inflow', linewidth=3)
        >>> plt.xlabel('Inflow magnetization (A/cm/mL)')
        >>> plt.ylabel('Steady-state magnetization (A/cm/mL)')
        >>> plt.legend()
        >>> plt.show()

        Note the magnetization of the two tissues is the same when the 
        inflow is at the steady-state of the isolated tissue: 

        >>> m_f = dc.Mz_ss(R1, TR, FA, v, f, j=f*m_c)/v
        >>> print(m_f - m_c)
        0.00023950879616352339

        Now we consider the same situation again, this time for a two-
        compartment tissue with one central compartment that exchanges with 
        the enviroment. 

        >>> R1 = [0.5, 1.5]
        >>> v = [0.3, 0.6]
        >>> PS = 0.1
        
        Compute magnetization without inflow:

        >>> Fw = [[0, PS], [PS, 0]]
        >>> M_c = dc.Mz_ss(R1, TR, FA, v, Fw)

        Compute magnetization with flow through the first compartment:

        >>> Fw = [[f, PS], [PS, 0]]
        >>> M_f = np.zeros((2, 100))
        >>> for i, m in enumerate(mi):
        >>>     M_f[:,i] = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f*m, 0])

        Plot the results for the central compartment:

        >>> plt.plot(mi, M_c[0]/v[0] + mi*0, label='No inflow', linewidth=3)
        >>> plt.plot(mi, M_f[0,:]/v[0], label='Inflow', linewidth=3)
        >>> plt.xlabel('Inflow magnetization (A/cm/mL)')
        >>> plt.ylabel('Steady-state magnetization (A/cm/mL)')
        >>> plt.legend()
        >>> plt.show()

        We can verify again that the magnetization is the same as that of the 
        isolated tissue when the inflow is at the isolated steady-state: 

        >>> m_c = M_c[0]/v[0]
        >>> M_f = dc.Mz_ss(R1, TR, FA, v, Fw=f, j=f*m_c)
        >>> print(M_f[0]/v[0] - m_c) 
        0.0002984954839493209      
     
    """
    # One compartment
    if np.isscalar(R1): 
        return me*_Nz_ss_1c(R1, TR, FA, v, Fw, j)
    
    elif np.ndim(R1)==1:
        if np.isscalar(v):
            nt = np.size(R1)
            M = np.empty(nt)
            for t in range(nt):
                jt = None if j is None else j[t]
                M[t] = Mz_ss(R1[t], TR, FA, v, Fw, jt, me)
            return M
        return me*_Nz_ss(R1, TR, FA, v, Fw, j)

    if np.isscalar(v):
        raise ValueError(
            'In a tissue with n compartments, v must be an n-element array.')
    n = np.shape(R1)
    R1 = np.reshape(R1, (n[0], -1))
    if j is not None:
        j = np.reshape(j, (n[0], -1))
    M = np.empty(R1.shape)
    for t in range(R1.shape[1]):
        jt = None if j is None else j[:,t]
        M[:,t] = Mz_ss(R1[:,t], TR, FA, v, Fw, jt, me) 
    return M.reshape(n)   


def _Nz_ss_1c(R1, TR, FA, v, Fw=0, j=None):
    K = R1 + Fw/v
    J = R1*v if j is None else R1*v + j
    E = np.exp(-np.multiply(TR, K))
    cFA = np.cos(FA*np.pi/180)
    n = (1-E) / (1-cFA*E)
    if K==0:
        return n
    else:
        return n * J/K


def _Nz_ss(R1, TR, FA, v, Fw=0, j=None):   
    nc = np.size(R1)

    # Closed tissue with constant permeabilities
    if np.isscalar(Fw): 
        if Fw == np.inf: 
            return _Nz_ss_fex(R1, TR, FA, v)
        elif Fw == 0: 
            return _Nz_ss_nex(R1, TR, FA, v)
        else:
            Fw = np.full((nc, nc), Fw) - np.diag(np.full(nc, Fw))
            return _Nz_ss_aex(R1, TR, FA, v, Fw, j)
        
    # Open tissue
    if np.shape(Fw) != (nc, nc):
        raise ValueError('Fw must be a square array with size equal to the '
                         'number of compartments.')
    PSw = np.array(Fw) - np.diag(np.diag(Fw))
    ninf = np.count_nonzero(np.isinf(PSw))
    if np.linalg.norm(PSw) == 0:
        return _Nz_ss_nex(R1, TR, FA, v, Fw, j)
    elif ninf == nc*nc-nc:
        return _Nz_ss_fex(R1, TR, FA, v, Fw, j)
    elif 0 < ninf < nc*nc-nc:
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Nz_ss_aex(R1, TR, FA, v, Fw, j)


def _Nz_ss_fex(R1, TR, FA, v, Fw=0, j=None):
    R1 = np.sum(np.multiply(v, R1)) / np.sum(v)
    if j is None:
        N = _Nz_ss_1c(R1, TR, FA, np.sum(v))
    else:
        fo = np.diag(Fw)
        N = _Nz_ss_1c(R1, TR, FA, np.sum(v), np.sum(fo), np.sum(j))
    return np.stack([N*vc/np.sum(v) for vc in v])

def _Nz_ss_nex(R1, TR, FA, v, Fw=0, j=None):
    if j is None:
        N = [_Nz_ss_1c(R1[c], TR, FA, v[c]) for c in range(np.size(v))]
    else:
        N = []
        fo = np.diag(Fw)
        for c in range(np.size(v)):
            Nc = _Nz_ss_1c(R1[c], TR, FA, v[c], fo[c], j[c]) 
            N.append(Nc)
    return np.stack(N)
      
def _Nz_ss_aex(R1, TR, FA, v, Fw=0, j=None):
    K = _Mz_K(R1, v, Fw)
    J = _Mz_J(R1, v, 1, j=j)
    E = expm(-TR*K)
    I = np.eye(np.size(R1))
    cFA = np.cos(FA*np.pi/180)
    A = np.dot(K, I-cFA*E)
    Ainv = np.linalg.inv(A)
    return np.dot(Ainv, np.dot(I-E, J))        




def Mz_spgr(R1, T, TR, FA, v=1, Fw=0, j=None, n0=0, me=1):
    """Longitudinal tissue magnetization of a spoiled gradient-echo sequence.

    See section :ref:`basics-relaxation-T1` for more detail.

    Args:
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a
          single compartment, R1 can be scalar or a 1D time-array.
        T (float): time since the first rf-pulse.
        TR (float): repetition time between rf-pulses.
        FA (float): flip angle of the rf-pulse.
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        n0 (array-like, optional): initial relative magnetization at T=0. 
          If this is a scalar, all compartments are assumed to have the same 
          initial magnetization. Defaults to 0.
        me (array-like, optional): equilibrium magnetization of the tissue 
          compartments. If a scalar value is provided, all compartments are 
          assumed to have the same equilibrium magnetization. Defaults to 1.

    Returns:
        np.ndarray: Tissue magnetization in the compartments after a time T.

    Example:

        Compare SPGR tissue magnetization of a two-compartment tissue
        after inversion against free recovery and steady-state:

    .. plot::
        :include-source:
        :context: close-figs

        Import packages:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Define constants:

        >>> FA, TR = 12, 0.005
        >>> TI = np.linspace(0,3,100)
        >>> R1 = [1, 0.5]
        >>> v = [0.3, 0.7]
        >>> f, PS = 0.5, 0.1
        >>> Fw = [[f, PS], [PS, 0]]

        Compute magnetization:

        >>> Mspgr = dc.Mz_spgr(R1, TI, TR, FA, v, Fw, j=[f, 0], n0=-1) 
        >>> Mfree = dc.Mz_free(R1, TI, v, Fw, j=[f, 0], n0=-1)
        >>> Mss = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f, 0])

        Plot the results for the peripheral compartment:

        >>> c = 1
        >>> plt.title('Peripheral compartment')
        >>> plt.plot(TI, Mfree[c,:], label='Free', linewidth=3)
        >>> plt.plot(TI, v[c]+0*TI, label='Equilibrium', linewidth=3)
        >>> plt.plot(TI, Mspgr[c,:], label='SPGR', linewidth=3)
        >>> plt.plot(TI, Mss[c]+0*TI, label='Steady-state', linewidth=3)
        >>> plt.xlabel('Time since start of pulse sequence (sec)')
        >>> plt.ylabel('Tissue magnetization (A/cm/cm3)')
        >>> plt.legend()
        >>> plt.show()

        This verifies that the free recovery magnetization goes to equilibrium, 
        and that the SPGR magnetization relaxes to the steady state at a 
        shorter time than the free recovery.

    """
    if not np.isscalar(T):
        if np.isscalar(R1):
            M = np.empty(np.size(T))
            for k, Tk in enumerate(T):
                M[k] = Mz_spgr(R1, Tk, TR, FA, v, Fw, j, n0, me)
        else:
            M = np.empty(np.shape(R1) + (np.size(T), ))
            for k, Tk in enumerate(T):
                M[...,k] = Mz_spgr(R1, Tk, TR, FA, v, Fw, j, n0, me)
        return M

    if np.isscalar(R1):
        nx = T/TR
        ncFA = np.cos(FA*np.pi/180)**nx
        M0 = n0*v*me
        Mss = Mz_ss(R1, TR, FA, v, Fw, j, me)
        K = _Mz_K(R1, v, Fw)
        E = np.exp(-T*K)
        return Mss + ncFA*E*(M0-Mss)

    elif np.ndim(R1)==1:

        if np.isscalar(v):
            nt = np.size(R1)
            M = np.empty(nt)
            for t in range(nt):
                jt = None if j is None else j[t]
                n0t = n0 if np.isscalar(n0) else n0[t]
                M[t] = Mz_spgr(R1[t], T, TR, FA, v, Fw, jt, n0t, me)
            return M
    
        nx = T/TR
        ncFA = np.cos(FA*np.pi/180)**nx
        M0 = np.multiply(np.multiply(n0, v), me)
        Mss = Mz_ss(R1, TR, FA, v, Fw, j, me)
        K = _Mz_K(R1, v, Fw)
        E = expm(-T*K)
        return Mss + ncFA*np.dot(E, M0-Mss)

    if np.isscalar(v):
        raise ValueError(
            'In a tissue with n compartments, v must be an n-element array.')
    n = np.shape(R1)
    R1 = np.reshape(R1, (n[0], -1))
    if j is not None:
        j = np.reshape(j, (n[0], -1))
    M = np.empty(R1.shape)
    for t in range(R1.shape[1]):
        jt = None if j is None else j[:,t]
        if np.ndim(n0)==2:
            n0t = n0[:,t]
        else:
            n0t = n0
        M[:,t] = Mz_spgr(R1[:,t], T, TR, FA, v, Fw, jt, n0t, me)
    return M.reshape(n) 







def signal(sequence, R1, S0, TR=None, FA=None, TC=None):
    # Private for now - generalize to umbrella signal function
    if sequence == 'SRC':
        return signal_free(S0, R1, TC, FA)
    if sequence == 'SR':
        return signal_spgr(S0, R1, TC, TR, FA)
    elif sequence == 'SS':
        return signal_ss(S0, R1, TR, FA)
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


def signal_free(S0, R1, T, FA, v=1, Fw=0, j=None, n0=0, R10=None):
    """Signal with readout after a free recovery.

    Args:
        S0 (float): Signal scaling factor (arbitrary units).
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a 
          tissue with a single compartment, R1 can have any shape. If R1 is a 
          scalar, the result for a well-mixed tissue is returned.
        T (float): time to center of k-space
        FA (float): flip angle
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        n0 (array-like, optional): initial relative magnetization at T=0. 
          If this is a scalar, all compartments are assumed to have the same 
          initial magnetization. Defaults to 0.
        R10 (float, optional): R1-value where S0 is defined. If not provided, 
          S0 is the scaling factor corresponding to infinite R10. Defaults 
          to None.

    Returns:
        np.ndarray: Signal in the same units as S0 and with the same 
        dimensions as R1.
    """
    if R10 is not None:
        S0 = S0/signal_free(1, R10, T, FA, v, Fw, j, n0)
    Mz = Mz_free(R1, T, v, Fw, j, n0)
    if np.isscalar(Mz):
        return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
    elif np.ndim(Mz)==1:
        if np.isscalar(v):
            return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
        else:
            return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz))
    else:
        return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz, axis=0))


def signal_ss(S0, R1, TR, FA, v=1, Fw=0, j=None, R10=None) -> np.ndarray:
    """Signal of a spoiled gradient echo sequence applied in steady state.

    Args:
        S0 (float): Signal scaling factor (arbitrary units).
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a 
          tissue with a single compartment, R1 can have any shape. If R1 is a 
          scalar, the result for a well-mixed tissue is returned.
        TR (float): Repetition time, or time between successive selective 
          excitations, in sec. 
        FA (float): Flip angle in degrees.
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        R10 (float, optional): R1-value where S0 is defined. If not provided, 
          S0 is the scaling factor corresponding to infinite R10. Defaults 
          to None.

    Returns:
        np.ndarray: Signal in the same units as S0
    """
    if R10 is not None:
        S0 = S0/signal_ss(1, R10, TR, FA, v, Fw, j)
    Mz = Mz_ss(R1, TR, FA, v, Fw, j)
    if np.isscalar(Mz):
        return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
    elif np.ndim(Mz)==1:
        if np.isscalar(v):
            return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
        else:
            return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz))
    else:
        return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz, axis=0))


def signal_spgr(S0, R1, T, TR, FA, TP=0.0, 
                  v=1, Fw=0, j=None, n0=0, R10=None) -> np.ndarray:
    """Signal of an SPGR sequence.

    Args:
        S0 (float): Signal scaling factor (arbitrary units).
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        T (float): time since the first readout rf-pulse.
        TR (float): Repetition time, or time between successive selective 
          excitations, in sec. 
        FA (array-like): Flip angle in degrees.
        TP (float, optional): Time (sec) between the preparation pre-pulse and 
          the first readout pulse. Defaults to 0.
        v (array-like, optional): volume fractions of the compartments. For a 
          one-compartment tissue this is a scalar - otherwise it is an 
          array with one value for each compartment. Defaults to 1.
        Fw (array-like, optional): Water flow between the compartments and to 
          the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
          array, where n is the number of compartments, and the off-diagonal 
          elements Fw[j,i] are the permeability for water moving from 
          compartment i into j. The diagonal elements Fw[i,i] quantify the 
          flow of water from compartment i to outside. For a closed system 
          with equal permeabilities between all compartments, a scalar value 
          for Fw can be provided. Defaults to 0.
        j (array-like, optional): normalized tissue magnetization flux. j has 
          to have the same shape as R1. Defaults to None.
        n0 (array-like, optional): initial relative magnetization at T=0. 
          If this is a scalar, all compartments are assumed to have the same 
          initial magnetization. Defaults to 0.
        R10 (float, optional): R1-value where S0 is defined. If not provided, 
          S0 is the scaling factor corresponding to infinite R10. Defaults to 
          None.

    Raises:
        ValueError: If TP is larger than TC.

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    if R10 is not None:
        S0 = S0/signal_spgr(1, R10, T, TR, FA, TP, v, Fw, j, n0)
    if TP>0:
        N0 = Mz_free(R1, TP, v, Fw, j, n0)
        if np.ndim(N0)==2:
            n0 = N0
            for t in range(N0.shape[1]):
                n0[:,t] = np.divide(N0[:,t], v)
        else:
            n0 = np.divide(N0, v)
    Mz = Mz_spgr(R1, T, TR, FA, v, Fw, j, n0)
    if np.isscalar(Mz):
        return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
    elif np.ndim(Mz)==1:
        if np.isscalar(v):
            return S0 * np.sin(FA*np.pi/180) * np.abs(Mz)
        else:
            return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz))
    else:
        return S0 * np.sin(FA*np.pi/180) * np.abs(np.sum(Mz, axis=0))


def signal_lin(R1, S0: float) -> np.ndarray:
    """Signal for any sequence operating in the linear regime.

    Args:
        R1 (array-like): Longitudinal relaxation rate in 1/sec.
        S0 (float): Signal scaling factor (arbitrary units).

    Returns:
        np.ndarray: Signal in arbitrary units, of the same length as R1.
    """
    return S0 * R1





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
        np.ndarray: Concentration inM , same length as S.
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
        >>> FA = 15         # deg

        Generate ground truth concentrations and signal data:

        >>> t = np.arange(0, 5*60, 0.1)     # sec
        >>> C = 0.003*(1-np.exp(-t/60))     # M
        >>> R1 = 1/T10 + r1*C               # Hz
        >>> S = dc.signal_free(100, R1, TC, FA)  # au

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
