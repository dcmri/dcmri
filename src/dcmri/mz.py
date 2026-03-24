from scipy.linalg import expm
import numpy as np



def params_Mz(sequence):
    return {
        'free': ['TC'],
        'SS': ['TR', 'FA'],
        'SR': ['TC', 'TR', 'FA', 'TP'],
        'IR': ['TC', 'TR', 'FA', 'TP'],
        'SPGR': ['TC', 'TR', 'FA', 'TP', 'n_init'],
        'SSI': ['TF', 'TR', 'FA'],
        'None': [],
    }[sequence]


def Mz(
    # model
    sequence, 
    # tissue
    R1=None, v=1, Fw=0, j=None, n_init=0, me=1,
    # parameters
    TR=None, FA=None, TC=None, TF=None, TP=None,      
):
    """Longitudinal magnetization.

    See section :ref:`basics-relaxation-T1` for more detail.

    Args:
        R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
          with n compartments, the first dimension of R1 must be n. For a
          single compartment, R1 can be scalar or a 1D time-array.
        T (float): duration of free recovery.
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
        n_init (array-like, optional): initial relative magnetization at T=0. 
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

        >>> Mz = dc._Mz_free(R1, TI, n_init=-1)
        >>> Mz_e = dc._Mz_free(R1, TI, n_init=-1, Fw=f, j=f)
        >>> Mz_i = dc._Mz_free(R1, TI, n_init=-1, Fw=f, j=-f)

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
        >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=[f, 0])

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
        >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=j)

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
        >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=j)

        >>> plt.plot(t, Mz[0,:,5], label='Central compartment', linewidth=3)
        >>> plt.plot(t, Mz[1,:,5], label='Peripheral compartment', linewidth=3)
        >>> plt.xlabel('Time (sec)')
        >>> plt.ylabel('Magnetization (A/cm)')
        >>> plt.legend()
        >>> plt.show()      

    """
    # Possible shapes for R1:
    # scalar, 1D (nt, ), 1D (nc, ) and 2D (nc, nt)

    # Keep input shape for return values
    input_shape = np.shape(R1)

    # Ensure inputs are 1D or 2D arrays
    R1 = np.atleast_1d(R1)
    v = np.atleast_1d(v)
    Fw = np.atleast_1d(Fw)
    if j is None:
        j = np.zeros_like(R1)
    else:
        j = np.atleast_1d(j)
    n_init = np.atleast_1d(n_init) # One value per compartment

    # The number of compartments is decided by the size of v
    nc = v.size

    # In a multicompartment system
    # A constant Fw = a closed system with constant permeability
    # A constant init = same init in all compartments
    if nc > 1:
        if Fw.size==1:
            Fw = np.full((nc, nc), Fw[0])
            np.fill_diagonal(Fw, 0)
        if n_init.size==1:
            n_init = np.full(nc, n_init[0])  

    # Find nt from the size of R1 (nc, nt)
    R1 = R1.reshape(nc, -1)
    nt = R1.shape[1]

    # Reshape others inputs to standard shape
    v = v.reshape(nc)
    j = j.reshape(nc, nt)
    Fw = Fw.reshape(nc, nc)
    n_init = n_init.reshape(nc)

    # Delegate computation in standard form to helper functions
    if sequence =='free':
        Mz = _Mz_free(R1, v, Fw, j, me, n_init, TC)
    elif sequence == 'SS':
        Mz = _Mz_ss(R1, v, Fw, j, me, TR, FA)
    elif sequence == 'SPGR':
        Mz = _Mz_spgr(R1, v, Fw, j, n_init, me, TC, TR, FA, TP) 
    elif sequence == 'SR':
        Mz = _Mz_spgr(R1, v, Fw, j, 0, me, TC, TR, FA, TP) 
    elif sequence == 'IR':
        Mz = _Mz_spgr(R1, v, Fw, j, -1, me, TC, TR, FA, TP)
    elif sequence == 'SSI':
        Mz = _Mz_spgr(R1, v, Fw, j, 1, me, TF, TR, FA, 0)
    elif sequence == 'None':
        Mz = np.full_like(R1, me)

    # Return result in original shape
    if input_shape == ():
        return Mz[0,0]
    else:
        return Mz.reshape(input_shape)



def _Mz_free(R1: np.ndarray, v: np.ndarray, Fw, j:np.ndarray, me, n_init, T):

    nc, nt = R1.shape
        
    # One compartment

    if nc==1:

        # 1. Compute K and M0 for all t simultaneously
        K = R1 + Fw[0,0] / v
        M0 = n_init * me * v
        
        # 2. Compute J and E
        J = (R1 * v + j) * me
        E = np.exp(-T * K)
        Kinv = np.divide(1.0, K, out=np.zeros_like(K, dtype=float), where=K != 0)
        M = E * M0 + (1 - E) * Kinv * J
        return M

    # Multiple compartments

    Id = np.eye(nc)
    M0 = n_init * me * v

    # --- Single Time Point (R1t is 1D) ---
    def mfree(R1t, jt):
        K = _Mz_K(R1t, v, Fw)
        J = (R1t * v + jt) * me
        E = expm(-T * K)
        KinvJ = np.linalg.solve(K, J)
        M = E @ M0 + (Id - E) @ KinvJ
        return M        

    M = [mfree(R1[:,t].T, j[:,t].T) for t in range(nt)]
    
    return np.array(M).T

    
def _Mz_ss(R1: np.ndarray, v: np.ndarray, Fw, j: np.ndarray, me, TR, FA) -> np.ndarray:
    nc, nt = R1.shape
    
    # One compartment
    if nc==1:
        M = [me * _Nz_ss_1c(R1[0,t], v, Fw[0,0], j[0,t], TR, FA) for t in range(nt)]

    # Multiple compartments 
    else:
        M = [me * _Nz_ss(R1[:,t], v, Fw, j[:,t], TR, FA) for t in range(nt)]
    
    M = np.array(M).reshape(nc, nt) 
    return M  


def _Mz_spgr(R1, v, Fw, j, n_init, me, T, TR, FA, TP): 

    nc, nt = R1.shape
    Id = np.eye(nc)
    M0 = n_init * v * me

    # One compartment
    if nc==1:
        def mz_spgr(R1t, jt):
            nx = T/TR
            ncFA = np.cos(np.radians(FA))**nx
            Mss = me * _Nz_ss_1c(R1t, v, Fw[0,0], jt, TR, FA)
            K = _Mz_K(R1t, v, Fw)
            if TP > 0:
                EP = np.exp(-TP * K)
                J = (R1t * v + jt) * me
                Kinv = np.zeros_like(K, dtype=float)
                Kinv = np.divide(1.0, K, out=Kinv, where=K != 0)
                M0t = EP * M0 + (1 - EP) * Kinv * J
            else:
                M0t = M0
            E = np.exp(-T * K)
            return Mss + ncFA * E * (M0t - Mss)
        
        M = [mz_spgr(R1[0,t], j[0,t]) for t in range(nt)]
        return np.array(M).reshape(nc, nt)

    # Multiple compartments  
    def mz_spgr_nc(R1t, jt):
        nx = T/TR
        ncFA = np.cos(np.radians(FA))**nx
        Mss = me * _Nz_ss_aex(R1t, v, Fw, jt, TR, FA)
        K = _Mz_K(R1t, v, Fw)
        if TP > 0:
            EP = expm(-TP * K)
            J = (R1t * v + jt) * me
            KinvJ = np.linalg.solve(K, J)
            M0t = EP @ M0 + (Id - EP) @ KinvJ
        else:
            M0t = M0
        E = expm(-T * K)
        return Mss + ncFA * E @ (M0t - Mss)

    M = [mz_spgr_nc(R1[:,t], j[:,t]) for t in range(nt)]
    return np.array(M).T



### HELPERS


def _Nz_ss_1c(R1, v, Fw, j, TR, FA):
    K = R1 + Fw/v if v > 0 else R1
    J = R1 * v + j
    E = np.exp(-TR * K)
    cFA = np.cos(FA*np.pi/180)
    n = (1-E) / (1-cFA*E)
    if K==0:
        return n
    else:
        return n * J/K


def _Nz_ss(R1, v, Fw, j, TR, FA):   
    off_diag = ~np.eye(Fw.shape[0], dtype=bool)
    PSw = Fw[off_diag]

    if np.all(PSw == 0):
        return _Nz_ss_nex(R1, v, Fw, j, TR, FA)
    
    elif np.all(np.isinf(PSw)):
        return _Nz_ss_fex(R1, v, Fw, j, TR, FA)
    
    elif 0 < np.count_nonzero(np.isinf(PSw)):
        raise NotImplementedError(
            'Water exchange with some (but not all) infinite PS '
            'values is currently not implemented.')
    else:
        return _Nz_ss_aex(R1, v, Fw, j, TR, FA)


def _Nz_ss_fex(R1, v, Fw, j, TR, FA):
    R1 = np.sum(v * R1) / np.sum(v)
    fo = np.diag(Fw)
    N = _Nz_ss_1c(R1, np.sum(v), np.sum(fo), np.sum(j), TR, FA)
    Nc = [N * vc / np.sum(v) for vc in v]
    return np.stack(Nc)

def _Nz_ss_nex(R1, v, Fw, j, TR, FA):
    nc = v.size
    fo = np.diag(Fw)
    Nc = [_Nz_ss_1c(R1[c], v[c], fo[c], j[c], TR, FA) for c in range(nc)]
    return np.stack(Nc) 

def _Nz_ss_aex(R1, v, Fw, j, TR, FA):
    K = _Mz_K(R1, v, Fw)
    J = R1 * v + j
    E = expm(-TR * K)
    I = np.eye(R1.size)
    cFA = np.cos(np.radians(FA))

    # A = K @ (I - cos(FA) * E)
    A = K @ (I - cFA * E)
    
    # solve(A, B) is better than dot(inv(A), B)
    return np.linalg.solve(A, (I - E) @ J)  

def _Mz_K(R1, v, Fw):
    nc = v.size
    # Case 1: Single Compartment
    if nc==1:
        return R1 + Fw / v

    # Case 2: Multi-Compartment
    # Off-diagonal elements: -Fw[row, col] / v[col]
    K = -Fw / v

    # Diagonal elements: R1[i] + (Sum of water leaving i) / v[i]
    # Summing axis=0 gives the total flow out of each compartment (the columns)
    total_outflow = np.sum(Fw, axis=0)
    diag_elements = R1 + total_outflow / v
    
    # Overwrite the diagonal of our K matrix
    np.fill_diagonal(K, diag_elements)
    
    return K
    