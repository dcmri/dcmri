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

from scipy.linalg import expm
import numpy as np

import dcmri.lexicon_utils as lexicon
import dcmri.mz_lib as mz_lib



# TODO: For some ss sequences there is some duplication with K, J and KinvJ computed multiple times
# This needs rationalising

class Mz:
    def __init__(self, sequence='SS', **params):
        if sequence not in self._params_dict():
            raise ValueError(f"Sequence {sequence} is not defined. The options are {self._params_dict().keys()}.")
        
        self._cnfg = {'sequence': sequence}
        self._pars = lexicon.init(self._params())

        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()

    def _params_dict(self):
        return {
            'SS': ['TR', 'FA'],
            'SR': ['TC', 'TR', 'FA', 'TP', 'TA'],
            'IR-SS': ['TC', 'TR', 'FA', 'TP', 'TA'],
            'PR-SS': ['TC', 'TR', 'FA', 'TP', 'TA', 'PA'],
            'PR': ['TC', 'TR', 'FA', 'TP', 'TA', 'PA'],
            'SSI': ['TR', 'FA', 'TF', 'SA'],
            'GE-EPI': ['TE', 'TR', 'FA'],
            'SE-EPI': ['TE', 'TR', 'FA'],
            'None': [],
        }

    def _params(self):
        return self._params_dict()[self._cnfg['sequence']]
    
    def __call__(self, R1, v=None, Fw=None, j=None, me=None, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        # Set defaults for the optional arguments
        if v is None:
            nd = np.array(R1).ndim
            if nd==2:
                raise ValueError("For a multicompartment tissue, the volume fractions must be provided")
            else:
                v = 1
        if Fw is None:
            Fw = 0
        if j is None:
            j = np.zeros_like(R1)
        if me is None:
            me = 1

        # Possible input shapes for R1:
        # scalar, 1D (nt), 1D (nc), 2D (nc, nt)
        #    
        # Keep input shape for return values
        input_shape = np.shape(R1)

        # Convert arguments to standard 2D shape (nc, nt) for computations

        # -- The number of compartments is decided by the size of v
        v = np.atleast_1d(v)
        nc = v.size

        # --- Check formatting of Fw
        # Special case: In a multicompartment system, a constant 
        # Fw = a closed system with constant permeability.
        Fw = np.atleast_1d(Fw)
        if nc > 1:
            if Fw.size==1:
                Fw = np.full((nc, nc), Fw[0])
                np.fill_diagonal(Fw, 0)
        if Fw.size != nc * nc:
            raise ValueError("For an n-compartment tissue, Fw must have shape (n, n).")
        Fw = Fw.reshape(nc, nc)
        
        # Reshape R1 to (nc, nt) and derive nt
        R1 = np.atleast_1d(R1)
        if nc > 1:
            if R1.size > nc:
                if R1.ndim != 2:
                    raise ValueError("For a tissue with nc compartments and nt time points, R1 must have shape (nc, nt).")
                if R1.shape[0] != nc:
                    raise ValueError("For a tissue with nc compartments and nt time points, R1 must have shape (nc, nt).")
                nt = R1.shape[1]
            else:
                nt = 1
        else:
            nt = R1.size
        R1 = R1.reshape(nc, nt)

        # Reshape influx to standard shape
        j = np.atleast_1d(j)
        if j.size != nc * nt:
            raise ValueError('For a tissue with nc compartments and nt time points, the influx j must have shape (nc, nt).')
        j = j.reshape(nc, nt)

        # Delegate computation in standard form to helper functions
        sequence = self._cnfg['sequence']
        if sequence == 'SS':
            Mz = _Mz_spgr_in_ss(R1, v, Fw, j, me, p['TR'], p['FA'])
        elif sequence == 'SR':
            Mz = _Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'], p['TP'], p['TA'], 90) 
        elif sequence == 'IR-SS':
            Mz = _Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'], p['TP'], p['TA'], 180)
        elif sequence == 'PR-SS':
            Mz = _Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'], p['TP'], p['TA'], p['PA']) 
        elif sequence == 'PR':
            Mz = _Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'], p['TP'], p['TA'], p['PA']) 
        elif sequence == 'SSI':
            Mz = _Mz_ssi(R1, v, Fw, j, me, p['TR'], p['FA'], p['TF'], p['SA'])
        elif sequence == 'GE-EPI':
            Mz = _Mz_ge(R1, v, Fw, j, me, p['TE'], p['TR'], p['FA'])
        elif sequence == 'SE-EPI':
            Mz = _Mz_se(R1, v, Fw, j, me, p['TE'], p['TR'], p['FA'])
        elif sequence == 'None': 
            Mz = np.full_like(R1, me)

        # Return result in original shape
        if input_shape == ():
            return Mz[0,0]
        else:
            return Mz.reshape(input_shape)

    
def _Mz_spgr_in_ss(R1, v, Fw, j, me, TR, FA) -> np.ndarray:
    """Spoiled gradient echo sequence in steady state"""
    nc, nt = R1.shape

    M = [mz_lib.Mz_ss_spgr(R1[:,t], v, Fw, j[:,t], me, TR, FA) for t in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def _Mz_pr_spgr(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA): 
    """This models SPGR with a preparation pulse and linear k-space ordering

    - A preparation pulse PA at the start of each time interval, 
    - Free recovery over a time TP
    - FA readout pulses separated by TR for a duration of 2 * (TC-TP)
    - Free recovery until the start of the next time interval. 
    - And a readout at time TC after the preparation pulse.

    R1 is assumed to be constant on each time interval.
    """
    nc, nt = R1.shape
    Mt = v * me

    M = []
    for k in range(nt):
        M_sig, Mt = mz_lib.Mz_pr_spgr_prop(Mt, R1[:,k].T, v, Fw, j[:,k].T, me, PA, TP, TC, TR, FA, TA)
        M.append(M_sig)
    
    return np.array(M).T.reshape(nc, nt)


def _Mz_pr_spgr_in_ss(R1, v, Fw, j, me, TC, TR, FA, TP, TA, PA):
    """This models SPGR with a preparation pulse and linear k-space ordering
    running in the steady state

    R1 is assumed to be constant on each time interval.
    """
    nc, nt = R1.shape

    M = []
    for k in range(nt):
        Mss = mz_lib.Mz_pr_spgr_ss(R1[:,k].T, v, Fw, j[:,k].T, me, PA, TP, TC, TR, FA, TA)
        M_sig, _ = mz_lib.Mz_pr_spgr_prop(Mss, R1[:,k].T, v, Fw, j[:,k].T, me, PA, TP, TC, TR, FA, TA)
        M.append(M_sig)    
    
    return np.array(M).T.reshape(nc, nt)


def _Mz_ssi(R1, v, Fw, j, me, TR, FA, TF, SA): 
    """This models steady-state imaging with inflow effects

    - Initial magnetization determined by saturation slabs outside the imaging volume. Without slabs, n_init=1 
    - FA readout pulses separated by TR for a duration of TF (inflow time)

    Each readout starts the same - no build=up effects

    R1 is assumed to be constant on each time interval.
    """
    nc, nt = R1.shape
    cFA = np.cos(np.radians(FA))
    n = np.floor(TF / TR) # n pulses to readout
    nFA = cFA**n
    cSA = np.cos(np.radians(SA))
    M0 = cSA * v * me

    def _Mz_ssi_prop(R1_t, j_t):
        K_t = mz_lib.Mz_K(R1_t, v, Fw)
        Mss_t = mz_lib.Mz_ss_spgr(R1_t, v, Fw, j_t, me, TR, FA)
        # FA-pulses until time TF to get the readout
        if nc==1:
            En_t = np.exp(-TF * K_t)
            M_sig_t = Mss_t + nFA * En_t * (M0 - Mss_t)
        else:
            En_t = expm(-TF * K_t)
            M_sig_t = Mss_t + nFA * En_t @ (M0 - Mss_t)           
        return M_sig_t

    M = [_Mz_ssi_prop(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def _Mz_ge(R1, v, Fw, j, me, TE, TR, FA): 
    """Mz for one slice in a GE-EPI sequence
    """
    nc, nt = R1.shape
    pulse = [[FA, TR]]
    read = [[FA, TE]]

    def _Mz_ge_t(R1_t, j_t):
        # Compute steady state
        Mss_t = mz_lib.Mz_ss(R1_t, v, Fw, j_t, me, pulse)
        # Compute Mz at signal collection
        M_sig_t = mz_lib.Mz_prop(Mss_t, R1_t, v, Fw, j_t, me, read)
        return M_sig_t

    M = [_Mz_ge_t(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)


def _Mz_se(R1, v, Fw, j, me, TE, TR, FA): 
    """Mz for one slice in a GE-EPI sequence
    """
    nc, nt = R1.shape
    pulse = [[FA, TE / 2], [180, TR - TE/2]]
    read = [[FA, TE]]

    def _Mz_se_t(R1_t, j_t):
        # Compute steady state
        Mss_t = mz_lib.Mz_ss(R1_t, v, Fw, j_t, me, pulse)
        # Compute Mz at signal collection
        M_sig_t = mz_lib.Mz_prop(Mss_t, R1_t, v, Fw, j_t, me, read)
        return M_sig_t

    M = [_Mz_se_t(R1[:,k].T, j[:,k].T) for k in range(nt)]
    return np.array(M).T.reshape(nc, nt)