"""Signal after readout of given Mz.

Args:
    S0 (float): Signal scaling factor (arbitrary units).
    R2 (array-like): Transverse relaxation rate R1 or R2* in 1/sec. 
    FAR (float): Readout flip angle (deg)
    TE (float): Echo time (sec)
    noise_sdev (float, optional): standard deviation of the signal noise. 

Returns:
    np.ndarray: Signal in the same units as S0 and with the same 
    dimensions as Mz.
"""  

"""Longitudinalitudinal magnetization.

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
from copy import deepcopy

from scipy.special import i0, i1
import numpy as np

from dcmri.core import SuperFunc
from dcmri.lexicon import SEQUENCES
from dcmri.lexicon import MZ_PREP
from dcmri import bloch


# TODO: For some ss sequences there is some duplication with K, J and KinvJ computed multiple times
# This needs rationalising




class Signal(SuperFunc):

    configs = {'sequence': deepcopy(list(SEQUENCES.keys()))}
    
    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(R1=None, R2s=None, R2=None, R1i=None, Fi=None)
        self._override_pars(**params)
    
    def _params(self):
        sequence = self._cnfg['sequence']
        pars = ['R1', 'R2s', 'R2', 'R1i', 'Fi', 'me', 'v', 'Fw']
        pars += SEQUENCES[sequence]['parameters']['prep']
        pars += SEQUENCES[sequence]['parameters']['read']
        pars = list(set(pars))
        pars.sort()
        return pars
    
    def __call__(self, **params):
        p = self._update_pars(**params)

        # Possible input shapes for R1:
        # scalar, 1D (nt), 1D (nc), 2D (nc, nt)
        sequence = self._cnfg['sequence']
        tissue_mz_sequence = SEQUENCES[sequence]['mz_prep_tissue']
        inflow_mz_sequence = SEQUENCES[sequence]['mz_prep_inflow']

        # Inflow of magnetization
        if p['R1i'] is None: 
            j = None
        else:
            R1i = np.atleast_1d(p['R1i'])
            R1shape = np.atleast_1d(p['R1']).shape
            if R1shape != R1i.shape:
                raise ValueError(f"R1 and R1i must have the same shape. R1 has shape {R1shape} and R1i has shape {R1i.shape}.")
            if p['Fi'] is None:
                raise ValueError(f"Fi must be provided if R1i is provided for sequence {self._cnfg['sequence']}.")
            
            # inflow = 1 closed compartment
            pi = p | {'v': 1, 'Fw': 0} 
            mz_inflow = Longitudinal(inflow_mz_sequence, **pi)

            # Compute magnetization inflow
            Fi = np.array(p['Fi'])
            if Fi.size==1:
                j = Fi * mz_inflow(R1i)
            else:
                if Fi.size != R1i.shape[0]:
                    raise ValueError(f"Fi must have the same number of elements as the first dimension of R1i. Fi has {Fi.size} elements and R1i has shape {R1i.shape}.")
                j = np.zeros_like(R1i)
                for i in range(Fi.size):
                    j[i,:] = Fi[i] * mz_inflow(R1i[i,:])

        # Magnetization and readout 
        if p['R1'] is None:
            # No R1 provided -> DSC without T1-weighting
            if tissue_mz_sequence in ['GE-EPI']:
                if p['R2s'] is None:
                    raise ValueError('For R2s-weighted sequences, an R2s value must be provided.')
                Mz = np.full_like(p['R2s'], p['me'])
            elif tissue_mz_sequence in ['SE-EPI']:
                if p['R2'] is None:
                    raise ValueError('For R2-weighted sequences, an R2 value must be provided.')
                Mz = np.full_like(p['R2'], p['me'])
            elif tissue_mz_sequence in ['DE-EPI']:
                if (p['R2'] is None) and (p['R2s'] is None):
                    raise ValueError('For R2/R2s-weighted sequences, R2 and R2s values must be provided.')
                if np.size(p['R2']) != np.size(p['R2s']):
                    raise ValueError('For R2/R2s-weighted sequences, R2 and R2s must have the same size.')
                Mz = np.full_like(p['R2'], p['me'])
            else:
                raise ValueError('For T1-weighted sequences, an R1 value must be provided.')
        else:
            # R1 provided -> include T1-weighting in DCE and DSC.
            Mz = Longitudinal(tissue_mz_sequence, **p)(p['R1'], j)
        return Readout(sequence, **p)(Mz=Mz, R2=p['R2'], R2s=p['R2s'])
    


class Longitudinal(SuperFunc):

    configs = {'sequence': deepcopy(list(MZ_PREP.keys()))}

    def __init__(self, sequence='SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(v=None, Fw=0, me=1)
        self._override_pars(**params)

    def _params(self):
        pars = ['v', 'Fw', 'me']
        pars += deepcopy(MZ_PREP[self._cnfg['sequence']]['parameters'])
        pars = list(set(pars))
        pars.sort()
        return pars
    
    def __call__(self, R1=None, j=None, **params):
        p = self._update_pars(**params)

        if R1 is None:
            raise ValueError("Cannot compute Mz without R1. Please provide R1 as an argument.")
        if j is None:
            j = np.zeros_like(R1)
        
        v = p['v']
        Fw = p['Fw']
        me = p['me']
        
        # Set defaults for v
        if v is None:
            nd = np.array(R1).ndim
            if nd==2:
                raise ValueError("For a multicompartment tissue, the volume fractions must be provided")
            else:
                v = 1

        # Possible input shapes for R1:
        # scalar, 1D (nt), 1D (nc), 2D (nc, nt)
    
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
        # Keep input shape for return values
        input_shape = np.shape(R1)

        R1 = np.atleast_1d(R1)
        if nc > 1:
            if R1.size > nc:
                if R1.ndim != 2:
                    raise ValueError(f"For a tissue with {nc} compartments and nt time , R1 must have shape ({nc}, nt).")
                if R1.shape[0] != nc:
                    raise ValueError(f"For a tissue with {nc} compartments and nt time points, R1 must have shape ({nc}, nt).")
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

        if sequence == 'Eq': 
            Mz = np.full_like(R1, me)
        elif sequence == 'IR-SS':
            Mz = bloch.Mz_ge(R1, v, Fw, j, me, p['TA'], 180)
        elif sequence == 'SR-SS':
            Mz = bloch.Mz_ge(R1, v, Fw, j, me, p['TA'], 90)
        elif sequence == 'PR-SS':
            Mz = bloch.Mz_ge(R1, v, Fw, j, me, p['TA'], p['PA'])

        elif sequence == 'SPGR':
            Mz = bloch.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], 0, p['TA'], 0) 
        elif sequence == 'SR-SPGR':
            Mz = bloch.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 90) 
        elif sequence == 'IR-SPGR':
            Mz = bloch.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 180) 
        elif sequence == 'PR-SPGR':
            Mz = bloch.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], p['PA'])
        
        elif sequence == 'SPGR-SS':
            Mz = bloch.Mz_spgr_in_ss(R1, v, Fw, j, me, p['TR'], p['FA'] * p['B1corr'])
        elif sequence == 'SR-SPGR-SS':
            Mz = bloch.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 90) 
        elif sequence == 'IR-SPGR-SS':
            Mz = bloch.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 180)
        elif sequence == 'PR-SPGR-SS':
            Mz = bloch.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], p['PA']) 

        elif sequence == 'SSI':
            Mz = bloch.Mz_ssi(R1, v, Fw, j, me, p['TR'], p['FA'] * p['B1corr'], p['TF'], p['SA'])

        elif sequence == 'GE-EPI':
            Mz = bloch.Mz_ge(R1, v, Fw, j, me, p['TR'], p['FA'] * p['B1corr'])
        elif sequence == 'SE-EPI':
            Mz = bloch.Mz_se(R1, v, Fw, j, me, p['TE'], p['TR'], p['FA'] * p['B1corr'])
        elif sequence == 'DE-EPI':
            Mz = bloch.Mz_se(R1, v, Fw, j, me, p['TE2'], p['TR'], p['FA'] * p['B1corr'])

        # Return result in original shape
        if input_shape == ():
            return Mz[0,0]
        else:
            return Mz.reshape(input_shape)


class Readout(SuperFunc): 
    configs = {'sequence': deepcopy(list(SEQUENCES.keys()))}
    
    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(Mz=None, R2s=None, R2=None)
        self._override_pars(**params)
    
    def _params(self):
        pars = ['Mz', 'R2s', 'R2']
        pars += SEQUENCES[self._cnfg['sequence']]['parameters']['read']
        pars.sort()
        return deepcopy(pars)

    def __call__(self, **params):
        p = self._update_pars(**params)

        # T2-weighted sequences must have R2 or R2* specified. 
        if self._cnfg['sequence'] in ['GE-EPI']:
            if p['R2s'] is None:
                raise ValueError("R2* must be provided for GE-EPI sequence.")
            return _mz_readout(p['Mz'], p['R2s'], p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])

        elif self._cnfg['sequence'] in ['SE-EPI']:
            if p['R2'] is None:
                raise ValueError("R2 must be provided for SE-EPI sequence.")
            return _mz_readout(p['Mz'], p['R2'], p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])
        
        elif self._cnfg['sequence'] in ['DE-EPI']:
            if (p['R2'] is None) or (p['R2s'] is None):
                raise ValueError("R2 and R2s must both be provided for DE-EPI sequence.")
            if np.size(p['R2']) != np.size(p['R2s']):
                raise ValueError('For R2/R2s-weighted sequences, R2 and R2s must have the same size.')
            S_GE = _mz_readout(p['Mz'], p['R2s'], p['S0'], p['FA'] * p['B1corr'], p['TE1'], p['noise_sdev'])
            S_SE = _mz_readout(p['Mz'], p['R2'], p['S0'], p['FA'] * p['B1corr'], p['TE2'], p['noise_sdev'])
            return np.stack((S_GE, S_SE)) # n_channels, n_times

        # T1-weighted sequences with R2/R2* weighting. R2* required if TE > 0.
        elif p['TE'] > 0:
            if p['R2s'] is None:
                raise ValueError(f"R2* is required for a {self._cnfg['sequence']} sequence with TE > 0.")  
            return _mz_readout(p['Mz'], p['R2s'], p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])
        elif p['TE'] == 0:
            return _mz_readout(p['Mz'], 0, p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])

    
def _mz_readout(Mz, R2, S0, FA, TE, noise_sdev):
    # Mz has shape 1D (nt, ) or 2D (nc, nt)
    Mz = np.array(Mz)
    if Mz.ndim==1:
        Mz_total = Mz
    else:
        Mz_total = Mz.sum(axis=0)
    sFA = np.sin(np.radians(FA))
    decay = np.exp(-TE * R2)
    Mxy = S0 * decay * sFA * Mz_total
    return _signal_rice(np.abs(Mxy), noise_sdev)
    

def _signal_rice(nu, sigma)-> np.ndarray:
    if sigma==0:
        return nu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        K = nu**2 / (2*sigma**2)
        arg = K/2
        pref = sigma * np.sqrt(np.pi/2)
        rice_mean = pref * np.exp(-K/2) * ((1+K)*i0(arg) + K*i1(arg))
    # Nan values are points where the distribution is indistinguisable from Gaussian
    return np.where(np.isnan(rice_mean) | np.isinf(rice_mean), nu, rice_mean)
