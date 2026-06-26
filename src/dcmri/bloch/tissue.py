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

import numpy as np

from dcmri.core.function import Function
from dcmri.core.sequences import SEQUENCES
from dcmri.bloch import seqs


# TODO: For some ss sequences there is some duplication with K, J and KinvJ computed multiple times
# This needs rationalising


class Longitudinal(Function):
    configs = {
        'sequence': deepcopy(list(SEQUENCES.keys())),
        'inflow': [False, True],
    }
    def __init__(self, sequence='SPGR-SS', inflow=False, defaults=None, **params):
        cnfg = {
            'sequence': sequence,
            'inflow': inflow,
        }
        self._set_config(cnfg)
        self._set_params(defaults) 

    def params(self):
        seq = self._cnfg['sequence']
        # Sequence parameters
        pars = deepcopy(SEQUENCES[seq]['parameters']['prep'])
        # Tissue parameters
        pars += ['v', 'Fw', 'me']
        if SEQUENCES[seq]['mz_prep_tissue'] != 'Eq':
            pars += ['R1']
        if self._cnfg['inflow']:
            if SEQUENCES[seq]['mz_prep_inflow'] != 'Eq':
                pars += ['Fi', 'R1i']
        pars = list(set(pars))
        pars.sort()
        return pars
    
    def __call__(self, **params):
        p = self._update_params(params)
        sequence = self._cnfg['sequence']

        # --- Format vw
        v = np.atleast_1d(p['v'])
        nc = v.size # -- The number of compartments is decided by the size of v

        # If the sequence does not have T1-weighting, return scalar equilibrium
        if 'R1' not in self.params():
            return np.full(nc, p['me'])

        # Possible input shapes for R1:
        # scalar, 1D (nt), 1D (nc), 2D (nc, nt)  
        # Corresponding Mz output shapes are scalar, (nc, ) or (nc, nt)
        # Arguments are converted to standard 2D shape (nc, nt) for computations   
        input_shape = np.shape(p['R1'])
        R1 = np.atleast_1d(p['R1'])
        if R1.ndim==2:
            if nc != R1.shape[0]:
                raise ValueError("v must have one element for each tissue compartment")
        
        # --- Check formatting of Fw
        # Reshape Fw to (nc, nc)
        Fw = np.atleast_1d(p['Fw'])
        if nc > 1:
            if Fw.size==1:
                Fw = np.full((nc, nc), Fw[0])
                np.fill_diagonal(Fw, 0)
        if Fw.size != nc * nc:
            raise ValueError("For an n-compartment tissue, Fw must have shape (n, n).")
        Fw = Fw.reshape(nc, nc)
      
        # Reshape R1 to (nc, nt)
        if nc == 1:
            nt = R1.size
        else:
            if R1.size > nc:
                if R1.ndim != 2:
                    raise ValueError(f"For a tissue with {nc} compartments and nt time , R1 must have shape ({nc}, nt).")
                # if R1.shape[0] != nc:
                #     raise ValueError(f"For a tissue with {nc} compartments and nt time points, R1 must have shape ({nc}, nt).")
                nt = R1.shape[1]
            else:
                nt = 1
        R1 = R1.reshape(nc, nt)

        # Inflow of magnetization
        j = None
        if self._cnfg['inflow']:
            if 'R1i' not in p: # inflow at equilibrium
                j = np.full_like(R1, p['me'])
            elif np.size(p['R1i']) != nc * nt:
                raise ValueError(f'R1i must have the same size as R1 ({nc * nt}) ')  
            else:
                # Compute magnetization inflow
                mz_prep_inflow = SEQUENCES[sequence]['mz_prep_inflow']

                R1i = np.atleast_1d(p['R1i']).reshape(nc, nt)
                Fi = np.array(p['Fi'])
                if Fi.size != R1i.shape[0]:
                    raise ValueError(f"Fi must have the same number of elements as the first dimension of R1i. Fi has {Fi.size} elements and R1i has shape {R1i.shape}.")
                Fi = Fi.reshape(nc)
                j = np.zeros_like(R1i)
                for i in range(Fi.size):
                    vi, Fwi, ji = 1, 0, None # inflow = 1 closed compartment
                    j[i,:] = Fi[i] * _Mz(mz_prep_inflow, R1i[i,:], vi, Fwi, ji, p)

        # Delegate computation in standard form to helper functions
        mz_prep_sequence = SEQUENCES[sequence]['mz_prep_tissue']
        Mz = _Mz(mz_prep_sequence, R1, v, Fw, j, p)

        # Return result must one of the input shapes for Readout: scalar, (nc, ) or (nc, nt)
        if input_shape == ():  #scalar, 1D (nt), 1D (nc), 2D (nc, nt)
            return Mz[0,0] #input scalar, return scalar
        elif np.size(input_shape)==1:
            if nc==1: # input 1D (nt)
                return Mz # return (1, nt)
            else: # input 1D (nc)
                return Mz[:,0] # return (nc, )
        else: # input 2D (nc, nt)
            return Mz # output 2D (nc, nt)
        



class Readout(Function): 
    configs = {'sequence': deepcopy(list(SEQUENCES.keys()))}
    
    def __init__(self, sequence='3D-SPGR-SS', defaults=None, **params):
        cnfg = {
            'sequence': sequence
        }
        self._set_config(cnfg)
        self._set_params(defaults)
    
    def params(self):
        seq = self._cnfg['sequence']
        # Sequence parameters
        pars = SEQUENCES[seq]['parameters']['read']
        # Tissue parameters
        weighting = SEQUENCES[seq]['parameters']['tissue']
        if 'R2s' in weighting:
            pars += ['R2s']
        if 'R2' in weighting:
            pars += ['R2']
        pars.sort()
        return deepcopy(pars)

    def __call__(self, Mz, **params) -> np.ndarray: # (n_channels, n_times) or (n_times)
        p = self._update_params(params)
        seq = self._cnfg['sequence']

        # # Check that required parameters are provided
        # if 'R2s' in p:
        #     if p['R2s'] is None:
        #         raise ValueError("R2* must be provided for a T2*-weighted sequence.")
        # if 'R2' in p:
        #     if p['R2'] is None:
        #         raise ValueError("R2 must be provided for a T2-weighted sequence.")

        # Possible shapes for Mz are scalar, (ncomps, ) or (ncomps, nt)
        # R2 and R2s must match Mz in size but not shape
        # All other parameters are scalars

        # Result is returned in shape (n_channels, n_times) or (n_times) or (n_channels) or scalar

        # --- Convert input shapes to standard format (ncomps, ntimes)
        input_shape = np.shape(Mz)
        Mz = np.atleast_1d(Mz)
        if Mz.ndim==1:
            nc, nt = Mz.size, 1
            Mz = Mz.reshape(nc, nt)
        else:
            nc, nt = Mz.shape

        if 'R2s' in self.params():
            R2s = np.atleast_1d(p['R2s'])
            if R2s.size == nt:
                R2s = np.stack([R2s] * nc, axis=0)
            if R2s.size != nc * nt:
                raise ValueError(f"Size of R2* ({R2s.size}) does not match dimensions ({nc}, {nt}) of Mz.")
            R2s = R2s.reshape(nc, nt)

        if 'R2' in self.params():
            R2 = np.atleast_1d(p['R2'])
            if R2.size == nt:
                R2 = np.stack([R2] * nc, axis=0)
            if R2.size != nc * nt:
                raise ValueError(f"Size of R2 ({R2.size}) does not match dimensions ({nc}, {nt}) of Mz.")
            R2 = R2.reshape(nc, nt)

        if seq in ['Eq-SE-EPI', 'SE-EPI']:
            signal = seqs.mz_readout(Mz, R2, p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])
            signal = signal.reshape(1, -1)
        
        elif seq in ['Eq-DE-EPI', 'DE-EPI']:
            # if np.size(R2) != np.size(R2s):
            #     raise ValueError('R2 and R2s must have the same size.')
            GE = seqs.mz_readout(Mz, R2s, p['S0'], p['FA'] * p['B1corr'], p['TE1'], p['noise_sdev'])
            SE = seqs.mz_readout(Mz, R2, p['S0'], p['FA'] * p['B1corr'], p['TE2'], p['noise_sdev'])
            signal = np.stack((GE, SE)) # n_channels, n_times

        elif seq in ['ZTE-3D-SPGR-SS', 'ZTE-3D-IR-SPGR-SS']:
            signal = seqs.mz_readout(Mz, np.zeros_like(Mz), p['S0'], p['FA'] * p['B1corr'], 0, p['noise_sdev'])
            signal = signal.reshape(1, -1)
        
        else:
            signal = seqs.mz_readout(Mz, R2s, p['S0'], p['FA'] * p['B1corr'], p['TE'], p['noise_sdev'])
            signal = signal.reshape(1, -1)

        # signal dimensions (n_channels, n_times)

        # If the input is scalar, return scalar or (n_channels)
        if input_shape == ():
            if signal.shape[0] == 1:
                return signal[0,0]
            return signal[:,0]
        
        # If the input is array, return array
        
        # If n_channels=1, return shape (n_times,)
        if signal.shape[0] == 1:
            return signal[0,:]
        
        # return (n_channels, n_times)
        return signal
 


class Signal(Function):
    configs = {
        'sequence': list(SEQUENCES.keys()),
        'inflow': [False, True],
        'calibrate': [False, True],
    }
    def __init__(self, 
            sequence='3D-SPGR-SS', 
            inflow=False, 
            calibrate=False,
            defaults=None, 
            **params,
        ):
        cnfg = {
            'sequence': sequence,
            'inflow': inflow,
            'calibrate': calibrate,
        }
        self._set_config(cnfg)
        self._set_params(defaults) 
    
    def params(self):
        pars = Longitudinal(**self._cnfg).params()
        pars += Readout(**self._cnfg).params()
        derived = []
        if self._cnfg['calibrate']:
            pars += ['Sb']
            derived += ['S0']
        pars = {p for p in pars if p not in derived}
        return list(pars)
    
    def __call__(self, **params): 
        p = self._update_params(params)

        if self._cnfg['calibrate']:
            multichannel = self._cnfg['sequence'] in ['Eq-DE-EPI', 'DE-EPI'] # TODO: All signals multichannel? Or have property in SEQUENCE?

            # Baseline values
            baseline = {'S0': 1}
            for R in {'R1', 'R2', 'R2s'}:
                if R in p:
                    Rb = p[R] if np.isscalar(p[R]) else p[R][0]

                    # The baseline signal is not always a scalar constant (non steady state sequences)
                    # Extend into (nt,) array if p["Sb"] is (nc, nt) (multichannel) or (nt, ) (single channel)
                    if multichannel:
                        if p['Sb'].ndim == 2:
                            Rb = np.full(p['Sb'].shape[1], Rb)
                    else:
                        if not np.isscalar(p['Sb']):
                            Rb = np.full(p['Sb'].shape[0], Rb)
                    baseline[R] = Rb

            # Derive S0
            s_cal = _signal(self._cnfg, p | baseline)
            p['S0'] = np.mean(np.where(s_cal > 0, p["Sb"] / s_cal, 0.0))

        return _signal(self._cnfg, p)
    

def _signal(cnfg, pars):
    if 'R1' in pars:
        Mz_arr = Longitudinal(**cnfg, defaults=pars)()
    elif 'R2' in pars:
        Mz_arr = np.full_like(pars['R2'], pars['me']).reshape(1, -1)
    elif 'R2s' in pars:
        Mz_arr = np.full_like(pars['R2s'], pars['me']).reshape(1, -1)
    return Readout(**cnfg, defaults=pars)(Mz_arr)
    




def _Mz(mz_prep_sequence, R1:np.ndarray, v, Fw, j, p):
    if j is None:
        j = np.zeros_like(R1) # wasteful. Catch j=None in lib.functions
    # All library functions require shape (nc, nt)
    R1 = R1.reshape(-1, R1.shape[-1])
    j = j.reshape(R1.shape)
    me = p['me']
    
    # if mz_prep_sequence == 'Eq': 
    #     return np.full_like(R1, me)
    if mz_prep_sequence == 'IR-SS':
        return seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, p['TA'], 180)
    if mz_prep_sequence == 'SR-SS':
        return seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, p['TA'], 90)
    if mz_prep_sequence == 'PR-SS':
        return seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, p['TA'], p['PA'])
    if mz_prep_sequence == 'SPGR':
        return seqs.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], 0, p['TA'], 0) 
    if mz_prep_sequence == 'SR-SPGR':
        return seqs.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 90) 
    if mz_prep_sequence == 'IR-SPGR':
        return seqs.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 180) 
    if mz_prep_sequence == 'PR-SPGR':
        return seqs.Mz_pr_spgr(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], p['PA'])
    if mz_prep_sequence == 'SPGR-SS':
        return seqs.Mz_spgr_in_ss(R1, v, Fw, j, me, p['TR'], p['FA'] * p['B1corr'])
    if mz_prep_sequence == 'SR-SPGR-SS':
        return seqs.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 90) 
    if mz_prep_sequence == 'IR-SPGR-SS':
        return seqs.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], 180)
    if mz_prep_sequence == 'PR-SPGR-SS':
        return seqs.Mz_pr_spgr_in_ss(R1, v, Fw, j, me, p['TC'], p['TR'], p['FA'] * p['B1corr'], p['TP'], p['TA'], p['PA']) 
    if mz_prep_sequence == 'SSI':
        return seqs.Mz_ssi(R1, v, Fw, j, me, p['TR'], p['FA'] * p['B1corr'], p['TF'], p['SA'])
    if mz_prep_sequence == 'SE-SS':
        return seqs.Mz_se(R1, v, Fw, j, me, p['TE'], p['TR'], p['FA'] * p['B1corr'])
    if mz_prep_sequence == 'DE-SS':
        return seqs.Mz_se(R1, v, Fw, j, me, p['TE2'], p['TR'], p['FA'] * p['B1corr'])