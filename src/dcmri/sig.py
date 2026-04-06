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

from copy import deepcopy

from scipy.special import i0, i1
import numpy as np

import dcmri.mz as mz
from dcmri.ui import SuperFunc
from dcmri.lexicon import SEQUENCES


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
            mz_inflow = mz.Mz(inflow_mz_sequence, **pi)

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
            Mz = mz.Mz(tissue_mz_sequence, **p)(p['R1'], j)
        return Readout(sequence, **p)(Mz=Mz, R2=p['R2'], R2s=p['R2s'])


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
