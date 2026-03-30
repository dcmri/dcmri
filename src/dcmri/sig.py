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

from types import MappingProxyType

from scipy.special import i0, i1
import numpy as np

import dcmri.mz as mz
import dcmri.lexicon_utils as lexicon


class Signal:
    # TODO: Only one sequence keyword!!!
    # All systems are optn but need to include more differentiation 
    # in sequence names, e.g 3D-SS (inflow=SS) versus 2D-SR-SS (inflow=SR)
    def __init__(self, sequence='SS', inflow_sequence='SS', **params):
        if sequence not in self._params_dict():
            raise ValueError(f"Sequence {sequence} is not defined. The options are {self._params_dict().keys()}.")
        if inflow_sequence not in self._params_dict():
            raise ValueError(f"Sequence {inflow_sequence} is not defined. The options are {self._params_dict().keys()}.")

        self._cnfg = {'sequence': sequence, 'inflow_sequence': inflow_sequence}
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
        seq = self._cnfg['sequence']
        iseq = self._cnfg['inflow_sequence']
        pars = mz.Mz(seq)._params()

        # TODO: Modify - returns [] if iseq is None (for closed systems)
        pars += mz.Mz(iseq)._params() 
        pars += Readout()._params()
        return list(set(pars))
    
    def __call__(self, R1=1, R2=1, v=None, Fw=None, Fi=None, R1i=None, me=None, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        seq = self._cnfg['sequence']
        iseq = self._cnfg['inflow_sequence']

        # Inflow of magnetization
        if R1i is not None: 
            # TODO: raise Exception if inflow_sequence is None
            Fi = np.array(Fi)
            if Fi.size==1:
                j = Fi * mz.Mz(iseq)(R1i, me=me, **p)
            else:
                j = np.zeros_like(R1i)
                for i in range(Fi.size):
                    j[i,:] = Fi[i] * mz.Mz(iseq)(R1i[i,:], me=me, **p)
        else:
            j = None

        # Magnetization and readout
        magn = mz.Mz(seq)(R1, v, Fw, j, me, **p)
        return Readout()(magn, R2, **p) # TODO: REPLACE FAR as keyword


class Readout:
    def __init__(self, **params):
        self._cnfg = {}
        self._pars = lexicon.init(self._params())
        # Override parameters
        [self._pars.update({p:v}) for p, v in params.items() if p in self._params()]

    def params(self):
        return self._pars.copy()
    
    def _params(self):
        return ['S0', 'FAR', 'TE', 'noise_sdev'] # TODO: FAR -> FA!!!!!
    
    def __call__(self, Mz:np.ndarray, R2=1, **params):
        # Update keyword parameters
        if params == {}:
            p = self._pars
        else:
            p = self._pars.copy()
            [p.update({k:v}) for k, v in params.items() if k in self._pars]

        # Mz has shape 1D (nt, ) or 2D (nc, nt)
        Mz = np.array(Mz)
        if Mz.ndim==1:
            Mz_total = Mz
        else:
            Mz_total = Mz.sum(axis=0)

        sFA = np.sin(np.radians(p['FAR']))
        decay = np.exp(-p['TE'] * R2)
        Mxy = p['S0'] * decay * sFA * Mz_total
        return _signal_rice(np.abs(Mxy), p['noise_sdev'])
    

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
