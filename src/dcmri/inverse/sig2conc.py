from copy import deepcopy

import numpy as np

from dcmri.core.sequences import SEQUENCES
import dcmri.inverse.lib as solve
from dcmri.core.function import Function
from dcmri.bloch.tissue import Signal


invertible_seqs = [s for s, v in SEQUENCES.items() if v['steady-state']]

class SignalToConc(Function):
    configs = {
        'sequence': list(invertible_seqs + ['lin']),
        'inflow': [False, True],
        'calibrate': [False, True],
    }
    def __init__(
            self, 
            sequence='3D-SPGR-SS', 
            inflow=False, 
            calibrate=True,
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
        sequence = self._cnfg['sequence']

        if sequence in ['3D-SPGR-SS']:
            pars = ['r1', 'FA', 'TR', 'B1corr', 'n0']
            if self._cnfg['calibrate']:
                pars += ['R1b']  
            else:
                pars += ['S0']  

        elif sequence in ['ZTE-3D-SPGR-SS']:
            pars = ['r1', 'FA', 'TR', 'B1corr', 'n0']
            if self._cnfg['calibrate']:
                pars += ['R1b']  
            else:
                pars += ['S0'] 

        elif sequence == 'lin':
            pars = ['r1', 'n0']
            if self._cnfg['calibrate']:
                pars += ['R1b']  
            else:
                pars += ['S0'] 

        elif sequence in ['Eq-GE-EPI', 'GE-EPI']:
            pars = ['n0', 'r2s', 'TE']

        elif sequence in ['Eq-SE-EPI', 'SE-EPI']:
            pars = ['n0', 'r2', 'TE']

        elif sequence in ['Eq-DE-EPI', 'DE-EPI']:
            pars = ['n0', 'r2', 'r2s', 'TE1', 'TE2']

        else:
            derived = ['R1', 'R2s', 'TE', 'v', 'Fw', 'me']
            pars = Signal(sequence).params()
            pars += ['n0', 'r1']
            if self._cnfg['calibrate']:
                pars += ['R1b'] 
                derived += ['S0']
            else:
                pars += ['S0'] 
            pars = {p for p in pars if p not in derived}

        pars = list(set(pars))
        pars.sort()
        return pars

    def __call__(self, S, **params):
        # Input shape is either (n_samples, n_channels, n_times) or (n_channels, n_times) or (n_times)
        # Output shapes are the same
        p = self._update_params(params)
        
        # Check input
        S = np.array(S)
        if S.size <= 1:
            raise ValueError("Signal needs more than 1 time point for concentration calculation")
        
        # Reshape S to standard form (n_samples, n_channels, n_times)
        ndim = S.ndim
        if ndim == 1:
            S = S[None, None, :]
            #S = S.reshape(1, 1, S.shape[0]) # n_samples, n_channels, n_times
        elif ndim == 2: # (n_channels, n_times)
            S = S[None, :, :]
            #S = S.reshape(1, S.shape[0], S.shape[1]) # n_samples, n_channels, n_times

        # Shape R1b -> (n_samples)
        if 'R1b' in p:
            R1b = np.atleast_1d(p['R1b'])
            if R1b.size == 1:
                R1b = np.full(S.shape[0], R1b[0])
            if R1b.size != S.shape[0]:
                raise ValueError('R1b must have the same number of elements as samples in S.')
            p['R1b'] = R1b
        
        # Shape S0 -> (n_samples) - same for each channel
        if 'S0' in p:
            S0 = np.atleast_1d(p['S0'])
            if S0.size == 1:
                S0 = np.full(S.shape[0], S0[0])
            if S0.size != S.shape[0]:
                raise ValueError('S0 must have the same number of elements as samples in S.')
            p['S0'] = S0

        # Delegate computation to specialised functions
        sequence = self._cnfg['sequence']

        if sequence in ['3D-SPGR-SS']:
            conc = solve.conc_ss(S[:,0,:], **p)
            conc = conc[:, None, :]

        elif sequence in ['ZTE-3D-SPGR-SS']:
            conc = solve.conc_ss(S[:,0,:], **p)  
            conc = conc[:, None, :]

        elif sequence == 'lin':
            conc = solve.conc_dce_lin(S[:,0,:], **p)
            conc = conc[:, None, :]

        elif sequence in ['Eq-GE-EPI', 'GE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], n0=p['n0'], r2=p['r2s'], TE=p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['Eq-SE-EPI', 'SE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], n0=p['n0'], r2=p['r2'], TE=p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['Eq-DE-EPI', 'DE-EPI']:
            S_GE, S_SE = S[:,0,:], S[:,1,:]
            conc_ge = solve.conc_dsc(S_GE, n0=p['n0'], r2=p['r2s'], TE=p['TE1'])
            conc_se = solve.conc_dsc(S_SE, n0=p['n0'], r2=p['r2'], TE=p['TE2'])
            conc_ge = conc_ge[:, None, :]
            conc_se = conc_se[:, None, :]
            conc = np.concatenate((conc_ge, conc_se), axis=1)

        else:
            Sn_model = Signal(sequence, defaults=p)
            if self._cnfg['calibrate']:
                conc = solve.conc_dce(Sn_model, S[:,0,:], r1=p['r1'], n0=p['n0'], R1b=p['R1b'])
            else:
                conc = solve.conc_dce(Sn_model, S[:,0,:], r1=p['r1'], n0=p['n0'], S0=p['S0'])
            conc = conc[:, None, :]

        if ndim==1:
            return conc[0,0,:]
        elif ndim==2:
            return conc[0,:,:]
        else:
            return conc

