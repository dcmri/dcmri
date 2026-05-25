from copy import deepcopy

import numpy as np

from dcmri.lexicon import SEQUENCES
import dcmri.inverse.lib as solve
from dcmri.core import LayerFunction
from dcmri.bloch import Signal


invertible_seqs = [s for s, v in SEQUENCES.items() if v['steady-state']]

class SignalToConc(LayerFunction):

    configs = {'sequence': deepcopy(invertible_seqs) + ['lin']}

    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(S0=None, R1i=None)

        # Overide Lexicon defaults
        if sequence in ['lin', 'DE-EPI', 'Eq-DE-EPI']: # dual weighting but also dual channel so well defined
            pass

        # Set user-defined parameters
        self._override_pars(**params)

    def _params(self):
        sequence = self._cnfg['sequence']
        pars = ['n0']
        if sequence == 'lin':
            pars += ['S0', 'R10', 'r1']
        else:
            weights = SEQUENCES[sequence]['parameters']['tissue']
            if 'R2s' in weights:
                pars += ['r2s']
            if 'R2' in weights:
                pars += ['r2']
            if 'R1' in weights:
                pars += ['R10', 'r1']
            pars += SEQUENCES[sequence]['parameters']['prep']
            pars += SEQUENCES[sequence]['parameters']['read']

        pars = list(set(pars))
        pars.sort()
        return pars

    def __call__(self, S, **params):
        p = self._update_pars(**params)
        sequence = self._cnfg['sequence']

        # Input shape is either (n_samples, n_channels, n_times) or (n_channels, n_times) or (n_times)
        # Output shapes are the same

        # Reshape S to standard form (n_samples, n_channels, n_times)
        S = np.array(S)
        if S.size <= 1:
            raise ValueError("Signal needs more than 1 time point for concentration calculation")
        
        ndim = S.ndim
        if ndim == 1:
            S = S[None, None, :]
            #S = S.reshape(1, 1, S.shape[0]) # n_samples, n_channels, n_times
        elif ndim == 2: # (n_channels, n_times)
            S = S[None, :, :]
            #S = S.reshape(1, S.shape[0], S.shape[1]) # n_samples, n_channels, n_times

        # Shape R10 -> (n_samples)
        if 'R10' in p:
            if p['R10'] is not None:
                R10 = np.atleast_1d(p['R10'])
                if R10.size == 1:
                    R10 = np.full(S.shape[0], R10[0])
                if R10.size != S.shape[0]:
                    raise ValueError('R10 must have the same number of elements as samples in S.')
                p = {k:v for k, v in p.items() if k != 'R10'} | {'R10': R10}
        
        # Shape S0 -> (n_samples) - same for each channel
        if p['S0'] is not None:
            S0 = np.atleast_1d(p['S0'])
            if S0.size == 1:
                S0 = np.full(S.shape[0], S0[0])
            if S0.size != S.shape[0]:
                raise ValueError('S0 must have the same number of elements as samples in S.')
            p = {k:v for k, v in p.items() if k != 'S0'} | {'S0': S0}

        # Delegate computation to specialised functions
        if sequence == 'lin':
            conc = solve.conc_dce_lin(S[:,0,:], p['n0'], p['R10'], p['S0'], p['r1'])
            conc = conc[:, None, :]

        elif sequence in ['3D-SPGR-SS', 'ZTE-3D-SPGR-SS']:
            p['TE'] = 0
            conc = solve.conc_ss(S[:,0,:], **p)  
            conc = conc[:, None, :]

        elif sequence in ['Eq-GE-EPI', 'GE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], p['n0'], p['r2s'], p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['Eq-SE-EPI', 'SE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], p['n0'], p['r2'], p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['Eq-DE-EPI', 'DE-EPI']:
            S_GE, S_SE = S[:,0,:], S[:,1,:]
            conc_ge = solve.conc_dsc(S_GE, p['n0'], p['r2s'], p['TE1'])
            conc_se = solve.conc_dsc(S_SE, p['n0'], p['r2'], p['TE2'])
            conc_ge = conc_ge[:, None, :]
            conc_se = conc_se[:, None, :]
            conc = np.concatenate((conc_ge, conc_se), axis=1)

        else:
            Sn_model = Signal(sequence, **p)
            conc = solve.conc_dce(Sn_model, S[:,0,:], **p)
            conc = conc[:, None, :]

        if ndim==1:
            return conc[0,0,:]
        elif ndim==2:
            return conc[0,:,:]
        else:
            return conc

