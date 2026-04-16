from copy import deepcopy

import numpy as np

from dcmri.lexicon import SEQUENCES
import dcmri.inverse.lib as solve
from dcmri.core import SuperFunc
from dcmri.bloch import Signal



class SignalToConc(SuperFunc):

    configs = {'sequence': deepcopy(list(SEQUENCES.keys())) + ['lin']}

    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(S0=None, R1i=None)

        # Overide Lexicon defaults
        if sequence == 'lin':
            pass
        elif SEQUENCES[sequence]['type'] == 'DCE':
            self._pars['TE'] = 0 # This only works in the absence of T2-weighting

        # Set user-defined parameters
        self._override_pars(**params)

    def _params(self):
        sequence = self._cnfg['sequence']
        if sequence == 'lin':
            pars = ['S0']
        else:
            pars = SEQUENCES[sequence]['parameters']['prep']
            pars += SEQUENCES[sequence]['parameters']['read']

        pars += ['n0']

        if sequence == 'GE-EPI':
            pars += ['r2s']
        elif sequence == 'SE-EPI':
            pars += ['r2']
        elif sequence == 'DE-EPI':
            pars += ['r2', 'r2s']
        elif sequence == 'lin':
            pars += ['R10', 'r1']
        elif SEQUENCES[sequence]['type'] == 'DCE':
            pars += ['R10', 'r1', 'R20s']

        pars = list(set(pars))
        pars.sort()
        return pars

    def __call__(self, S, **params):
        p = self._update_pars(**params)
        sequence = self._cnfg['sequence']

        # Check if sequence is invertible
        if sequence != 'lin':
            if not SEQUENCES[sequence]['steady-state']:
                raise ValueError(
                    "Only steady-state sequences can be directly inverted. If you want to "
                    "use this function on a non-steady-state sequence, make sure to include "
                    "some dummy pulses in the signal and then invert using the steady-state signal model."
                )

        # Shape S to standard form (n_samples, n_times) or (n_samples, n_times, n_signals)
        S = np.array(S)
        input_shape = S.shape
        if S.size <= 1:
            raise ValueError("Signal needs more than 1 time point for concentration calculation")
        if S.ndim == 1:
            S = S.reshape(1, -1) # n_samples, n_times
        if sequence == 'DE-EPI': # Shape either (n_samples, n_channels, n_times) or (n_channels, n_times)
            if S.ndim==2: # (n_channels, n_times)
                nt = S.shape[1]
                S = S.reshape(-1, 2, nt) # n_samples, n_channels, n_times

        # Shape R10
        if 'R10' in p:
            if p['R10'] is not None:
                R10 = np.atleast_1d(p['R10'])
                if R10.size == 1:
                    R10 = np.full(S.shape[0], R10[0])
                if R10.size != S.shape[0]:
                    raise ValueError('R10 must have the same number of elements as samples in S.')
                p = {k:v for k, v in p.items() if k != 'R10'} | {'R10': R10}
        
        # Shape S0
        if p['S0'] is not None:
            S0 = np.atleast_1d(p['S0'])
            if S0.size == 1:
                S0 = np.full(S.shape[0], S0[0])
            if S0.size != S.shape[0]:
                raise ValueError('S0 must have the same number of elements as samples in S.')
            p = {k:v for k, v in p.items() if k != 'S0'} | {'S0': S0}

        # Delegate computation to specialised functions
        if sequence == 'lin':
            conc = solve.conc_dce_lin(S, p['n0'], p['R10'], p['S0'], p['r1'])

        elif sequence == '3D-SPGR-SS':
            conc = solve.conc_ss(S, **p)  

        elif SEQUENCES[sequence]['type'] == 'DCE':
            Sn_model = Signal(sequence, **p)
            conc = solve.conc_dce(Sn_model, S, **p) 

        elif sequence == 'GE-EPI':
            conc = solve.conc_dsc(S, p['n0'], p['r2s'], p['TE'])

        elif sequence == 'SE-EPI':
            conc = solve.conc_dsc(S, p['n0'], p['r2'], p['TE'])

        elif sequence == 'DE-EPI':
            S_GE, S_SE = S[:,0,:], S[:,1,:]
            conc_ge = solve.conc_dsc(S_GE, p['n0'], p['r2s'], p['TE1'])
            conc_se = solve.conc_dsc(S_SE, p['n0'], p['r2'], p['TE2'])
            conc = np.stack((conc_ge, conc_se))

        return conc.reshape(input_shape)

