from copy import deepcopy

import numpy as np

from dcmri.lexicon import SEQUENCES
from dcmri import sig
from dcmri.func import SuperFunc



class SignalToConc(SuperFunc):

    configs = {'sequence': deepcopy(list(SEQUENCES.keys())) + ['lin']}

    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(S0=None, R1i=None)

        # Overide Lexicon defaults
        if sequence == 'lin':
            pass
        elif SEQUENCES[sequence]['type'] == 'DCE':
            self._pars['TE'] = 0 # This only works in the absense of T2-weighting

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
            conc = _conc_dce_lin(S, p['n0'], p['R10'], p['S0'], p['r1'])

        elif sequence == '3D-SPGR-SS':
            conc = _conc_ss(S, **p)  

        elif SEQUENCES[sequence]['type'] == 'DCE':
            conc = _conc_dce_lookup(sequence, S, **p) 

        elif sequence == 'GE-EPI':
            conc = _conc_dsc(S, p['n0'], p['r2s'], p['TE'])

        elif sequence == 'SE-EPI':
            conc = _conc_dsc(S, p['n0'], p['r2'], p['TE'])

        elif sequence == 'DE-EPI':
            S_GE, S_SE = S[:,0,:], S[:,1,:]
            conc_ge = _conc_dsc(S_GE, p['n0'], p['r2s'], p['TE1'])
            conc_se = _conc_dsc(S_SE, p['n0'], p['r2'], p['TE2'])
            conc = np.stack((conc_ge, conc_se))

        return conc.reshape(input_shape)


def _conc_dce_lookup(sequence, S, n0=None, R10=None, S0=None, r1=None, R20s=None, **params):
    Sn_model = sig.Signal(sequence, **params)

    # TE does not affect the concentration
    TE = 0 if R20s is None else params['TE']

    #Normalize signal
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        Sn0 = Sn_model(R1=R10, TE=TE, R2s=R20s, S0=1, v=1, Fw=0, me=1, R1i=None, Fi=None) # Baseline R20 absorbed in S0
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn_data = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Create lookup table
    c_step = 0.01 * 1e-3
    c_max = 0.01
    c_range = np.arange(0, c_max, c_step)
    R1_min = 0
    R1_lookup = R1_min + r1 * c_range
    Sn_lookup = Sn_model(R1=R1_lookup, TE=TE, R2s=R20s, S0=1, v=1, Fw=0, me=1, R1i=None, Fi=None)

    # # Check that lookup values are strictly increasing
    # if not np.all(np.diff(Sn_lookup) > 0):
    #     raise ValueError(
    #         f"Cannot convert signal to concentration directly. "
    #         "The signal values are not monotonously increasing for the given concentration range. \n"
    #         "You can avoid direct inversion by fitting straight to signal data."
    #     )

    # Look up R1 values
    R1 = np.interp(Sn_data, Sn_lookup, R1_lookup)
    
    # Convert to concentration
    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1


def _conc_dsc(S, n0, r2, TE) -> np.ndarray:
    # S/Sb = exp(-TE(R2-R2b))
    #   ln(S/Sb) = -TE(R2-R2b)
    #   R2-R2b = -ln(S/Sb)/TE
    # R2 = R2b + r2C
    #   C = (R2-R2b)/r2
    #   C = -ln(S/Sb)/TE/r2

    # 1. Calculate Sb (the baseline)
    Sb = np.mean(S[:, :n0], axis=1)[:, np.newaxis]
    # Reshape Sb to (n_samples, 1) to divide S (n_samples, n_times)
    S_normalized = np.divide(S, Sb, out=np.zeros_like(S, dtype=float), where=Sb != 0)

    # 2. Calculate Concentration
    S_safe = np.clip(S_normalized, 1e-10, None)
    C = -np.log(S_safe) / (TE * r2)
    return C
    

def _conc_ss(S, n0=None, R10=None, S0=None, r1=None, R20s=None, **p) -> np.ndarray:
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R10)) / (1-cFA*exp(-TR*R10))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn
    Sn = sig.Signal('3D-SPGR-SS', **p)

    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        Sn0 = Sn(R1=R10, R2s=R20s, S0=1)
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Invert analytically
    FA = np.radians(p['FA'] * p['B1corr'])
    cFA = np.cos(FA)
    sFA = np.sin(FA)
    Sn = Sn / sFA
    En = (1 - Sn) / (1 - cFA * Sn)
    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.where(En <= 0, 0, -np.log(En)/p['TR'])

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1


def _conc_dce_lin(S, n0, R10, S0, r1):
    # S = S0 * R1
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        S0 = Sb / R10
        S0 = np.divide(Sb, R10, out=np.zeros_like(Sb, dtype=float), where=R10 > 0)

    S0 = S0[:, np.newaxis] 
    R1 = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1