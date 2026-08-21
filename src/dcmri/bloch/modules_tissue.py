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
import numpy as np
from scipy.interpolate import interp1d

from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.bloch import functions_dynamic
from dcmri.bloch import functions_sequences


# class JzPrep(Module)


class MzPrep(Module): 
    configs = {
        'sequence': set(SEQUENCES.keys()),
        'inflow': {False, True},
    }
    defaults = {
        'sequence': 'SPGR-SS',
        'inflow': False,
    }
    def inputs(self):
        seq = self.config['sequence']
        # Sequence parameters
        inputs = set(SEQUENCES[seq]['parameters']['prep'])
        # Tissue parameters
        weighting = SEQUENCES[seq]['parameters']['tissue']
        inputs |= {'tR'}
        if 'R1' in weighting:
            inputs |= {'R1'}
        inputs |= {'v', 'Fw', 'me'}
        if self.config['inflow']:
            inputs |= {'Fi', 'R1i'}
        return inputs
    
    def outputs(self):
        return {'tM', 'Mz'} # (compartments, times)
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        sequence = self.config['sequence']

        # --- Reshape vw to (nc, )
        v = np.atleast_1d(p['v'])
        nc = v.size # -- The number of compartments is decided by the size of v

        # --- Reshape Fw to (nc, nc)
        Fw = np.atleast_1d(p['Fw'])
        if nc > 1:
            if Fw.size==1:
                Fw = np.full((nc, nc), Fw[0])
                np.fill_diagonal(Fw, 0)
        if Fw.size != nc * nc:
            raise ValueError("For an n-compartment tissue, Fw must have shape (n, n).")
        Fw = Fw.reshape(nc, nc)

        # --- Reshape R1 to (nc, nt)     
        if 'R1' not in p: # Assume full recovery between pulses (T1 = 0)
            nt = np.size(p['tR'])
            R1 = np.full((nc, nt), np.inf)
        else:
            try:
                R1 = np.reshape(p['R1'], (nc, -1))  
            except:
                raise ValueError(f"For a tissue with {nc} compartments and nt times, R1 must have shape ({nc}, nt).")
            nt = R1.shape[-1]

        # --- Reshape tR1 to (nt, )
        tR = np.atleast_1d(p['tR'])

        # Compute magnetization inflow
        j, tj = None, None
        if self.config['inflow']:

            # Format R1i
            if 'R1i' not in p: # Assume full recovery between pulses (T1 = 0)
                R1i = np.full((nc, nt), np.inf)
            else:
                try:
                    R1i = np.reshape(p['R1i'], (nc, nt))  
                except:
                    raise ValueError(f'R1i ({p['R1i'].size}) must have the same size as R1 ({nc * nt}) ') 

            # Format Fi
            Fi = np.array(p['Fi'])
            if Fi.size != nc:
                raise ValueError(f"Fi must have the same number of elements as the first dimension of R1i. Fi has {Fi.size} elements and R1i has shape {R1i.shape}.")
            Fi = Fi.reshape(nc)

            # Compute j for each compartment
            mz_prep_inflow = SEQUENCES[sequence]['mz_prep_inflow']

            for i in range(Fi.size):
                if not np.isnan(Fi[i]):
                    vi, Fwi, ji = 1, 0, None # inflow = 1 closed compartment
                    tj, Mzi = _Mz(sequence, mz_prep_inflow, tR, R1i[i], vi, Fwi, ji, p)
                    if j is None:
                        j = np.zeros((nc, ) + tj.shape)
                    j[i, :, :] = Fi[i] * Mzi[0, :, :]  # (mL/min/cm3) * (magn/mL) = magn/min/cm3

        # Delegate computation to helper functions
        mz_prep_sequence = SEQUENCES[sequence]['mz_prep_tissue']
        tM, Mz = _Mz(sequence, mz_prep_sequence, tR, R1, v, Fw, j, p, tj)

        # Return dimensions (compartments, times)
        results = {'tM': tM, 'Mz': Mz}
        return self.map_results(results)


def _Mz(sequence, mz_prep_sequence, tR1, R1, v, Fw, j, p, tj=None):

    # Catch the scalar case
    if R1.ndim == 1:
        R1 = R1.reshape(1, -1)
        v = np.full(1, v,)
        Fw = np.full((1, 1), Fw)
        if j is not None:
            j = j.reshape(1, -1)

    if mz_prep_sequence == 'Eq':
        Mz = np.full(R1.shape + (1, ), p['me'])
        return tR1.reshape((tR1.size, 1)), Mz
    if mz_prep_sequence == 'IR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return functions_dynamic.Mz_dyn_spgr_ss(tR1, R1, v, Fw, j, p['me'], TA, 180, 1, tj=tj)
    if mz_prep_sequence == 'SR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return functions_dynamic.Mz_dyn_spgr_ss(tR1, R1, v, Fw, j, p['me'], TA, 90, 1, tj=tj)
    if mz_prep_sequence == 'PR-SS':
        TA = functions_sequences.repetition_time(sequence, p)
        return functions_dynamic.Mz_dyn_spgr_ss(tR1, R1, v, Fw, j, p['me'], TA, p['PA'], 1, tj=tj)
    if mz_prep_sequence == 'SPGR':
        return functions_dynamic.Mz_dyn_spgr(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], tj=tj) 
    if mz_prep_sequence == 'SR-SPGR':
        _check_TP(p['TP'])
        t0 = p['iz'] * (p['TP'] + p['Nph'] * p['TR'] + p['TD']) if sequence == '2D-SR-SPGR' else 0
        return functions_dynamic.Mz_dyn_pr_spgr(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], 90, t0=t0, tj=tj) 
    if mz_prep_sequence == 'IR-SPGR':
        _check_TP(p['TP'])
        return functions_dynamic.Mz_dyn_pr_spgr(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], 180, tj=tj) 
    if mz_prep_sequence == 'PR-SPGR':
        _check_TP(p['TP'])
        return functions_dynamic.Mz_dyn_pr_spgr(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], p['PA'], tj=tj)
    if mz_prep_sequence == 'SPGR-SS':
        return functions_dynamic.Mz_dyn_spgr_ss(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], tj=tj)
    if mz_prep_sequence == 'SR-SPGR-SS':
        _check_TP(p['TP'])
        return functions_dynamic.Mz_dyn_pr_spgr_ss(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], 90, tj=tj) 
    if mz_prep_sequence == 'IR-SPGR-SS':
        _check_TP(p['TP'])
        return functions_dynamic.Mz_dyn_pr_spgr_ss(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], 180, tj=tj)
    if mz_prep_sequence == 'PR-SPGR-SS':
        _check_TP(p['TP'])
        return functions_dynamic.Mz_dyn_pr_spgr_ss(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TP'], p['TD'], p['PA'], tj=tj) 
    if mz_prep_sequence == 'SSI':
        return functions_dynamic.Mz_dyn_spgr_ssi(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], p['Nph'], p['TF'], p['SA'])
    if mz_prep_sequence == 'GE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-GE-EPI' else 0
        return functions_dynamic.Mz_dyn_spgr_ss(tR1, R1, v, Fw, j, p['me'], p['TR'], p['FA'] * p['B1corr'], 1, t0=t0, tj=tj)
    if mz_prep_sequence == 'SE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-SE-EPI' else 0
        return functions_dynamic.Mz_dyn_se(tR1, R1, v, Fw, j, p['me'], p['TE'], p['TR'], p['FA'] * p['B1corr'], t0=t0, tj=tj)
    if mz_prep_sequence == 'DE-SS':
        t0 = p['iz'] * p['TR'] / p['Nz'] if sequence == '2D-DE-EPI' else 0
        return functions_dynamic.Mz_dyn_se(tR1, R1, v, Fw, j, p['me'], p['TE2'], p['TR'], p['FA'] * p['B1corr'], t0=t0, tj=tj)

def _check_TP(TP):
    if TP==0:
        raise ValueError("The delay time (TP) after a preparation pulse must be greater than 0.")

class MxyReadMz(Module): 
    configs = {
        'sequence': set(SEQUENCES.keys()),
    }
    defaults = {
        'sequence': '3D-SPGR-SS'
    }    
    def inputs(self):
        params = SEQUENCES[self.config['sequence']]['parameters']
        # Tissue parameters
        weighting = params['tissue']
        # Sequence parameters
        inputs = {'tR', 'tM', 'Mz'} # shape (nc, n_times) 
        inputs |= set(params['read'])
        if 'R2s' in weighting:
            inputs |= {'R2s'}
        if 'R2' in weighting:
            inputs |= {'R2'}
        return inputs
    
    def outputs(self):
        # (components, compartments, times) 
        # or 
        # (channels, components, compartments, times)
        return {'Mxy'}

    def lexicon_data(self, qvalues):
        nc, ntR, ntM = 2, 5, 3
        p = {
            'tM': np.ones(ntM), 
            'Mz': np.ones((nc, ntM)), 
            'tR': np.arange(ntR), 
            'R2':np.ones((nc, ntR)), 
            'R2s':np.ones(ntR),
        }
        return self.update_data(p)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        seq = self.config['sequence']

        # Inputs:
        # Mz (compartments, acq times)
        # R2 (compartments, sim times)
        # R2s (sim times, )

        # Output Mxy (channels, components, compartments, acq times)

        Mz = p['Mz'] # (ncomps, ntimes)
        nc, nt = Mz.shape

        if 'R2s' in self._inputs:
            R2s = np.interp(p['tM'], np.atleast_1d(p['tR']), np.atleast_1d(p['R2s']))
            
        if 'R2' in self._inputs:
            R2 = np.reshape(p['R2'], (nc, -1))  
            R2 = _interpolate_2d(p['tM'], p['tR'], R2)

        FA = p['FA'] * p['B1corr']
        
        if seq in ['2D-SE-EPI', '3D-SE-EPI']:
            Mxy = np.zeros((1, 2, nc, nt), dtype=float) # (channels, components, compartments, times)
            for c in range(nc):
                Mxy[0, 0, c, :] = functions_sequences.mz_readout(Mz[c, :], R2[c, :], FA, p['TE'])
        
        elif seq in ['2D-DE-EPI', '3D-DE-EPI']:
            Mxy = np.zeros((2, 2, nc, nt), dtype=float)
            for c in range(nc):
                Mxy[0, 0, c, :] = functions_sequences.mz_readout(Mz[c, :], R2s, FA, p['TE1'])
                Mxy[1, 0, c, :] = functions_sequences.mz_readout(Mz[c, :], R2[c, :], FA, p['TE2'])

        elif seq in ['ZTE-3D-SPGR-SS', 'ZTE-3D-IR-SPGR-SS']:
            Mxy = np.zeros((1, 2, nc, nt), dtype=float)
            R2s = np.zeros(nt)
            for c in range(nc):
                Mxy[0, 0, c, :] = functions_sequences.mz_readout(Mz[c, :], R2s, FA, 0)
        
        else:
            Mxy = np.zeros((1, 2, nc, nt), dtype=float) 
            for c in range(nc):
                Mxy[0, 0, c, :] = functions_sequences.mz_readout(Mz[c, :], R2s, FA, p['TE'])

        # (channels, components, compartments, times)
        # or
        # (components, compartments, times)

        results = {'Mxy': Mxy}
        return self.map_results(results)


def _interpolate_2d(t_new, tR, R):
    if np.size(tR)==1:
        return np.tile(R, (1, np.size(t_new)))
    
    f = interp1d(tR, R, axis=1, kind='linear')
    return f(t_new) 


class Magnetization(Module): 
    configs = {
        'sequence': set(SEQUENCES.keys()),
        'inflow': {False, True},
    }
    defaults = {
        'sequence': '3D-SPGR-SS',
        'inflow': False,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._mz_prep = MzPrep(**self.config)
        self._mxy_read = MxyReadMz(**self.config)
        self.map_io(imap, omap)
        
    def inputs(self):
        inputs = self._mz_prep.mapped_inputs()
        inputs |= self._mxy_read.mapped_inputs() - {'tM', 'Mz'}
        return inputs 
    
    def outputs(self):
        return {'tM', 'M'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # All current sequences use an Mz prep followed by a readout
        # This could be generalized in the future to sequences that have mixed T1/T2 prep

        # Currently not considering spatial encoding so only need the center line
        k0 = functions_sequences.pulse_readout(self.config['sequence'], p)

        p |= self._mz_prep(p)
        p['Mz'] = p['Mz'][:, :, k0] # (compartments, times)
        p['tM'] = p['tM'][:, k0] # (times, )
        Mxy = self._mxy_read(p)['Mxy'] # (channels, components, compartments, times) 

        M = np.zeros((Mxy.shape[0], 3, Mxy.shape[2], Mxy.shape[3]), dtype=Mxy.dtype)
        M[:, :2, :, :] = Mxy
        for c in range(M.shape[0]):
            M[c, 2, :, :] = p['Mz']

        results = {'tM': p['tM'], 'M': M}
        return self.map_results(results)