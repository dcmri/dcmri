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

    >>> Mz = dc.Mz_wrapper_free(R1, TI, n_init=-1)
    >>> Mz_e = dc.Mz_wrapper_free(R1, TI, n_init=-1, Fw=f, j=f)
    >>> Mz_i = dc.Mz_wrapper_free(R1, TI, n_init=-1, Fw=f, j=-f)

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
    >>> Mz = dc.Mz_wrapper_free(R1, TI, v, Fw, n_init=-1, j=[f, 0])

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
    >>> Mz = dc.Mz_wrapper_free(R1, TI, v, Fw, n_init=-1, j=j)

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
    >>> Mz = dc.Mz_wrapper_free(R1, TI, v, Fw, n_init=-1, j=j)

    >>> plt.plot(t, Mz[0,:,5], label='Central compartment', linewidth=3)
    >>> plt.plot(t, Mz[1,:,5], label='Peripheral compartment', linewidth=3)
    >>> plt.xlabel('Time (sec)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()      

"""
import numpy as np
from scipy.interpolate import interp1d

from dcmri.core.tools import get_sequence
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.core.module import Module
from dcmri.bloch.functions_dynamic import Mz_wrapper
from dcmri.bloch import functions_sequences


# TODO class JzPrep(Module)


class MzPrep(Module): 
    configs = {
        'sequence': get_sequence('name'),
        'tof_corr': {False, True},
        'inflow': {False, True},
    }
    defaults = {
        'sequence': 'SPGR-SS',
        'tof_corr': False,
        'inflow': False,
    }
    def __init__(self, imap: dict=None, omap: dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        if self.config['tof_corr'] and self.config['sequence'] != '3D-SPGR-SS':
            raise InvalidConfiguration(f"Time-of-flight correction is only available for sequence 3D-SPGR-SS. You are running sequence {self.config['sequence']}. Either choose tof_corr=False or sequence='3D-SPGR-SS'.")
        self.map_io(imap, omap, iomap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        # Output
        o = {}

        # --- Reshape v to (nc, nc)
        v = np.atleast_1d(p['vw'])
        nc = v.size # -- The number of compartments is decided by the size of v

        # Start and end of acquisition
        tstart = p['tstart'] 
        t_end = tstart + p['tacq']

        # Apply TOF correction if requested
        if self.config['tof_corr']:
            sequence = '3D-SPGR-SSI'
        else:
            sequence = self.config['sequence']

        # --- Result without T1 weighting
        if 'R1' not in get_sequence('tissue_params', sequence):
            o['tM'] = tstart + functions_sequences.acquisition_times(sequence, p, p['tacq'])
            ntM = len(o['tM'])
            o['Mz'] = np.repeat(v[:, None] * p['me'], ntM, axis=1)
            return self.map_results(o)
    
        # --- Reshape Kw to (nc, nc)
        Kw = np.atleast_1d(p['Kw'])
        if nc > 1:
            if Kw.size==1:
                Kw = np.full((nc, nc), Kw[0])
                np.fill_diagonal(Kw, 0)
        if Kw.size != nc * nc:
            raise ValueError("For an n-compartment tissue, Kw must have shape (n, n).")
        Kw = Kw.reshape(nc, nc)

        # --- Reshape tR1 and R1 to (nc, nt)   
        tR = np.atleast_1d(p['tR'])  
        ntR = len(tR)
        R1 = np.reshape(p['R1'], (nc, ntR))

        # Compute magnetization inflow
        j, tj = None, None
        if self.config['inflow']:

            # Format R1i
            ni = len(p['inlets'])
            R1i = np.reshape(p['R1i'], (ni, ntR)) 

            # Format Fi
            Fwi = np.array(p['Fwi'])
            if Fwi.size != ni:
                raise ValueError(f"Fwi must have the length {ni}")
            Fwi = Fwi.reshape(ni)

            # Compute j for each compartment
            mz_prep_inflow = get_sequence('mz_prep_inflow', sequence)

            for i in range(ni):
                tj, Mzi = Mz_wrapper(sequence, mz_prep_inflow, tR, R1i[i], p, v=1, Kw=0, tstart=tstart, t_end=t_end)
                if j is None:
                    j = np.zeros((nc, ) + tj.shape)
                inlet = p['inlets'][i] 
                j[inlet, :, :] = Fwi[i] * Mzi[0, :, :]  # (mL/min/cm3) * (magn/mL) = magn/min/cm3

        # Delegate computation to helper functions
        mz_prep_sequence = get_sequence('mz_prep_tissue', sequence)

        o['tM'], o['Mz'] = Mz_wrapper(sequence, mz_prep_sequence, tR, R1, p, v, Kw, tj, j, tstart=tstart, t_end=t_end)

        # Return dimensions (compartments, times)
        return self.map_results(o)

    def inputs(self):
        inputs = {'me', 'vw', 'tstart', 'tacq'}

        if self.config['tof_corr']:
            sequence = '3D-SPGR-SSI'
        else:
            sequence = self.config['sequence']

        inputs |= get_sequence('prep_params', sequence)
        if 'R1' not in get_sequence('tissue_params', sequence):
            return inputs
        
        inputs |= {'Kw', 'tR', 'R1'}
        if self.config['inflow']:
            inputs |= {'Fwi', 'R1i', 'inlets'}
        return inputs
    
    def outputs(self):
        return {'tM', 'Mz'} # (compartments, times)

    def dummy_data(self, nc=2):
        data = self.init_data()
        ntR = 5
        data |= {
            'tacq': ntR-1,
            'tR': np.arange(ntR),
            'R1': np.ones((nc, ntR)),
            'vw': np.ones(nc) / nc, 
            'Kw': np.ones((nc, nc)),
            'R1i': np.ones((nc, ntR)),
            'Fwi': np.ones(nc),
            'inlets': np.arange(nc),
        }
        return data



class MxyReadMz(Module): 
    configs = {
        'sequence': get_sequence('name'),
    }
    defaults = {
        'sequence': '3D-SPGR-SS'
    }    
    def inputs(self):
        inputs = {'tR', 'tM', 'Mz'} # shape (nc, n_times) 
        inputs |= get_sequence('read_params', self.config['sequence'])

        weighting = get_sequence('tissue_params', self.config['sequence'])
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

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)
        seq = self.config['sequence']

        # Inputs:
        # Mz (compartments, acq times)
        # R2 (compartments, sim times)
        # R2s (sim times, )

        # Output Mxy (channels, components, compartments, acq times)

        Mz = np.atleast_2d(p['Mz']) # (ncomps, ntimes)
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

    def dummy_data(self, nc=2):
        data = self.init_data()
        ntR, ntM = 5, 3
        data |= {
            'tacq': ntR-1,
            'tM': np.ones(ntM), 
            'Mz': np.ones((nc, ntM)), 
            'tR': np.arange(ntR), 
            'R2':np.ones((nc, ntR)), 
            'R2s':np.ones(ntR),
        }
        return data


def _interpolate_2d(t_new, tR, R):
    if np.size(tR)==1:
        return np.tile(R, (1, np.size(t_new)))
    
    f = interp1d(tR, R, axis=1, kind='linear')
    return f(t_new) 


class Magnetization(Module): 
    configs = {
        'sequence': get_sequence('name'),
        'tof_corr': {False, True},
        'inflow': {False, True},
    }
    defaults = {
        'sequence': '3D-SPGR-SS',
        'tof_corr': False,
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

    def dummy_data(self, nc=2):
        data = self.init_data()
        ntR = 5
        data |= {
            'tacq': ntR-1,
            'tR': np.arange(ntR),
            'R1': np.ones((nc, ntR)),
            'R2':np.ones((nc, ntR)), 
            'R2s':np.ones(ntR),
            'vw': np.ones(nc) / nc, 
            'Kw': np.ones((nc, nc)),
            'R1i': np.ones((nc, ntR)),
            'Fwi': np.ones(nc),
            'inlets': [0, 1]
        }
        return data