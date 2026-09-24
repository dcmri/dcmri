
import numpy as np
from scipy.interpolate import interp1d

from dcmri.core.tools import get_sequence
from dcmri.core.module import InvalidConfig
from dcmri.core.module import Module
from dcmri.bloch.functions_dynamic import Mz_wrapper_k0, Mz_wrapper_k_all
from dcmri.bloch import functions_sequences

# TODO in dummy_data use submodule methods to avoid repetition

# +--------------------------------------------------------------------------------------------------+
# |                                   MzPrep - all configs (n = 3)                                   |
# +----------+--------------------------------------------------------------------------+------------+
# | Key      | Values                                                                   | Default    |
# +----------+--------------------------------------------------------------------------+------------+
# | sequence | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,        | 3D-SPGR-SS |
# |          | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR,   |            |
# |          | 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR,     |            |
# |          | 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS               |            |
# | tof_corr | False, True                                                              | False      |
# | inflow   | inlet, none, pool                                                        | none       |
# +--------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------+
# |                                                 MzPrep - all inputs (n = 26)                                                 |
# +--------+------------+-----------------------------------------------+-----------------+-------+--------------+-------+-------+
# | Key    | Unit       | Name                                          | Group           | Init  | Bounds       | DICOM | OSIPI |
# +--------+------------+-----------------------------------------------+-----------------+-------+--------------+-------+-------+
# | FA     | deg        | flip angle                                    | Sequence        | 15    | (0, 180)     |       |       |
# | Nph    |            | number of acquired phase lines in k-space     | Sequence        | 128   | (0, 1000)    |       |       |
# | Nz     |            | number of slices in a multi-slice acquisition | Sequence        | 64    | (0, 1000)    |       |       |
# | PA     | deg        | preparation Pulse Flip Angle                  | Sequence        | 90    | (0, 180)     |       |       |
# | SA     | deg        | saturation Slab Flip Angle                    | Sequence        | 0     | (0, 180)     |       |       |
# | TA     | sec        | acquisition time                              | Sequence        | 2.0   | (0, 30)      |       |       |
# | TD     | sec        | prepulse delay                                | Sequence        | 0.05  | (0, 1)       |       |       |
# | TE     | sec        | echo time                                     | Sequence        | 0.001 | (0, 10)      |       |       |
# | TE2    | sec        | second echo time in a multi-echo sequence     | Sequence        | 0.005 | (0, 1)       |       |       |
# | TP     | sec        | preparation delay                             | Sequence        | 0.05  | (0, 1)       |       |       |
# | TR     | sec        | repetition time                               | Sequence        | 0.005 | (0, 1)       |       |       |
# | iz     |            | slice number in a multi-slice acquisition     | Sequence        | 0     | (0, 1000)    |       |       |
# | tacq   | sec        | acquisition duration                          | Sequence        | 240   | (0, 10000.0) |       |       |
# | tstart | sec        | start of the acquisition                      | Sequence        | 0     | (0, 10000.0) |       |       |
# +--------+------------+-----------------------------------------------+-----------------+-------+--------------+-------+-------+
# | B1corr |            | B1-correction factor                          | Electromagnetic | 1     | (0, 5)       |       |       |
# | Mzi    | A/cm       | longitudinal inlet magnetization              | Electromagnetic | 1     | (0, 5)       |       |       |
# | R1     | Hz         | tissue R1                                     | Electromagnetic | 0.65  | (0, 5)       |       |       |
# | R1i    | Hz         | inlet R1                                      | Electromagnetic | 0.65  | (0, 5)       |       |       |
# | me     | A cm2/mL   | equilibrium magnetization                     | Electromagnetic | 1     | (0, 5)       |       |       |
# | tMi    | sec        | inlet magnetization time points               | Electromagnetic | 0.0   |              |       |       |
# | tR     | sec        | relaxation rate time points                   | Electromagnetic | 0.0   |              |       |       |
# +--------+------------+-----------------------------------------------+-----------------+-------+--------------+-------+-------+
# | Fwi    | mL/sec/cm3 | inflow in all water compartments              | Physiological   | 0.02  | (0, 1)       |       |       |
# | Kw     | mL/sec/cm3 | water exchange matrix                         | Physiological   | 0     | (0, 1)       |       |       |
# | TF     | sec        | inflow time                                   | Physiological   | 0.5   | (0, 10)      |       |       |
# | inlets |            | water inlet compartments                      | Physiological   | (0,)  |              |       |       |
# | vw     | mL/cm3     | water volume fraction                         | Physiological   | 1     | (0, 1)       |       |       |
# +------------------------------------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------+
# |                                      MzPrep - all outputs (n = 2)                                     |
# +-----+------+----------------------------------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name                                   | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+----------------------------------------+-----------------+------+--------+-------+-------+
# | Mz  | A/cm | longitudinal magnetization             | Electromagnetic | 1    | (0, 5) |       |       |
# | tMz | sec  | longitudinal magnetization time points | Electromagnetic | 0.0  |        |       |       |
# +-------------------------------------------------------------------------------------------------------+

class MzPrep(Module): 
    configs = {
        'sequence': get_sequence('name'),
        'tof_corr': {False, True},
        'inflow': {'none', 'pool', 'inlet'}, 
    }
    defaults = {
        'sequence': '3D-SPGR-SS',
        'tof_corr': False,
        'inflow': 'none',
    }
    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap: dict=None, omap: dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        if self.config['tof_corr'] and self.config['sequence'] != '3D-SPGR-SS':
            raise InvalidConfig(f"Time-of-flight correction is only available for sequence 3D-SPGR-SS. You are running sequence {self.config['sequence']}. Either choose tof_corr=False or sequence='3D-SPGR-SS'.")
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
            o['tMz'] = tstart + functions_sequences.acquisition_times(sequence, p, p['tacq'])
            ntM = len(o['tMz'])
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
        if self.config['inflow'] == 'none':

            j, tj = None, None

        elif self.config['inflow'] == 'inlet':

            tj = p['tMi']
            ni = len(p['inlets'])

            # Format Fi
            Fwi = np.array(p['Fwi'])
            if Fwi.size != ni:
                raise ValueError(f"Fwi must have the length {ni}")
            Fwi = Fwi.reshape(ni) 

            for i in range(ni):
                if i==0:
                    j = np.zeros((nc, ) + p['Mzi'].shape[1:])
                inlet = p['inlets'][i] 

                # NOTE: extra dim because not in center (yet)
                j[inlet, :, :] = Fwi[i] * p['Mzi'][i, :, :]  # (mL/min/cm3) * (magn/mL) = magn/min/cm3
   
        elif self.config['inflow'] == 'pool':

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
                tj, Mzi = Mz_wrapper_k_all(sequence, mz_prep_inflow, tR, R1i[i], p, v=1, Kw=0, tstart=tstart, t_end=t_end)
                if i==0:
                    j = np.zeros((nc, ) + tj.shape)
                inlet = p['inlets'][i] 

                # NOTE: extra dim because center=False
                j[inlet, :, :] = Fwi[i] * Mzi[0, :, :]  # (mL/min/cm3) * (magn/mL) = magn/min/cm3

        # Delegate computation to helper functions
        mz_prep_sequence = get_sequence('mz_prep_tissue', sequence)
        o['tMz'], o['Mz'] = Mz_wrapper_k0(sequence, mz_prep_sequence, tR, R1, p, v, Kw, tj, j, tstart=tstart, t_end=t_end)

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
        if self.config['inflow'] == 'inlet':
            inputs |= {'Fwi', 'tMi', 'Mzi', 'inlets'}
        elif self.config['inflow'] == 'pool':
            inputs |= {'Fwi', 'R1i', 'inlets'}
        return inputs
    
    def outputs(self):
        return {'tMz', 'Mz'} # (compartments, times)

    def dummy_data(self, nc=2):
        p = self.init_data()
        ntR = 5
        tR = np.arange(ntR)
        R1 = np.ones((nc, ntR))
        tacq = ntR - 1
        tstart = p['tstart'] 
        t_end = tstart + tacq
        sequence = self.config['sequence']
        mz_prep_inflow = get_sequence('mz_prep_inflow', sequence)
        tMi, Mzi = Mz_wrapper_k_all(sequence, mz_prep_inflow, tR, R1[0], p, 
                            v=1, Kw=0, tstart=tstart, t_end=t_end)
        p |= {
            'tacq': ntR-1,
            'tR': tR,
            'R1': R1,
            'vw': np.ones(nc) / nc, 
            'Kw': np.ones((nc, nc)),
            'R1i': np.ones((nc, ntR)),
            'tMi': tMi,
            'Mzi': np.stack(nc * [Mzi], axis=0),
            'Fwi': np.ones(nc),
            'inlets': np.arange(nc),
        }
        return p


# +--------------------------------------------------------------------------------------------------+
# |                                 MxyReadMz - all configs (n = 1)                                  |
# +----------+--------------------------------------------------------------------------+------------+
# | Key      | Values                                                                   | Default    |
# +----------+--------------------------------------------------------------------------+------------+
# | sequence | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,        | 3D-SPGR-SS |
# |          | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR,   |            |
# |          | 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR,     |            |
# |          | 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS               |            |
# +--------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------------------------------+
# |                                                MxyReadMz - all inputs (n = 11)                                                |
# +--------+------+---------------------------------------------------------+-----------------+-------+-----------+-------+-------+
# | Key    | Unit | Name                                                    | Group           | Init  | Bounds    | DICOM | OSIPI |
# +--------+------+---------------------------------------------------------+-----------------+-------+-----------+-------+-------+
# | FA     | deg  | flip angle                                              | Sequence        | 15    | (0, 180)  |       |       |
# | Nk0    |      | number of acquired phase lines to the center of k-space | Sequence        | 64    | (0, 1000) |       |       |
# | TE     | sec  | echo time                                               | Sequence        | 0.001 | (0, 10)   |       |       |
# | TE1    | sec  | first echo time in a multi-echo sequence                | Sequence        | 0.001 | (0, 1)    |       |       |
# | TE2    | sec  | second echo time in a multi-echo sequence               | Sequence        | 0.005 | (0, 1)    |       |       |
# +--------+------+---------------------------------------------------------+-----------------+-------+-----------+-------+-------+
# | B1corr |      | B1-correction factor                                    | Electromagnetic | 1     | (0, 5)    |       |       |
# | Mz     | A/cm | longitudinal magnetization                              | Electromagnetic | 1     | (0, 5)    |       |       |
# | R2     | Hz   | tissue R2                                               | Electromagnetic | 2.0   | (0, 5)    |       |       |
# | R2s    | Hz   | tissue R2*                                              | Electromagnetic | 20    | (0, 5)    |       |       |
# | tMz    | sec  | longitudinal magnetization time points                  | Electromagnetic | 0.0   |           |       |       |
# | tR     | sec  | relaxation rate time points                             | Electromagnetic | 0.0   |           |       |       |
# +-------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------+
# |                             MxyReadMz - all outputs (n = 1)                             |
# +-----+------+--------------------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name                     | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+--------------------------+-----------------+------+--------+-------+-------+
# | Mxy | A/cm | transverse magnetization | Electromagnetic | 1    | (0, 5) |       |       |
# +-----------------------------------------------------------------------------------------+

class MxyReadMz(Module): 
    configs = {
        'sequence': get_sequence('name'),
    }
    defaults = {
        'sequence': '3D-SPGR-SS'
    }  
    _all_inputs = None
    _all_outputs = None

    def inputs(self):
        inputs = {'tR', 'tMz', 'Mz'} # shape (nc, n_times) 
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
            R2s = np.interp(p['tMz'], np.atleast_1d(p['tR']), np.atleast_1d(p['R2s']))
            
        if 'R2' in self._inputs:
            R2 = np.reshape(p['R2'], (nc, -1))  
            R2 = _interpolate_2d(p['tMz'], p['tR'], R2)

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
            'tMz': np.ones(ntM), 
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


# +--------------------------------------------------------------------------------------------------+
# |                               Magnetization - all configs (n = 3)                                |
# +----------+--------------------------------------------------------------------------+------------+
# | Key      | Values                                                                   | Default    |
# +----------+--------------------------------------------------------------------------+------------+
# | sequence | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,        | 3D-SPGR-SS |
# |          | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR,   |            |
# |          | 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR,     |            |
# |          | 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS               |            |
# | tof_corr | False, True                                                              | False      |
# | inflow   | inlet, none, pool                                                        | none       |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------+
# |                                                  Magnetization - all inputs (n = 30)                                                   |
# +--------+------------+---------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | Key    | Unit       | Name                                                    | Group           | Init  | Bounds       | DICOM | OSIPI |
# +--------+------------+---------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | FA     | deg        | flip angle                                              | Sequence        | 15    | (0, 180)     |       |       |
# | Nk0    |            | number of acquired phase lines to the center of k-space | Sequence        | 64    | (0, 1000)    |       |       |
# | Nph    |            | number of acquired phase lines in k-space               | Sequence        | 128   | (0, 1000)    |       |       |
# | Nz     |            | number of slices in a multi-slice acquisition           | Sequence        | 64    | (0, 1000)    |       |       |
# | PA     | deg        | preparation Pulse Flip Angle                            | Sequence        | 90    | (0, 180)     |       |       |
# | SA     | deg        | saturation Slab Flip Angle                              | Sequence        | 0     | (0, 180)     |       |       |
# | TA     | sec        | acquisition time                                        | Sequence        | 2.0   | (0, 30)      |       |       |
# | TD     | sec        | prepulse delay                                          | Sequence        | 0.05  | (0, 1)       |       |       |
# | TE     | sec        | echo time                                               | Sequence        | 0.001 | (0, 10)      |       |       |
# | TE1    | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001 | (0, 1)       |       |       |
# | TE2    | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005 | (0, 1)       |       |       |
# | TP     | sec        | preparation delay                                       | Sequence        | 0.05  | (0, 1)       |       |       |
# | TR     | sec        | repetition time                                         | Sequence        | 0.005 | (0, 1)       |       |       |
# | iz     |            | slice number in a multi-slice acquisition               | Sequence        | 0     | (0, 1000)    |       |       |
# | tacq   | sec        | acquisition duration                                    | Sequence        | 240   | (0, 10000.0) |       |       |
# | tstart | sec        | start of the acquisition                                | Sequence        | 0     | (0, 10000.0) |       |       |
# +--------+------------+---------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | B1corr |            | B1-correction factor                                    | Electromagnetic | 1     | (0, 5)       |       |       |
# | Mzi    | A/cm       | longitudinal inlet magnetization                        | Electromagnetic | 1     | (0, 5)       |       |       |
# | R1     | Hz         | tissue R1                                               | Electromagnetic | 0.65  | (0, 5)       |       |       |
# | R1i    | Hz         | inlet R1                                                | Electromagnetic | 0.65  | (0, 5)       |       |       |
# | R2     | Hz         | tissue R2                                               | Electromagnetic | 2.0   | (0, 5)       |       |       |
# | R2s    | Hz         | tissue R2*                                              | Electromagnetic | 20    | (0, 5)       |       |       |
# | me     | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1     | (0, 5)       |       |       |
# | tMi    | sec        | inlet magnetization time points                         | Electromagnetic | 0.0   |              |       |       |
# | tR     | sec        | relaxation rate time points                             | Electromagnetic | 0.0   |              |       |       |
# +--------+------------+---------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | Fwi    | mL/sec/cm3 | inflow in all water compartments                        | Physiological   | 0.02  | (0, 1)       |       |       |
# | Kw     | mL/sec/cm3 | water exchange matrix                                   | Physiological   | 0     | (0, 1)       |       |       |
# | TF     | sec        | inflow time                                             | Physiological   | 0.5   | (0, 10)      |       |       |
# | inlets |            | water inlet compartments                                | Physiological   | (0,)  |              |       |       |
# | vw     | mL/cm3     | water volume fraction                                   | Physiological   | 1     | (0, 1)       |       |       |
# +----------------------------------------------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------+
# |                                  Magnetization - all outputs (n = 4)                                  |
# +-----+------+----------------------------------------+-----------------+------+--------+-------+-------+
# | Key | Unit | Name                                   | Group           | Init | Bounds | DICOM | OSIPI |
# +-----+------+----------------------------------------+-----------------+------+--------+-------+-------+
# | M   | A/cm | magnetization                          | Electromagnetic | 1    | (0, 5) |       |       |
# | Mz  | A/cm | longitudinal magnetization             | Electromagnetic | 1    | (0, 5) |       |       |
# | tM  | sec  | magnetization time points              | Electromagnetic | 0.0  |        |       |       |
# | tMz | sec  | longitudinal magnetization time points | Electromagnetic | 0.0  |        |       |       |
# +-------------------------------------------------------------------------------------------------------+

class Magnetization(Module): 
    configs = MzPrep.configs | MxyReadMz.configs
    defaults = MzPrep.defaults | MxyReadMz.defaults

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)
        self._mz_prep = MzPrep(**self.config)
        self._mxy_read = MxyReadMz(**self.config)
        self.map_io(imap, omap, iomap)
        
    def inputs(self):
        inputs = self._mz_prep.mapped_inputs()
        inputs |= self._mxy_read.mapped_inputs() 
        inputs -= self._mz_prep.new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        return {'tM', 'M'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # All current sequences use an Mz prep followed by a readout
        # This could be generalized in the future to sequences that have mixed T1/T2 prep
        p |= self._mz_prep(p)
        p |= self._mxy_read(p) # (channels, components, compartments, times) 

        shape = p['Mxy'].shape
        p['M'] = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=p['Mxy'].dtype)

        p['M'][:, :2, :, :] = p['Mxy']
        for c in range(shape[0]):
            p['M'][c, 2, :, :] = p['Mz']

        return self.map_results(p)

    def dummy_data(self, nc=2):
        p = self.init_data()
        ntR = 5
        tR = np.arange(ntR)
        R1 = np.ones((nc, ntR))
        tacq = ntR - 1
        tstart = p['tstart'] 
        t_end = tstart + tacq
        sequence = self.config['sequence']
        mz_prep_inflow = get_sequence('mz_prep_inflow', sequence)
        tMi, Mzi = Mz_wrapper_k_all(
            sequence, mz_prep_inflow, tR, R1[0], p, 
            v=1, Kw=0, tstart=tstart, t_end=t_end
        )
        # tMi, Mzi = tMi[:, 0], Mzi[:, :, 0]
        p |= {
            'tacq': ntR-1,
            'tR': tR,
            'R1': R1,
            'R2': np.ones((nc, ntR)), 
            'R2s':np.ones(ntR),
            'vw': np.ones(nc) / nc, 
            'Kw': np.ones((nc, nc)),
            'R1i': np.ones((nc, ntR)),
            'tMi': tMi,
            'Mzi': np.stack(nc * [Mzi, Mzi], axis=0),
            'Fwi': np.ones(nc),
            'inlets': np.arange(nc),
        }
        return p