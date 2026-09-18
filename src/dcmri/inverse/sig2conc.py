import numpy as np

from dcmri.core.tools import get_sequence
import dcmri.inverse.lib as solve
from dcmri.core.module import Module
from dcmri.signal.modules_tissue import RelaxToSignal
from dcmri.bloch.functions_sequences import channels

invertible_seqs = get_sequence('steady-state')
analytical_inversion = ['3D-SPGR-SS', 'ZTE-3D-SPGR-SS', 'lin', '2D-GE-EPI', '2D-SE-EPI', '2D-DE-EPI']


# +--------------------------------------------------------------------------------------------------+
# |                                SignalToConc - all configs (n = 2)                                |
# +-----------+-------------------------------------------------------------------------+------------+
# | Key       | Values                                                                  | Default    |
# +-----------+-------------------------------------------------------------------------+------------+
# | sequence  | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR-SS, 2D-SR-SPGR, 3D-DE-EPI,     | 3D-SPGR-SS |
# |           | 3D-GE-EPI, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, |            |
# |           | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS,     |            |
# |           | ZTE-3D-SPGR-SS, lin                                                     |            |
# | calibrate | False, True                                                             | True       |
# +--------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------------------------------------------------------------------+
# |                                                   SignalToConc - all inputs (n = 23)                                                  |
# +--------+------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Key    | Unit | Name                                                    | Group           | Init  | Bounds        | DICOM | OSIPI     |
# +--------+------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | NSR    |      | noise-to-signal ratio                                   | Signal          | 0.0   | (0, 100000.0) |       |           |
# | S      | a.u. | signal                                                  | Signal          | 1.0   | (0, 5)        |       |           |
# | S0     | a.u. | signal scaling factor                                   | Signal          | 1.0   | (0, 5)        |       | Q.MS1.010 |
# | nb     | a.u. | number of baseline time points                          | Signal          | 1     |               |       |           |
# +--------+------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | FA     | deg  | flip angle                                              | Sequence        | 15    | (0, 180)      |       |           |
# | Nk0    |      | number of acquired phase lines to the center of k-space | Sequence        | 64    | (0, 1000)     |       |           |
# | Nph    |      | number of acquired phase lines in k-space               | Sequence        | 128   | (0, 1000)     |       |           |
# | PA     | deg  | preparation Pulse Flip Angle                            | Sequence        | 90    | (0, 180)      |       |           |
# | TA     | sec  | acquisition time                                        | Sequence        | 2.0   | (0, 30)       |       |           |
# | TD     | sec  | prepulse delay                                          | Sequence        | 0.05  | (0, 1)        |       |           |
# | TE     | sec  | echo time                                               | Sequence        | 0.001 | (0, 10)       |       |           |
# | TE1    | sec  | first echo time in a multi-echo sequence                | Sequence        | 0.001 | (0, 1)        |       |           |
# | TE2    | sec  | second echo time in a multi-echo sequence               | Sequence        | 0.005 | (0, 1)        |       |           |
# | TP     | sec  | preparation delay                                       | Sequence        | 0.05  | (0, 1)        |       |           |
# | TR     | sec  | repetition time                                         | Sequence        | 0.005 | (0, 1)        |       |           |
# | iz     |      | slice number in a multi-slice acquisition               | Sequence        | 0     | (0, 1000)     |       |           |
# | tacq   | sec  | acquisition duration                                    | Sequence        | 240   | (0, 10000.0)  |       |           |
# | tstart | sec  | start of the acquisition                                | Sequence        | 0     | (0, 10000.0)  |       |           |
# +--------+------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | B1corr |      | B1-correction factor                                    | Electromagnetic | 1     | (0, 5)        |       |           |
# | R1b    | Hz   | precontrast tissue R1                                   | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | r1     | Hz/M | longitudinal contrast agent relaxivity                  | Electromagnetic | 3500  | (0, 10000.0)  |       |           |
# | r2     | Hz/M | transverse contrast agent relaxivity                    | Electromagnetic | 4000  | (0, 10000.0)  |       |           |
# | r2s    | Hz/M | transverse contrast agent relaxivity                    | Electromagnetic | 20000 | (0, 100000.0) |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------+
# |                            SignalToConc - all outputs (n = 1)                            |
# +-----+----------+----------------------+-----------------+-------+--------+-------+-------+
# | Key | Unit     | Name                 | Group           | Init  | Bounds | DICOM | OSIPI |
# +-----+----------+----------------------+-----------------+-------+--------+-------+-------+
# | C   | mmol/cm3 | tissue concentration | Indicator       | 0.005 | (0, 1) |       |       |
# +------------------------------------------------------------------------------------------+

class SignalToConc(Module):
    configs = {
        'sequence': invertible_seqs | {'lin'},
        'calibrate': {False, True},
    }
    defaults = {
        'sequence': '3D-SPGR-SS',
        'calibrate': True,
    }
    _all_inputs = {'r1', 'TE', 'TR', 'TP', 'r2s', 'S', 'Nph', 'nb', 'tstart', 'TE2', 'NSR', 'R1b', 'Nk0', 'FA', 'B1corr', 'PA', 'TD', 'iz', 'tacq', 'TE1', 'TA', 'S0', 'r2'}
    _all_outputs = {'C'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        if self.config['sequence'] not in analytical_inversion:
            # Calibration done during inversion so not in RelaxToSignal
            config = self.config | {'calibrate': False, 'inflow': 'none'}
            self._R1_to_S = RelaxToSignal(**config)
        self.map_io(imap, omap)  

    def inputs(self):
        sequence = self.config['sequence']
        inputs = {'S'}

        if sequence in ['3D-SPGR-SS']:
            inputs |= {'r1', 'FA', 'TR', 'B1corr', 'nb'}
            if self.config['calibrate']:
                inputs |= {'R1b'}  
            else:
                inputs |= {'S0'}  

        elif sequence in ['ZTE-3D-SPGR-SS']:
            inputs |= {'r1', 'FA', 'TR', 'B1corr', 'nb'}
            if self.config['calibrate']:
                inputs |= {'R1b'}  
            else:
                inputs |= {'S0'} 

        elif sequence == 'lin':
            inputs |= {'r1', 'nb'}
            if self.config['calibrate']:
                inputs |= {'R1b'}  
            else:
                inputs |= {'S0'} 

        elif sequence in ['2D-GE-EPI', '3D-GE-EPI']:
            inputs |= {'nb', 'r2s', 'TE'}

        elif sequence in ['2D-SE-EPI', '3D-SE-EPI']:
            inputs |= {'nb', 'r2', 'TE'}

        elif sequence in ['2D-DE-EPI', '3D-DE-EPI']:
            inputs |= {'nb', 'r2', 'r2s', 'TE1', 'TE2'}

        else:
            derived = {'R1', 'R2s', 'TE', 'vw', 'Kw', 'me'}
            inputs |= self._R1_to_S.inputs() - {'tR'}
            inputs |= {'nb', 'r1'}
            if self.config['calibrate']:
                inputs |= {'R1b'} 
                derived |= {'S0'}
            else:
                inputs |= {'S0'} 
            inputs -= derived

        return inputs
    
    def outputs(self):
        return {'C'}  # (n_samples, n_channels, n_times) or (n_channels, n_times) or (n_times)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Input shape is either (n_samples, n_channels, n_times) or (n_channels, n_times) or (n_times)
        # Output shapes are the same        
        # Check input
        S = np.array(p['S'])
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
        sequence = self.config['sequence']
        p = {k: v for k, v in p.items() if k != 'S'}

        if sequence in ['3D-SPGR-SS']:
            conc = solve.conc_ss(S[:,0,:], **p)
            conc = conc[:, None, :]

        elif sequence in ['ZTE-3D-SPGR-SS']:
            conc = solve.conc_ss(S[:,0,:], **p)  
            conc = conc[:, None, :]

        elif sequence == 'lin':
            conc = solve.conc_dce_lin(S[:,0,:], **p)
            conc = conc[:, None, :]

        elif sequence in ['2D-GE-EPI', '3D-GE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], nb=p['nb'], r2=p['r2s'], TE=p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['2D-SE-EPI', '3D-SE-EPI']:
            conc = solve.conc_dsc(S[:,0,:], nb=p['nb'], r2=p['r2'], TE=p['TE'])
            conc = conc[:, None, :]

        elif sequence in ['2D-DE-EPI', '3D-DE-EPI']:
            S_GE, S_SE = S[:1,0,:], S[-1:,0,:] # TODO: do we really need to keep the first dim?
            conc_ge = solve.conc_dsc(S_GE, nb=p['nb'], r2=p['r2s'], TE=p['TE1'])
            conc_se = solve.conc_dsc(S_SE, nb=p['nb'], r2=p['r2'], TE=p['TE2'])
            conc_ge = conc_ge[:, None, :]
            conc_se = conc_se[:, None, :]
            conc = np.concatenate((conc_ge, conc_se), axis=1)

        else:
            if self.config['calibrate']:
                conc = solve.conc_dce(self._R1_to_S, S[:,0,:], r1=p['r1'], nb=p['nb'], R1b=p['R1b'], defaults=p)
            else:
                conc = solve.conc_dce(self._R1_to_S, S[:,0,:], r1=p['r1'], nb=p['nb'], S0=p['S0'], defaults=p)
            conc = conc[:, None, :]

        if ndim==1:
            conc = conc[0,0,:]
        elif ndim==2:
            conc = conc[0,:,:]
        else:
            conc = conc

        results = {'C': conc}
        return self.map_results(results)

    def dummy_data(self, nt=5):
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1
        S = np.ones((n_channels, components, nt))
        data |= {
            'S': S,
        }
        return data