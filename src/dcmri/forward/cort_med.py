# +--------------------------------------------------------------------------------------------------+
# |                               ForwardCortMed - all configs (n = 10)                                |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | water_exchange | F, N, R                                                            | F          |
# | baseline       | literature, measured                                               | literature |
# | kinetics       | 7C                                                                 | 7C         |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            ForwardCortMed - all inputs (n = 55)                                                            |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |               |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                             | Indicator       | 0.005      | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_kc         |            | noise-to-signal ratio in the kidney cortex              | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_km         |            | noise-to-signal ratio in the kidney medulla             | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_kc          | a.u.       | signal scaling factor in the kidney cortex              | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_km          | a.u.       | signal scaling factor in the kidney medulla             | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal_kc        | a.u.       | calibration signal in the kidney cortex                 | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_km        | a.u.       | calibration signal in the kidney medulla                | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal_kc       |            | indices of calibration signal in the kidney cortex      | Signal          | 0          |               |       |           |
# | iScal_km       |            | indices of calibration signal in the kidney medulla     | Signal          | 0          |               |       |           |
# | iStrig_kc      |            | indices of the signal trigger in the kidney cortex      | Signal          | None       |               |       |           |
# | iStrig_km      |            | indices of the signal trigger in the kidney medulla     | Signal          | None       |               |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                              | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)     |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_kc      |            | B1-correction factor in the kidney cortex               | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_km      |            | B1-correction factor in the kidney medulla              | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_cd          | Hz         | tissue R1 in collecting ducts                           | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_dt          | Hz         | tissue R1 in distal tubuli                              | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_gc          | Hz         | tissue R1 in glomerular capillaries                     | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_lh          | Hz         | tissue R1 in lis-of-Henle                               | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_pt          | Hz         | tissue R1 in proximal tubuli                            | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_vb          | Hz         | tissue R1 in the venous blood                           | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | E_ki           |            | extraction fraction in the kidney                       | Physiological   | 0.15       | (0, 1)        |       |           |
# | F_p_ki         | mL/sec/cm3 | flow per unit tissue in plasma of the kidney            | Physiological   | 0.02       | (0, 1)        |       |           |
# | H              |            | hematocrit                                              | Physiological   | 0.45       | (0, 1)        |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                 | Physiological   | 0.03       | (0, 100)      |       |           |
# | T_ar           | sec        | mean transit time in the artery                         | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_cd           | sec        | mean transit time in collecting ducts                   | Physiological   | 30         | (0, 180)      |       |           |
# | T_dt           | sec        | mean transit time in distal tubuli                      | Physiological   | 30         | (0, 180)      |       |           |
# | T_gc           | sec        | mean transit time in glomerular capillaries             | Physiological   | 4          | (0, 30)       |       |           |
# | T_lh           | sec        | mean transit time in lis-of-Henle                       | Physiological   | 60         | (0, 180)      |       |           |
# | T_pcv          | sec        | mean transit time in peritubular capillaries and veins  | Physiological   | 10         | (0, 30)       |       |           |
# | T_pt           | sec        | mean transit time in proximal tubuli                    | Physiological   | 60         | (0, 180)      |       |           |
# | ffc            |            | cortical flow fraction                                  | Physiological   | 0.8        | (0, 1)        |       |           |
# | v_cd           | mL/cm3     | volume fraction in collecting ducts                     | Physiological   | 1          | (0, 1)        |       |           |
# | v_dt           | mL/cm3     | volume fraction in distal tubuli                        | Physiological   | 1          | (0, 1)        |       |           |
# | v_gc           | mL/cm3     | volume fraction in glomerular capillaries               | Physiological   | 1          | (0, 1)        |       |           |
# | v_lh           | mL/cm3     | volume fraction in lis-of-Henle                         | Physiological   | 1          | (0, 1)        |       |           |
# | v_pt           | mL/cm3     | volume fraction in proximal tubuli                      | Physiological   | 1          | (0, 1)        |       |           |
# | v_vb           | mL/cm3     | volume fraction in the venous blood                     | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |               |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------+
# |                                                   ForwardCortMed - all outputs (n = 26)                                                   |
# +--------+----------+--------------------------------------------------------------+-----------------+-------+--------+-------+-----------+
# | Key    | Unit     | Name                                                         | Group           | Init  | Bounds | DICOM | OSIPI     |
# +--------+----------+--------------------------------------------------------------+-----------------+-------+--------+-------+-----------+
# | C_kc   | mmol/cm3 | tissue concentration in the kidney cortex                    | Indicator       | 0.005 | (0, 1) |       |           |
# | C_km   | mmol/cm3 | tissue concentration in the kidney medulla                   | Indicator       | 0.005 | (0, 1) |       |           |
# | ci_ki  | mmol/mL  | inlet concentration in the kidney                            | Indicator       | 0.005 |        |       |           |
# | tC     | sec      | concentration time points                                    | Indicator       | 0.0   |        |       |           |
# +--------+----------+--------------------------------------------------------------+-----------------+-------+--------+-------+-----------+
# | S0_kc  | a.u.     | signal scaling factor in the kidney cortex                   | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# | S0_km  | a.u.     | signal scaling factor in the kidney medulla                  | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# | S_kc   | a.u.     | signal in the kidney cortex                                  | Signal          | 1.0   | (0, 5) |       |           |
# | S_km   | a.u.     | signal in the kidney medulla                                 | Signal          | 1.0   | (0, 5) |       |           |
# | tS_kc  | sec      | signal time points in the kidney cortex                      | Signal          | 0.0   |        |       |           |
# | tS_km  | sec      | signal time points in the kidney medulla                     | Signal          | 0.0   |        |       |           |
# +--------+----------+--------------------------------------------------------------+-----------------+-------+--------+-------+-----------+
# | M_kc   | A/cm     | magnetization in the kidney cortex                           | Electromagnetic | 1     | (0, 5) |       |           |
# | M_km   | A/cm     | magnetization in the kidney medulla                          | Electromagnetic | 1     | (0, 5) |       |           |
# | Mz_kc  | A/cm     | longitudinal magnetization in the kidney cortex              | Electromagnetic | 1     | (0, 5) |       |           |
# | Mz_km  | A/cm     | longitudinal magnetization in the kidney medulla             | Electromagnetic | 1     | (0, 5) |       |           |
# | R1_kc  | Hz       | tissue R1 in the kidney cortex                               | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1_km  | Hz       | tissue R1 in the kidney medulla                              | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1i_kc | Hz       | inlet R1 in the kidney cortex                                | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R2_kc  | Hz       | tissue R2 in the kidney cortex                               | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2_km  | Hz       | tissue R2 in the kidney medulla                              | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2s_kc | Hz       | tissue R2* in the kidney cortex                              | Electromagnetic | 20    | (0, 5) |       |           |
# | R2s_km | Hz       | tissue R2* in the kidney medulla                             | Electromagnetic | 20    | (0, 5) |       |           |
# | tM_kc  | sec      | magnetization time points in the kidney cortex               | Electromagnetic | 0.0   |        |       |           |
# | tM_km  | sec      | magnetization time points in the kidney medulla              | Electromagnetic | 0.0   |        |       |           |
# | tMz_kc | sec      | longitudinal magnetization time points in the kidney cortex  | Electromagnetic | 0.0   |        |       |           |
# | tMz_km | sec      | longitudinal magnetization time points in the kidney medulla | Electromagnetic | 0.0   |        |       |           |
# | tR     | sec      | relaxation rate time points                                  | Electromagnetic | 0.0   |        |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------+

import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcCortMed
from dcmri.relaxivity.modules_rois import RelaxivityGeneric
from dcmri.bloch.modules_rois import WaterExchangeGeneric
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

configs = ConcToSignal.configs | WaterExchangeGeneric.configs | RelaxivityGeneric.configs | ConcCortMed.configs
defaults = ConcToSignal.defaults | WaterExchangeGeneric.defaults | RelaxivityGeneric.defaults | ConcCortMed.defaults

for k in ('tof_corr', 'inflow'):
    configs.pop(k, None)
    defaults.pop(k, None)


class ForwardCortMed(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {'T_pt', 'R1_dt', 'v_pt', 'TE1', 'TE2', 'NSR_kc', 'field_strength', 'Nk0', 'TR', 'R1_gc', 'NSR_km', 'agent', 'c_ar', 'B1corr_kc', 'Nph', 'B1corr_km', 'T_ar', 'Scal_kc', 'iStrig_kc', 'Scal_km', 'Nz', 'R1_vb', 'R1_pt', 'v_cd', 'me', 'TE', 'T_gc', 'R1_lh', 'T_dt', 'v_dt', 'PA', 'S0_kc', 'tstart', 'T_lh', 'v_vb', 'T_cd', 'iStrig_km', 'T_pcv', 'R1_cd', 'TA', 'F_p_ki', 'H', 'dt', 'FA', 'v_lh', 'iScal_kc', 'iz', 'PSw', 'S0_km', 'iScal_km', 'ffc', 'TD', 'v_gc', 'E_ki', 'TP'}
    _all_outputs = {'tS_kc', 'C_kc', 'S_km', 'R1_kc', 'R1_km', 'R2s_kc', 'tMz_kc', 'C_km', 'R2_kc', 'tR', 'R2_km', 'tMz_km', 'tC', 'M_kc', 'S0_km', 'M_km', 'R2s_km', 'S_kc', 'tS_km', 'R1i_kc', 'ci_ki', 'Mz_km', 'tM_km', 'tM_kc', 'Mz_kc', 'S0_kc'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)

        config = {
            'kc': self.config | {'inflow': 'pool'}, 
            'km': self.config | {'inflow': 'inlet'} # Medulla recieves inflow from the cortex outlet
        }
        self._conc = ConcCortMed(**self.config)
        self._tissue_rel = {}
        self._tissue_wex = {}
        self._conc_to_signal = {}

        for roi in ['kc', 'km']:
            self._tissue_rel[roi] = RelaxivityGeneric(**config[roi])
            self._tissue_wex[roi] = WaterExchangeGeneric(**config[roi])

            imap = {k: f'{k}_{roi}' for k in {'C', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr'}}
            imap |= {'ci':'ci_ki'}
            omap = {k: f"{k}_{roi}" for k in ConcToSignal.all_outputs() - {'tR'}}
            self._conc_to_signal[roi] = ConcToSignal(imap=imap, omap=omap, **config[roi])

        self.map_io(imap, omap, iomap)


    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Cortex and medulla concentration

        p |= self._conc(p)

        p['tacq'] = p['dt'] * (p['ci_ki'].size - 1)

        # Cortex signal

        roi = 'kc'

        p['v_t'] = [p[f'v_{c}'] for c in ['gc', 'vb', 'pt', 'dt']]
        p['R1_t'] = [p[f'R1_{c}'] for c in ['gc', 'vb', 'pt', 'dt']]

        p |= self._tissue_rel[roi](p) 

        p['F_b'] = p['F_p_ki'] / (1 - p['H'])

        p |= self._tissue_wex[roi](p) 
        p |= self._conc_to_signal[roi](p)

        # Medulla signal

        roi = 'km'

        p['v_t'] = [p[f'v_{c}'] for c in ['vb', 'lh', 'cd']]
        p['R1_t'] = [p[f'R1_{c}'] for c in ['vb', 'lh', 'cd']]

        p |= self._tissue_rel[roi](p)

        p['F_b'] = (p['F_p_ki'] / (1 - p['H'])) * (1 - p['ffc']) * (1 - p['E_ki']) 

        p |= self._tissue_wex[roi](p) 

        p['Mzi'] = p['Mz_kc'][0, :, :].reshape((1, ) + p['Mz_kc'].shape[1:]) # Cortex blood Mz is inlet for Medulla
        p['tMi'] = p['tMz_kc']

        p |= self._conc_to_signal[roi](p)

        return self.map_results(p)
    

    def inputs(self) -> set:
        comps = ['gc', 'vb', 'pt', 'dt', 'lh', 'cd']

        inputs = {f'v_{c}' for c in comps}
        inputs |= {f'R1_{c}' for c in comps}
        inputs |= {'F_p_ki', 'H', 'ffc', 'E_ki'}
        inputs |= self._conc.mapped_inputs()
        for roi in ['kc', 'km']:
            inputs |= self._tissue_rel[roi].mapped_inputs() 
            inputs |= self._tissue_wex[roi].mapped_inputs() 
            inputs |= self._conc_to_signal[roi].mapped_inputs()

        inputs -= {'tacq', 'v_t', 'R1_t', 'F_b', 'Mzi', 'tMi'} # remove derived
        inputs -= self._conc.new_mapped_outputs()
        for roi in ['kc', 'km']:
            inputs -= self._tissue_rel[roi].new_mapped_outputs()
            inputs -= self._tissue_wex[roi].new_mapped_outputs()
        return inputs 
    
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in ['kc', 'km']:
            outputs |= self._conc_to_signal[roi].mapped_outputs() 
        return outputs
    
    
    def dummy_data(self): 
        n0, nt = 1, 180

        p = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        for roi in ['kc', 'km']:
            p |= {
                f'iScal_{roi}': np.arange(n0, dtype=int),
                f'Scal_{roi}': Scal, 
            }
        p['c_ar'] = np.ones(nt)
        return p