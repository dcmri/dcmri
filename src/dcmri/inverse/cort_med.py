# +--------------------------------------------------------------------------------------------------+
# |                              InverseCortMed - all configs (n = 10)                               |
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
# |                                                           InverseCortMed - all inputs (n = 61)                                                           |
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
# | S_kc           | a.u.       | signal in the kidney cortex                             | Signal          | 1.0        | (0, 5)        |       |           |
# | S_km           | a.u.       | signal in the kidney medulla                            | Signal          | 1.0        | (0, 5)        |       |           |
# | Scal_kc        | a.u.       | calibration signal in the kidney cortex                 | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_km        | a.u.       | calibration signal in the kidney medulla                | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal_kc       |            | indices of calibration signal in the kidney cortex      | Signal          | 0          |               |       |           |
# | iScal_km       |            | indices of calibration signal in the kidney medulla     | Signal          | 0          |               |       |           |
# | iStrig_kc      |            | indices of the signal trigger in the kidney cortex      | Signal          | None       |               |       |           |
# | iStrig_km      |            | indices of the signal trigger in the kidney medulla     | Signal          | None       |               |       |           |
# | nb             | a.u.       | number of baseline time points                          | Signal          | 1          |               |       |           |
# | pfree          | a.u.       | set of free parameters                                  | Signal          | 1          |               |       |           |
# | tS_kc          | sec        | signal time points in the kidney cortex                 | Signal          | 0.0        |               |       |           |
# | tS_km          | sec        | signal time points in the kidney medulla                | Signal          | 0.0        |               |       |           |
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

# +-----------------------------------------------------------------------------------------------------------------+
# |                                       InverseCortMed - all outputs (n = 4)                                      |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | Key   | Unit | Name                                           | Group           | Init | Bounds | DICOM | OSIPI |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | loss  | a.u. | loss value of optimized model                  | Signal          | 1    |        |       |       |
# | pcov  | a.u. | dictionary with covariances of free parameters | Signal          | 1    |        |       |       |
# | popt  | a.u. | dictionary of optimized free parameter values  | Signal          | 1    |        |       |       |
# | psdev | a.u. | dictionary with parameter standard deviations  | Signal          | 1    |        |       |       |
# +-----------------------------------------------------------------------------------------------------------------+

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train
from dcmri.forward.cort_med import ForwardCortMed as Forward

configs = deepcopy(Forward.configs) 
defaults = deepcopy(Forward.defaults)

ROIS = ['kc', 'km']

class InverseCortMed(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'TE2', 'TA', 'TE', 'tS_kc', 'TR', 'v_cd', 'T_gc', 'H', 'c_ar', 'R1_gc', 'v_lh', 'B1corr_kc', 'Scal_kc', 'PA', 'iStrig_kc', 'PSw', 'T_pt', 'R1_vb', 'R1_lh', 'v_vb', 'R1_pt', 'T_lh', 'T_pcv', 'T_dt', 'Nz', 'v_dt', 'NSR_km', 'S0_km', 'T_ar', 'iScal_kc', 'S0_kc', 'iStrig_km', 'Nk0', 'Nph', 'R1_dt', 'tstart', 'v_gc', 'T_cd', 'NSR_kc', 'v_pt', 'pfree', 'FA', 'nb', 'dt', 'field_strength', 'Scal_km', 'me', 'TD', 'S_kc', 'S_km', 'TE1', 'TP', 'iScal_km', 'ffc', 'R1_cd', 'B1corr_km', 'tS_km', 'iz', 'agent', 'F_p_ki', 'E_ki'}
    _all_outputs = {'loss', 'pcov', 'psdev', 'popt'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        return tuple([pred[f'S_{roi}'][:, :, :len(time[i])].reshape(-1) for i, roi in enumerate(ROIS)])

    def _preproc(self, p):
        # Reshape signal if needed
        for roi in ROIS:
            if p[f'S_{roi}'].ndim == 1:
                p[f'S_{roi}'] = p[f'S_{roi}'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            for roi in ROIS:
                p[f'Scal_{roi}'] = p[f'S_{roi}'][:, :, :p['nb']]
                p[f'iScal_{roi}'] = np.arange(p['nb'])

    def _pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {f'S0_{roi}' for roi in ROIS}
        return {p: get_quantity(p)['bounds'] for p in pfree}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  
        self._preproc(p) 

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = tuple([p[f'tS_{roi}'].reshape(-1) for roi in ROIS])
        signal = tuple([p[f'S_{roi}'].reshape(-1) for roi in ROIS])
        output = train(self._predict, time, signal, p, p['pfree'], **kwargs)

        return self.map_results(output)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'pfree'}
        for roi in ROIS:
            inputs |= {f'tS_{roi}', f'S_{roi}'}
        if self.config['calibrate']:
            inputs |= {'nb'}
            for roi in ROIS:
                inputs |= {f'Scal_{roi}', f'iScal_{roi}'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'loss'}
        return outputs
    
    def test_data(self, data: dict=None): 
        p = self.init_data()

        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'F_p_ki': (0, 1), 'E_ki': (0, 1)}),
        }
        p |= self.forward.test_data()
        pred = self.forward(p)
        for roi in ROIS:
            p |= {
                f'tS_{roi}': pred[f'tS_{roi}'],
                f'S_{roi}': pred[f'S_{roi}'],
            }
        return self.input_data(p, data)

    def plot(self, data: dict, xlim=None, fname=None, show=True):
        p = self.map_data(data)  
        self._preproc(p)
        prediction = self.forward(p)

        if xlim is None: 
            xlim = [prediction['tR'][0], prediction['tR'][-1]]
        xlim = np.array(xlim) / 60

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot signals
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(prediction['tS_kc'], prediction['S_kc'], p['tS_kc'], p['S_kc'], ax0, ['lightcoral', 'darkred'])
        plot_data(prediction['tS_km'], prediction['S_km'], p['tS_km'], p['S_km'], ax0, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax1.set_title('Reconstruction of concentrations.')
        ax1.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax1.plot(prediction['tC'] / 60, 1000 * p['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C_kc'].sum(axis=0), linestyle='-', linewidth=3.0, color='darkred', label='Cortex')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C_km'].sum(axis=0), linestyle='-', linewidth=3.0, color='darkcyan', label='Medulla')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()