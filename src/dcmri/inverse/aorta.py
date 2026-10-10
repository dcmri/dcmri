# +--------------------------------------------------------------------------------------------------+
# |                               InverseAorta - all configs (n = 16)                                |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | inflow         | none, pool                                                         | none       |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | tof_corr       | False, True                                                        | False      |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | baseline       | literature, measured                                               | literature |
# | heartlung      | chain, comp, pfcomp                                                | pfcomp     |
# | organs         | 2cxm, comp                                                         | comp       |
# | kidneys        | None, comp, pass, plug                                             | None       |
# | liver          | None, comp, pass, plug                                             | None       |
# | lagut          | None, comp, pass, plucom                                           | None       |
# | bolus          | double, single                                                     | single     |
# +--------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                           InverseAorta - all inputs (n = 56)                                                           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit     | Name                                                    | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | agent          |          | contrast agent generic name                             | Indicator       | gadoterate |               |       |           |
# | bdel           | sec      | delay in a double injection                             | Indicator       | 30         | (-30, 30)     |       |           |
# | dose           | mL/kg    | contrast agent dose                                     | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg    | 1st contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg    | 2nd contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s     | injection rate                                          | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s     | 1st injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s     | 2nd injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR            |          | noise-to-signal ratio                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S              | a.u.     | signal                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S0             | a.u.     | signal scaling factor                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | iStrig         |          | indices of the signal trigger                           | Signal          | None       |               |       |           |
# | nb             | a.u.     | number of baseline time points                          | Signal          | 1          |               |       |           |
# | pfree          | a.u.     | set of free parameters                                  | Signal          | 1          |               |       |           |
# | tS             | sec      | signal time points                                      | Signal          | 0.0        |               |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg      | flip angle                                              | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |          | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |          | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |          | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg      | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg      | saturation Slab Flip Angle                              | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec      | acquisition time                                        | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec      | prepulse delay                                          | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec      | echo time                                               | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec      | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec      | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec      | preparation delay                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec      | repetition time                                         | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T        | magnetic field strength                                 | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |          | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq           | sec      | acquisition duration                                    | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart         | sec      | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr         |          | B1-correction factor                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz       | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec   | cardiac output                                          | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |          | transit time dispersion in the heart and Lungs          | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_or           |          | extraction fraction in the organs                       | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | TF             | sec      | inflow time                                             | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_b_or         | sec      | mean transit time in blood of the organs                | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_li         | sec      | mean transit time in extracellular space of the liver   | Physiological   | 30.0       | (0.1, 60)     |       |           |
# | T_e_or         | sec      | mean transit time in extracellular space of the organs  | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec      | mean transit time in the gut                            | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_hl           | sec      | mean transit time in the heart and Lungs                | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec      | mean transit time in the liver artery                   | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_p_lk         | sec      | mean transit time in plasma of the left kidney          | Physiological   | 5          | (0, 30)       |       |           |
# | T_p_rk         | sec      | mean transit time in plasma of the right kidney         | Physiological   | 5          | (0, 30)       |       |           |
# | ffa            |          | arterial flow fraction                                  | Physiological   | 0.2        | (0, 1)        |       |           |
# | vr_li          |          | Venous return in the liver                              | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_lk          |          | Venous return in the left kidney                        | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_or          |          | Venous return in the organs                             | Physiological   | 0.9        | (0, 1)        |       |           |
# | vr_rk          |          | Venous return in the right kidney                       | Physiological   | 0.15       | (0, 1)        |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |          | dose tolerance                                          | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec      | pseudo-continuous time step                             | Hyperparameters | 0.5        |               |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_ao         | cm3      | ROI volume in the aorta                                 | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | weight         | kg       | body weight                                             | Whole-body      | 70         | (0, 300)      |       |           |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                        InverseAorta - all outputs (n = 4)                                       |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | Key   | Unit | Name                                           | Group           | Init | Bounds | DICOM | OSIPI |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | loss  | a.u. | loss value of optimized model                  | Signal          | 1    |        |       |       |
# | pcov  | a.u. | dictionary with covariances of free parameters | Signal          | 1    |        |       |       |
# | popt  | a.u. | dictionary of optimized free parameter values  | Signal          | 1    |        |       |       |
# | psdev | a.u. | dictionary with parameter standard deviations  | Signal          | 1    |        |       |       |
# +-----------------------------------------------------------------------------------------------------------------+
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train_bat, train
from dcmri.forward.aorta import ForwardAorta as Forward

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

configs['bolus'].discard('dual') # Only meaningful for split protocols


class InverseAorta(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'B1corr', 'dt', 'tS', 'dose_tolerance', 'TE2', 'TF', 'NSR', 'S0', 'T_e_li', 'E_or', 'T_b_or', 'T_p_lk', 'rate_1', 'D_hl', 'T_e_or', 'R1_b', 'vr_rk', 'S', 'field_strength', 'T_p_rk', 'Nph', 'tstart', 'vol_ao', 'T_la', 'TP', 'dose_1', 'CO', 'Nk0', 'tacq', 'dose_2', 'TD', 'weight', 'nb', 'rate', 'TE1', 'PA', 'TE', 'bdel', 'iz', 'iStrig', 'vr_li', 'TA', 'T_hl', 'dose', 'agent', 'pfree', 'rate_2', 'Nz', 'FA', 'vr_or', 'ffa', 'vr_lk', 'TR', 'SA', 'T_gu', 'me'}
    _all_outputs = {'psdev', 'popt', 'loss', 'pcov'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        nt = np.size(time)
        pred = self.forward(self._pars)
        return pred['S'][:, :, :nt].reshape(-1)

    def _preproc(self, p):
        # Reshape signal if needed
        if p['S'].ndim == 1:
            p['S'] = p['S'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal'] = p['S'][..., :p['nb']]
            p['iScal'] = np.arange(p['nb'])

    def _pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        pfree |= {'BAT'}
        if not self.config['calibrate']:
            pfree |= {'S0'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data) 
        self._preproc(p)

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self.pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        #output = train_bat(self._predict, p['tS'], p['S'], p, p['pfree'], bats=['BAT'], **kwargs)
        output = train(self._predict, p['tS'], p['S'], p, p['pfree'], **kwargs)

        return self.map_results(output)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'tS', 'S', 'pfree'}
        if self.config['calibrate']:
            inputs |= {'nb'}
            inputs -= {'Scal', 'iScal'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'loss'}
    
    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.test_data()

        pred = self.forward(p)
        pfree = {'CO': (10, 300), 'BAT': (-60, 60)}
        
        p |= {
            'tS': pred['tS'],
            'S': pred['S'], 
            'nb': 5,
            'pfree': self.forward.filter_data(pfree),
        }
        return self.input_data(p, data)


    def plot(self, data: dict, xlim=None, fname:str=None, show=True):
        p = self.map_data(data)
        self._preproc(p)
        pred = self.forward(p)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(pred['tS'], pred['S'],p['tS'], p['S'], ax0, ['lightcoral', 'darkred'])

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        if 'C' in p:
            t, c = p['tC'], p['C'].reshape(1, -1)
            ax1.plot(t / 60, 1000 * c[0], linestyle='-', color='lightcoral', linewidth=5, label='Reference')
        ax1.plot(pred['tC'] / 60, 1000 * pred['C'][0], linestyle='-', color='darkred', linewidth=3, label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: 
            plt.savefig(fname)
        if show: 
            plt.show()
        else: 
            plt.close()   