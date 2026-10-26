# +--------------------------------------------------------------------------------------------------+
# |                               InverseKidney - all configs (n = 11)                               |
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
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | water_exchange | F, N, R                                                            | F          |
# | baseline       | literature, measured                                               | literature |
# | kinetics       | 2CF, 2CFU, 2PF, 2PFU, CPF, FN, HF, HFU                             | 2CF        |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            InverseKidney - all inputs (n = 43)                                                            |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                             | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR_ki         |            | noise-to-signal ratio in the kidney                     | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0_ki          | a.u.       | signal scaling factor in the kidney                     | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | S_ki           | a.u.       | signal in the kidney                                    | Signal          | 1.0        | (0, 5)         |       |           |
# | iStrig_ki      |            | indices of the signal trigger in the kidney             | Signal          | None       |                |       |           |
# | nb             | a.u.       | number of baseline time points                          | Signal          | 1          |                |       |           |
# | pfree          | a.u.       | set of free parameters                                  | Signal          | 1          |                |       |           |
# | tS_ki          | sec        | signal time points in the kidney                        | Signal          | 0.0        |                |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                              | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)      |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr_ki      |            | B1-correction factor in the kidney                      | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_u           | Hz         | tissue R1 in tubuli                                     | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FF             |            | filtration fraction                                     | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | F_b_ki         | mL/sec/cm3 | flow per unit tissue in blood of the kidney             | Physiological   | 0.02       | (0, 1)         |       |           |
# | F_p_ki         | mL/sec/cm3 | flow per unit tissue in plasma of the kidney            | Physiological   | 0.02       | (0, 1)         |       |           |
# | F_u            | mL/sec/cm3 | flow per unit tissue in tubuli                          | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                 | Physiological   | 0.03       | (0, 100)       |       |           |
# | T_ar           | sec        | mean transit time in the artery                         | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_u            | sec        | mean transit time in tubuli                             | Physiological   | 120        | (0, 600)       |       |           |
# | h_u            | Hz         | transit time distribution in tubuli                     | Physiological   | 1.0        | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                            | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_ki           | mL/cm3     | volume fraction in the kidney                           | Physiological   | 1          | (0, 1)         |       |           |
# | v_p_ki         | mL/cm3     | volume fraction in plasma of the kidney                 | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_u            | mL/cm3     | volume fraction in tubuli                               | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |                |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                       InverseKidney - all outputs (n = 4)                                       |
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
from dcmri.utils.fit import train_brute
from dcmri.forward.kidney import Kidney as ForwardModel
from dcmri.kinetics.functions_kidney import dpars_kidney
from dcmri.inverse.modules import SignalToConc
from dcmri.bloch.functions_sequences import repetition_time

configs = deepcopy(ForwardModel.configs) | {'aif': {'fixed', 'measured'}}
defaults = deepcopy(ForwardModel.defaults) | {'aif': 'fixed', 'calibrate': True}

class InverseKidney(Module):

    configs = configs
    defaults = defaults
    quantities = {
        'T_ar': {'init': 0, 'bounds': (0.0, 10), 'name': 'arterial mean transit time', 'unit': 'sec', 'group': 'phys', 'dicom_key': None, 'osipi_key': None},
    }

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = ForwardModel(**self.config)
        if self.config['aif'] == 'measured':
            self._sig2conc = SignalToConc(**self.config)
        self.map_io(imap, omap)

    def _predict(self, p):
        pred = self.forward(p)
        return pred['S_ki'][:, :, :len(p['tS_ki'])].reshape(-1)

    def _preproc(self, p):
        # Reshape signal if needed
        if p['S_ki'].ndim == 1:
            p['S_ki'] = p['S_ki'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal_ki'] = p['S_ki'][..., :p['nb']]
            p['iScal_ki'] = np.arange(p['nb'])

        if self.config['aif'] == 'measured':
            tr = repetition_time(self.config['sequence'], p)
            tmax = max(p['tS_ar']) + 2 * (tr + p['dt'])
            t = np.arange(0, tmax, p['dt'])
            c_ar = self._sig2conc(p | {'S': p['S_ar'], 'R1b': p['R1_b']})['C']
            p['c_ar'] = np.interp(t, p['tS_ar'], c_ar)            

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {'S0_ki'}
        pfree -= {'v_ki'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def _pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['phys', 'body']}
        p = dpars_kidney(p, kinetics=self.config['kinetics'])
        return p

    def _call(self, p: dict=None, **kwargs) -> dict:
        self._preproc(p)

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self.pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        signal = p['S_ki']
        output = train_brute(self._predict, signal, p, p['pfree'], **kwargs)
        output['pars'] = self._pder(p | output['popt'])

        return output

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'nb'}
            inputs -= {'Scal_ki', 'iScal_ki'}
        inputs |= {'tS_ki', 'S_ki', 'pfree'}
        if self.config['aif'] == 'measured':
            inputs |= {'tS_ar', 'S_ar', 'R1_b'}
            inputs -= {'c_ar'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'pars', 'loss'}
        if self.config['aif'] == 'measured':
            outputs |= {'c_ar'}
        return outputs
    
    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.test_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'FF': (0, 1), 'F_u': (0, 1)}),
            'tS_ki': pred['tS_ki'],
            'S_ki': pred['S_ki'],
            'tS_ar': pred['tS_ki'],
            'S_ar': np.ones(len(pred['tS_ki'])) ** 2,
        }
        return self.input_data(p, data)

    # def plot(self, data:dict, results:dict=None, xlim:list=None, fname:str=None, show=True):
    #     p = data if results is None else data | results['popt']
    #     p = self.forward.map_data(p) | self.map_data(p)
    #     self._preproc(p)
    #     pred = self.forward(p)

    def plot(self, data: dict, result:dict=None, xlim:list=None, fname:str=None, show=True):
        if result is None:
            result = self.__call__(data)
        p = self.map_data(data | result['popt'])

        self._preproc(p)
        pred = self.forward(p)

        if xlim is None:
            xlim = [pred['tR_ki'][0], pred['tR_ki'][-1]]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        for i in range(p['S_ki'].shape[0]):
            for j in range(p['S_ki'].shape[1]):
                ax0.plot((p['tS_ki'] + pred['tS_ki'][0])/ 60, p['S_ki'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax0.plot(pred['tS_ki'] / 60, pred['S_ki'][i, j, :], marker='o', linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(pred['tC_ki'] / 60, 1000 * p['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(pred['tC_ki'] / 60, 1000 * pred['C_ki'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
        ax1.plot(pred['tC_ki'] / 60, 1000 * pred['C_ki'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Tubuli')
           
        ax1.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()