import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcmri import KidneyModel
from dcmri.core.exceptions import InvalidConfiguration


# +--------------------------------------------------------------------------------------------------+
# |                                KidneyModel - all configs (n = 11)                                |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | inflow         | False, True                                                        | False      |
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
# |                                                             KidneyModel - all inputs (n = 42)                                                             |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                             | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                   | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0             | a.u.       | signal scaling factor                                   | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | Scal           | a.u.       | calibration signal                                      | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | iScal          |            | indices of calibration signal                           | Signal          | 0          |                |       |           |
# | iStrig         |            | indices of the signal trigger                           | Signal          | None       |                |       |           |
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
# | B1corr         |            | B1-correction factor                                    | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ki          | Hz         | tissue R1 in the kidney                                 | Electromagnetic | 0.65       | (0, 5)         |       |           |
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
# | h_u            | Hz         | transit time distribution in tubuli                     | Physiological   | 0          | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                            | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_ki           | mL/cm3     | volume fraction in the kidney                           | Physiological   | 1          | (0, 1)         |       |           |
# | v_p_ki         | mL/cm3     | volume fraction in plasma of the kidney                 | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_u            | mL/cm3     | volume fraction in tubuli                               | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |                |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------------+
# |                                      KidneyModel - all outputs (n = 14)                                     |
# +-----+------------+--------------------------------+-----------------+-------+-----------+-------+-----------+
# | Key | Unit       | Name                           | Group           | Init  | Bounds    | DICOM | OSIPI     |
# +-----+------------+--------------------------------+-----------------+-------+-----------+-------+-----------+
# | C   | mmol/cm3   | tissue concentration           | Indicator       | 0.005 | (0, 1)    |       |           |
# | ci  | mmol/mL    | inlet concentration            | Indicator       | 0.005 |           |       |           |
# | tC  | sec        | concentration time points      | Indicator       | 0.0   |           |       |           |
# +-----+------------+--------------------------------+-----------------+-------+-----------+-------+-----------+
# | S   | a.u.       | signal                         | Signal          | 1.0   | (0, 5)    |       |           |
# | S0  | a.u.       | signal scaling factor          | Signal          | 1.0   | (0, 5)    |       | Q.MS1.010 |
# +-----+------------+--------------------------------+-----------------+-------+-----------+-------+-----------+
# | M   | A/cm       | magnetization                  | Electromagnetic | 1     | (0, 5)    |       |           |
# | R1  | Hz         | tissue R1                      | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1i | Hz         | inlet R1                       | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R2  | Hz         | tissue R2                      | Electromagnetic | 2.0   | (0, 5)    |       |           |
# | R2s | Hz         | tissue R2*                     | Electromagnetic | 20    | (0, 5)    |       |           |
# | tM  | sec        | magnetization time points      | Electromagnetic | 0.0   |           |       |           |
# | tR  | sec        | relaxation rate time points    | Electromagnetic | 0.0   |           |       |           |
# | tS  | sec        | signal time points             | Electromagnetic | 0.0   |           |       |           |
# +-----+------------+--------------------------------+-----------------+-------+-----------+-------+-----------+
# | F_u | mL/sec/cm3 | flow per unit tissue in tubuli | Physiological   | 0.005 | (0, 0.05) |       |           |
# +-------------------------------------------------------------------------------------------------------------+

def test_kidney(cls=KidneyModel):
    def _test_config(cnfg):
        # if cnfg['sequence'] != '3D-SPGR-SS':
        #     return
        try:
            instance = cls(**cnfg)
        except InvalidConfiguration:
            return

        # print(cnfg)
        data = instance.dummy_data()

        # --- DIAGNOSTIC TIMING ---
        t0 = time.perf_counter()
        
        instance(data)

        elapsed = time.perf_counter() - t0
        
        # print(f"  [Total model execution time: {elapsed:.4f}s]")

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=False, sample=None, seed=51)

    configs = cls.all_configs(sample=1000, seed=51)
    for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}'):
        _test_config(cnfg)

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_kidney_instance():
    # model = KidneyModel()
    # print(model.config)
    # return
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': None, 'inflow': False, 'sequence': 'ZTE-3D-IR-SPGR-SS', 'magnitude': False, 'trigger': False, 'calibrate': False, 'compartments': ('bc', 'u'), 'baseline': 'literature', 'kinetics': '2PF'}
    try:
        model = KidneyModel(**cnfg)
    except InvalidConfiguration as e:
        print(e)
        return
    model.print_inputs()
    model.print_outputs()

    data = model.dummy_data()
    results = model(data)

    plt.plot(results['tS'], results['S'][0, 0, :], 'ro')
    # plt.plot(results['tC'], results['C'][0], 'ro')

    plt.show()

if __name__ == '__main__':
    test_kidney()
    # test_kidney_instance()

    print('All KidneyModel coverage tests passed!!')