import time
from joblib import Parallel, delayed

from tqdm import tqdm

import dcmri as dc
from dcmri.core.module import InvalidConfig


def test_inverse_model(cls, simple=False, io_sample=1e5, cnfg_sample=1e4, seed=51):
    def _test_config(cnfg):
        try:
            model = cls(**cnfg)
        except InvalidConfig:
            return

        data = model.test_data()
        result = model(data)
        #result = model(data, n_bat=5)

        model.plot(data | result['popt'], show=False)
        print(result['loss'])

        assert result['loss'] < 1e-3, f"Loss for config {cnfg} is {result['loss']}"

    cls.print_configs()
    cls.print_all_io(verbose=1, simple=simple, sample=io_sample, seed=seed)

    configs = cls.all_configs(sample=cnfg_sample, seed=seed)
    # [
    #     _test_config(cnfg)
    #     for cnfg in tqdm(configs, desc=f'Testing {cls.__name__}')
    # ]

    Parallel(n_jobs=-1)(
        delayed(_test_config)(cnfg) for cnfg in configs
    )

    print(f'Successfully covered {len(configs)} {cls.__name__} configurations!')


def test_inverse_model_instance(cls, cnfg):
    # model = InverseModel()
    # print(model.config)
    # return
    try:
        model = cls(**cnfg)
    except InvalidConfig as e:
        print(e)
        return
    
    model.print_inputs()
    model.print_outputs()

    # Direct from dummy data
    data = model.test_data()
    result = model(data, verbose=2)
    # result = model(data, n_bat=5, verbose=2)

    model.plot(data | result['popt'])  

    dc.print_quantities({k:v for k, v in data.items() if k in data['pfree']})
    dc.print_quantities(result['popt']) 

    print('Loss (%): ', result['loss'])



def test_inverse_aorta_liver_drug():
    model = dc.InverseAortaLiverDrug
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 'lagut': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'lin'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta_liver_split():
    model = dc.InverseAortaLiverSplit
    cnfg = {'sequence': '2D-SR-SPGR', 'tof_corr': False, 'inflow': 'none', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'N', 'baseline': 'measured', 'bolus': 'dual', 'heartlung': 'pfcomp', 'organs': '2cxm', 'lagut': 'comp', 'liver': '1I-IC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'quad'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta_liver_split_drug():
    model = dc.InverseAortaLiverSplitDrug
    cnfg = {'sequence': '3D-SPGR-SS', 'tof_corr': False, 'inflow': 'none', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 'lagut': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'lin'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta_liver():
    model = dc.InverseAortaLiver
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 'lagut': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_li': 'lin'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta():
    model = dc.InverseAorta
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'inflow': 'none', 'sequence': '3D-SR-SS', 'tof_corr': False, 'magnitude': False, 'trigger': True, 'calibrate': True, 'baseline': 'measured', 'heartlung': 'comp', 'organs': 'comp', 'kidneys': 'plug', 'liver': 'plug', 'lagut': 'pass', 'bolus': 'double'} 
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_liver():
    model = dc.InverseLiver
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'inflow': 'none', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'kinetics': '1I-EC', 'non_stationary': None, 'input': True}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta_kidneys():
    model = dc.InverseAortaKidneys
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'heartlung': 'pfcomp', 'organs': 'comp', 'kidneys': '2CF', 'bolus': 'single','t1_relaxation_ao': 'lin', 't1_relaxation_lk': 'lin', 't1_relaxation_rk': 'lin', 't2_relaxation_ao': None, 't2_relaxation_lk': None, 't2_relaxation_rk': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_lk': 'lin', 't2s_relaxation_rk': 'lin'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_aorta_portal_liver():
    model = dc.InverseAortaPortalLiver
    cnfg = {'inflow': 'none', 'sequence': '3D-SPGR-SS', 'tof_corr': False, 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'bolus': 'single', 'heartlung': 'pfcomp', 'organs': 'comp', 'liver': '1I-EC', 'non_stationary': None, 't1_relaxation_ao': 'lin', 't1_relaxation_pv': 'lin', 't1_relaxation_li': 'lin', 't2_relaxation_ao': None, 't2_relaxation_pv': None, 't2_relaxation_li': None, 't2s_relaxation_ao': 'lin', 't2s_relaxation_pv': 'lin', 't2s_relaxation_li':'lin'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_cort_med():
    model = dc.InverseCortMed
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'sequence': '3D-SPGR-SS', 'magnitude': True, 'trigger': False, 'calibrate': False, 'water_exchange': 'F', 'baseline': 'literature', 'kinetics': '7C'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_kidney():
    model = dc.InverseKidney
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': None, 'inflow': 'pool', 'sequence': 'ZTE-3D-SPGR-SS', 'magnitude': False, 'trigger': False, 'calibrate': True, 'water_exchange': 'R', 'baseline': 'measured', 'kinetics': 'HFU'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_tissue_x():
    model = dc.InverseTissueX
    cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'lin', 'inflow': 'none', 'sequence': '3D-SPGR', 'magnitude': False, 'trigger': False, 'calibrate': False, 'water_exchange': 'FF', 'kinetics': 'WV', 'baseline': 'literature'}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)

def test_inverse_tissue_ls():
    model = dc.InverseTissueLS
    cnfg = {'sequence': '3D-SPGR-SS', 'calibrate': True}
    test_inverse_model_instance(model, cnfg)
    test_inverse_model(model, simple=True, io_sample=1e5, cnfg_sample=1e5, seed=51)


if __name__ == '__main__':

    # test_inverse_aorta_liver_drug()
    # test_inverse_aorta_liver_split()
    # test_inverse_aorta_liver_split_drug()
    # test_inverse_aorta_liver()
    # test_inverse_aorta()
    # test_inverse_liver()
    # test_inverse_aorta_kidneys()
    # test_inverse_aorta_portal_liver()
    # test_inverse_cort_med()
    # test_inverse_kidney()
    # test_inverse_tissue_x()
    test_inverse_tissue_ls()

    print('All inverse model tests passed!!')