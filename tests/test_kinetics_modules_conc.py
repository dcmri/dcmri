import itertools

import numpy as np
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration

def test_conc_aorta_liver():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcAortaLiver.configs.keys())}
        try:
           conc = dc.ConcAortaLiver(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        for k, v in c.items():
            if k=='Cl':
                assert v.ndim==2
            else:
                assert v.ndim==1

    values = dc.ConcAortaLiver.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_aorta_kidneys():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcAortaKidneys.configs.keys())}
        try:
           conc = dc.ConcAortaKidneys(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['ca'].ndim==1
        assert c['Clk'].ndim==2
        assert c['Crk'].ndim==2

    values = dc.ConcAortaKidneys.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_aorta():
    def _test_config(cnfg):
        print('ConcAorta', cnfg)
        try:
           conc = dc.ConcAorta(**cnfg)
        except InvalidConfiguration:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        c = conc(data)
        assert c['ca'].ndim==1

    cnt = 0
    for cnfg in dc.ConcAorta.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcAorta configurations!')



def test_conc_cortmed():
    def _test_config(cnfg):
        try:
           conc = dc.ConcCortMed(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['Cc'].ndim==2
        assert c['Cm'].ndim==2

    [
        _test_config(cnfg) 
        for cnfg in dc.ConcCortMed.configurations()
    ]

def test_conc_kidney():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcKidney.configs.keys())}
        try:
           conc = dc.ConcKidney(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['Ck'].ndim==2

    values = dc.ConcKidney.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_liver():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcLiver.configs.keys())}
        print(cnfg)
        try:
           conc = dc.ConcLiver(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ci'] = np.ones(5) if '1I' in cnfg['kinetics'] else (np.ones(5), np.ones(5))
        c = conc(data)
        assert c['Cl'].ndim==2

    values = dc.ConcLiver.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_tissue_x():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcTissueX.configs.keys())}
        try:
           conc = dc.ConcTissueX(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['C'].ndim==2

    values = dc.ConcTissueX.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]


if __name__ == '__main__':
    test_conc_aorta_liver()
    test_conc_aorta_kidneys()
    test_conc_aorta()
    test_conc_cortmed()
    test_conc_kidney()
    test_conc_liver()
    test_conc_tissue_x()

    print('All conc tests passed!!')