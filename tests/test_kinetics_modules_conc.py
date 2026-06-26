import itertools

import numpy as np
import dcmri as dc

def test_conc_aorta_liver():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcAortaLiver.configs.keys())}
        try:
           flux = dc.ConcAortaLiver(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcAortaLiver.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_aorta_kidneys():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcAortaKidneys.configs.keys())}
        try:
           flux = dc.ConcAortaKidneys(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcAortaKidneys.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_aorta():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcAorta.configs.keys())}
        try:
           flux = dc.ConcAorta(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcAorta.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_cortmed():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcCortMed.configs.keys())}
        try:
           flux = dc.ConcCortMed(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcCortMed.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_kidney():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcKidney.configs.keys())}
        try:
           flux = dc.ConcKidney(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcKidney.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_liver():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcLiver.configs.keys())}
        try:
           flux = dc.ConcLiver(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.ConcLiver.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_conc_tissue_x():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.ConcTissueX.configs.keys())}
        try:
           flux = dc.ConcTissueX(cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

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