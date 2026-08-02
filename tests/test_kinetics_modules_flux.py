import itertools

import numpy as np
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration


def test_flux():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.Flux.configs.keys())}
        try:
           flux = dc.Flux(**cnfg)
        except ValueError:
            return
        data = {k: 1 for k in flux.inputs()}
        data['J'] = np.ones(5)
        data['h'] = [1]
        data['TT'] = [0, 1]
        if cnfg['block'] in ['bicomp', 'plucom', 'ncomp', '2cxm']:
            data['T'] = [1, 1]
        if cnfg['block'] == 'ncomp':
            data['J'] = np.ones((2,5))
            data['E'] = [[1, 1], [1, 1]]
        if cnfg['block'] == 'nscomp':
            data['T'] = np.ones(5)
        flux(data)

    values = dc.Flux.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]

def test_flux_tissue_x():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.FluxTissueX.configs.keys())}
        try:
           flux = dc.FluxTissueX(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.inputs()}
        data['ca'] = np.ones(5)
        flux(data)

    values = dc.FluxTissueX.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]


def test_flux_injection():
    def _test_config(cnfg):
        cnfg = {k: cnfg[i] for i, k in enumerate(dc.FluxInjection.configs.keys())}
        try:
           flux = dc.FluxInjection(**cnfg)
        except ValueError:
            return
        data = {k: dc.QVALUES[k] for k in flux.mapped_inputs()}
        flux(data)

    values = dc.FluxInjection.configs.values()
    [
        _test_config(cnfg) 
        for cnfg in itertools.product(*values)
    ]


def test_flux_aorta():
    def _test_config(cnfg):
        print(cnfg)
        try:
           flux = dc.FluxAorta(**cnfg)
        except InvalidConfiguration:
            return
        data = {k: dc.QVALUES[k] for k in flux.mapped_inputs()}
        flux(data)

    cnt = 0
    for cnfg in dc.FluxAorta.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} FluxAorta configurations!')



if __name__ == '__main__':
    test_flux()
    test_flux_injection()
    test_flux_aorta()
    test_flux_tissue_x()
    
    print('All kinetics flux tests passed!!')