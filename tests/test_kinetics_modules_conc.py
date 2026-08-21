import numpy as np
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration

def test_conc_aorta():
    def _test_config(cnfg):
        print('ConcAorta', cnfg)
        try:
           conc = dc.ConcAorta(**cnfg)
        except InvalidConfiguration:
            return
        data = dc.QVALUES | conc.lexicon_data(dc.QVALUES)
        c = conc(data)
        assert c['C_a'].ndim==2

    cnt = 0
    for cnfg in dc.ConcAorta.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcAorta configurations!')

def test_conc_aorta_liver():
    def _test_config(cnfg):
        print('ConcAortaLiver', cnfg)
        try:
           conc = dc.ConcAortaLiver(**cnfg)
        except InvalidConfiguration:
            return
        data = dc.QVALUES | conc.lexicon_data(dc.QVALUES)
        c = conc(data)
        assert c['C_l'].ndim==2

    cnt = 0
    for cnfg in dc.ConcAortaLiver.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcAortaLiver configurations!')

def test_conc_aorta_portal_liver():
    def _test_config(cnfg):
        print('ConcAortaPortalLiver', cnfg)
        try:
           conc = dc.ConcAortaPortalLiver(**cnfg)
        except InvalidConfiguration:
            return
        data = dc.QVALUES | conc.lexicon_data(dc.QVALUES)
        c = conc(data)
        assert c['C_l'].ndim==2

    cnt = 0
    for cnfg in dc.ConcAortaPortalLiver.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcAortaPortalLiver configurations!')

def test_conc_aorta_kidneys():
    def _test_config(cnfg):
        print('ConcAortaKidneys', cnfg)
        try:
           conc = dc.ConcAortaKidneys(**cnfg)
        except InvalidConfiguration:
            return
        data = dc.QVALUES | conc.lexicon_data(dc.QVALUES)
        c = conc(data)
        assert c['C_a'].ndim==2
        assert c['C_lk'].ndim==2
        assert c['C_rk'].ndim==2

    cnt = 0
    for cnfg in dc.ConcAortaKidneys.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcAortaKidneys configurations!')





def test_conc_cortmed():
    def _test_config(cnfg):
        print('ConcCortMed', cnfg)
        try:
           conc = dc.ConcCortMed(**cnfg)
        except InvalidConfiguration:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['Cc'].ndim==2
        assert c['Cm'].ndim==2

    cnt = 0
    for cnfg in dc.ConcCortMed.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcCortMed configurations!')

def test_conc_kidney():
    def _test_config(cnfg):
        print('ConcKidney', cnfg)
        try:
           conc = dc.ConcKidney(**cnfg)
        except InvalidConfiguration:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['Ck'].ndim==2

    cnt = 0
    for cnfg in dc.ConcKidney.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcKidney configurations!')

def test_conc_liver():
    def _test_config(cnfg):
        print('ConcLiver', cnfg)
        try:
           conc = dc.ConcLiver(**cnfg)
        except InvalidConfiguration:
            return
        data = dc.QVALUES | conc.lexicon_data(dc.QVALUES)
        data['ci_l'] = np.ones(5) if '1I' in cnfg['kinetics'] else (np.ones(5), np.ones(5))
        c = conc(data)
        assert c['C_l'].ndim==2

    cnt = 0
    for cnfg in dc.ConcLiver.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcLiver configurations!')

def test_conc_tissue_x():
    def _test_config(cnfg):
        print('ConcTissueX', cnfg)
        try:
           conc = dc.ConcTissueX(**cnfg)
        except InvalidConfiguration:
            return
        data = {k: dc.QVALUES[k] for k in conc.inputs()}
        data['ca'] = np.ones(5)
        c = conc(data)
        assert c['C'].ndim==2

    cnt = 0
    for cnfg in dc.ConcTissueX.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} ConcLiver configurations!')


if __name__ == '__main__':
    test_conc_aorta_liver()
    test_conc_aorta_portal_liver()
    test_conc_aorta_kidneys()
    test_conc_aorta()
    test_conc_cortmed()
    test_conc_kidney()
    test_conc_liver()
    test_conc_tissue_x()

    print('All conc tests passed!!')