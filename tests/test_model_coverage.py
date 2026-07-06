import numpy as np
import dcmri as dc

from dcmri.core.exceptions import InvalidConfiguration


def test_aorta():
    def _test_config(cnfg):
        # if cnfg != {'bolus': 'dual', 'heartlung': 'comp', 'organs': 'comp', 't2s_relaxation': None, 'sequence': 'Eq-SE-EPI', 'magnitude': False}:
        #     return
        print(cnfg)
        try:
           model = dc.AortaModel(**cnfg)
        except InvalidConfiguration:
            return
        data = model.map_lexicon(dc.QVALUES)
        data['tacq'] = 2 * np.arange(30)
        results = model(data)
        assert results['Sa'].ndim == 3

    cnt = 0
    for cnfg in dc.AortaModel.configurations():
        cnt += 1
        _test_config(cnfg)

    print(f'Successfully covered {cnt} aorta configurations!')


if __name__ == '__main__':
    test_aorta()

    print('All model coverage tests passed!!')