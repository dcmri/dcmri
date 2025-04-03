import numpy as np
import dcmri as dc


# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False


def test_ui_aorta():

    truth = {'BAT': 20}
    time, aif, _, _ = dc.fake_tissue(**truth)
    aorta = dc.Aorta(
        dt = 1.5,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 1.5,
        R10 = 1/dc.T1(3.0,'blood'), 
        heartlung = 'chain',
    )
    aorta.train(time, aif, xtol=1e-3)
    aorta.plot(time, aif, show=SHOW)
    rec = aorta.export_params()
    rec_aif = aorta.predict(time)
    assert rec['BAT'][1] == aorta.params('BAT')
    assert np.linalg.norm(aif-rec_aif) < 0.1*np.linalg.norm(aif)
    assert np.abs(rec['BAT'][1]-truth['BAT']) < 0.2*truth['BAT']


if __name__ == "__main__":


    test_ui_aorta()

    print('All ui_aorta tests passed!!')