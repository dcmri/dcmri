import os
import shutil

import numpy as np
import dcmri as dc



# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False


def test_ui_aorta_liver_docstring():
    time, aif, roi, gt = dc.fake_tissue2scan(R10=1/dc.T1(3.0,'liver'))

    xdata = (time[0], time[1], time[0], time[1])
    ydata = (aif[0], aif[1], roi[0], roi[1])

    model = dc.AortaLiver2scan(
        dt = 0.5,
        tmax = 420,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        dose2 = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        FA2 = 15,
        TS = 0.5,
        Th_i = 120,
        Th_f = 120,
    )
    bounds = {'Th_i':[0, 1e9], 'Th_f':[0, 1e9]}
    model.train(xdata, ydata, bounds=bounds, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)



def test_ui_aorta_liver2scan():

    time, aif, roi, gt = dc.fake_tissue2scan(R10 = 1/dc.T1(3.0,'liver'))
    xdata = (time[0], time[1], time[0], time[1])
    ydata = (aif[0], aif[1], roi[0], roi[1])

    model = dc.AortaLiver2scan(
        dt = 0.5,
        tmax = max(time[1]) + 10,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        dose2 = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
        Th_i = 120,
        Th_f = 120,
    )
    bounds = {'Th_i':[0, 1e9], 'Th_f':[0, 1e9]}
    model.train(xdata, ydata, bounds=bounds, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 5
    assert 60 < model.params('Th', round_to=0) < 80
    model.print_params(round_to=3)




if __name__ == "__main__":

    test_ui_aorta_liver_docstring()
    test_ui_aorta_liver2scan()

    print('All ui_liver tests passed!!')