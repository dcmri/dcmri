import os
import shutil

import numpy as np
import dcmri as dc



# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False



def test_ui_aorta_portal_liver():

    time, aif, vif, roi, gt = dc.fake_liver(sequence='SSI')
    
    model = dc.AortaPortalLiver(
        sequence = 'SSI',
        kinetics='2I-IC',
        dt = 0.5,
        tmax = max(time) + 10,
        weight = 70,
        agent = 'gadoxetate',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
    )

    xdata, ydata = (time, time, time), (aif, vif, roi)
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)
    assert 4 < model.cost(xdata, ydata) < 5



if __name__ == "__main__":

    test_ui_aorta_portal_liver()

    print('All ui_liver tests passed!!')