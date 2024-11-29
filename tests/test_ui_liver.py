import os
import shutil
from copy import deepcopy
import numpy as np
import dcmri as dc



# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False




def test_ui_aorta_liver():

    time, aif, roi, gt = dc.fake_tissue()
    xdata, ydata = (time,time), (aif,roi)

    model = dc.AortaLiver(
        dt = 0.5,
        tmax = 180,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        t0 = 10,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
        Th = 120,
    )
    model.free['Th'] = [0, np.inf]
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 10
    assert 75 < model.params('Th', round_to=0) < 85


def test_ui_aorta_liver2scan():

    time, aif, roi, gt = dc.fake_tissue2scan(R10 = 1/dc.T1(3.0,'liver'))
    xdata = (time[0], time[1], time[0], time[1])
    ydata = (aif[0], aif[1], roi[0], roi[1])

    model = dc.AortaLiver2scan(
        dt = 0.5,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        dose2 = 0.2,
        rate = 3,
        field_strength = 3.0,
        t0 = 10,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
        Th_i = 120,
        Th_f = 120,
    )
    model.free['Th_i'] = [0, np.inf]
    model.free['Th_f'] = [0, np.inf]
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 5
    assert 60 < model.params('Th', round_to=0) < 80

def test_ui_liver():
    time, aif, vif, roi, gt = dc.fake_liver()

    # Show dual-inlet model
    params = {
        'kinetics': '2I-IC',
        'aif': aif,
        'vif': vif,
        'dt': time[1],
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),  
        'R10v': 1/dc.T1(3.0, 'blood'),      
    }
    model = dc.Liver(**params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 0.1

    # Show single-inlet model
    params = {
        'kinetics': '1I-IC-D',
        'aif': aif,
        'dt': time[1],
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),        
    }
    model = dc.Liver(**params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1.5

    # Loop over all models
    for k in ['2I-EC', '2I-EC-HF', '1I-EC', '1I-EC-D', 
              '2I-IC', '2I-IC-HF', '2I-IC-U', '1I-IC-HF', 
              '1I-IC-D', '1I-IC-DU']:
        if k not in ['2I-IC-U', '1I-IC-DU']:
            stat = ['UE','U','E', None]
        else:
            stat = ['U', None]
        for s in stat:
            params_mdl = deepcopy(params)
            if k[0]=='2':
                params_mdl['vif'] = vif
            params_mdl['stationary'] = s
            params_mdl['kinetics'] = k
            model = dc.Liver(**params_mdl)
            model.train(time, roi, xtol=1e-2)
            # print(k, s, model.cost(time, roi))
            assert model.cost(time, roi) < 25

    # Display last result
    model.plot(time, roi, ref=gt, show=SHOW)


def test_ui_aorta_portal_liver():

    time, aif, vif, roi, gt = dc.fake_liver(sequence='SSI')
    xdata, ydata = (time, time, time), (aif, vif, roi)

    model = dc.AortaPortalLiver(
        sequence = 'SSI',
        dt = 0.5,
        tmax = 180,
        weight = 70,
        agent = 'gadoxetate',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        t0 = 10,
        TR = 0.005,
        FA = 15,
        TS = 0.5,
    )
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)
    assert model.cost(xdata, ydata) < 4



if __name__ == "__main__":

    # make_tmp()

    test_ui_aorta_liver()
    test_ui_aorta_liver2scan()
    test_ui_liver()
    test_ui_aorta_portal_liver()

    print('All mods tests passed!!')