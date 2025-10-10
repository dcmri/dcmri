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
    #
    # Use `fake_tissue` to generate synthetic test data from
    # experimentally-derived concentrations:
    #
    # Use `fake_liver` to generate synthetic test data:
    #
    time, aif, vif, roi, gt = dc.fake_liver()
    #
    # Since this model generates two time curves, the x- and y-data are
    # tuples:
    #
    xdata, ydata = (time,time), (aif,roi)
    #
    # Build an aorta-liver model and parameters to match the
    # conditions of the fake liver data:
    #
    model = dc.AortaLiver(
        dt = 0.5,
        tmax = 180,
        weight = 70,
        agent = 'gadoxetate',
        field_strength = 3.0,
        dose = 0.2,
        rate = 3,
        TR = 0.005,
        FA = 15,
    )
    #
    # Train the model on the data:
    #
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    #
    # Plot the reconstructed signals and concentrations and compare
    # against the experimentally derived data:
    #
    model.plot(xdata, ydata, show=SHOW)
    #
    # We can also have a look at the model parameters after training:
    #
    model.print_params(round_to=3)



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
    model.free['Th_i'] = [0, np.inf]
    model.free['Th_f'] = [0, np.inf]
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    model.print_params(round_to=3)



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
        TR = 0.005,
        FA = 15,
        TS = 0.5,
        Th = 120,
    )
    model.free['Th'] = [0, np.inf]
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 10
    assert 90 < model.params('Th', round_to=0) < 100


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
    model.free['Th_i'] = [0, np.inf]
    model.free['Th_f'] = [0, np.inf]
    model.train(xdata, ydata, n0=10, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 5
    assert 60 < model.params('Th', round_to=0) < 80
    model.print_params(round_to=3)


def test_ui_liver():
    time, aif, vif, roi, gt = dc.fake_liver()

    # Show dual-inlet model
    params = {
        'kinetics': '2I-IC',
        't': time,
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),
        'R10v': 1/dc.T1(3.0, 'blood'),
    }
    model = dc.Liver(**params)
    model.train(time, roi, aif, vif, n0=10)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 0.1

    # Show single-inlet model
    params = {
        'kinetics': '1I-IC-HFD',
        't': time,
        'H': 0.45,
        'field_strength': 3,
        'agent': 'gadoxetate',
        'TR': 0.005,
        'FA': 15,
        'R10': 1/dc.T1(3.0,'liver'),
        'R10a': 1/dc.T1(3.0, 'blood'),        
    }
    model = dc.Liver(**params)
    model.train(time, roi, aif, n0=10)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1.5
    pars = model.export_params()
    assert 0.2 < pars['ve'][1] < 0.3
    model.print_params(round_to=3)

    # Loop over all models
    for k in ['2I-EC', '2I-EC-HF', '1I-EC', '1I-EC-D', 
              '2I-IC', '2I-IC-HF', '2I-IC-U', '1I-IC-HF', 
              '1I-IC-HFD', '1I-IC-HFDU']:
        params['kinetics'] = k
        if '-EC' in k:
            non_stat = [None]
        elif k not in ['2I-IC-U', '1I-IC-HFDU']:
            non_stat = ['UE','U','E', None]
        else:
            non_stat = ['U', None]
        for ns in non_stat:
            params['non_stationary'] = ns
            model = dc.Liver(**params)
            if k[0]=='2':
                model.train(time, roi, aif, vif, n0=10, xtol=1e-2)
            else:
                model.train(time, roi, aif, n0=10, xtol=1e-2)
            model.export_params()
            assert model.cost(time, roi) < 25

    # Display last result
    model.plot(time, roi, ref=gt, show=SHOW)


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

    test_ui_aorta_liver_docstring()
    test_ui_liver()
    test_ui_aorta_liver()
    test_ui_aorta_liver2scan()
    test_ui_aorta_portal_liver()

    print('All ui_liver tests passed!!')