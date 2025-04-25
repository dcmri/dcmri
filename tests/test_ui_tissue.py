import os
import shutil
import numpy as np
import dcmri as dc



# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False




def test_ui_tissue():

    # Create data
    time, aif, roi, gt = dc.fake_tissue()

    params = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
        'aif': aif,
        'dt': time[1],
        # 'ca': gt['cb'],
        # 't': gt['t'],
        'r1': dc.relaxivity(3,'blood','gadodiamide'),
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10a': 1/dc.T1(3.0,'blood'),
        'R10': 1/dc.T1(3.0,'muscle'),
    }

    # Train model and check results
    model = dc.Tissue(**params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    pars = model.export_params()
    cost0 = model.cost(time, roi)
    assert cost0 < 1 # perc
    assert np.abs(pars['Fp'][1]-gt['Fp']) < 0.1*gt['Fp']
    assert np.abs(pars['vp'][1]-gt['vp']) < 0.1*gt['vp']

    # Try again fixing up to the wrong range
    # Check that the fit is less good
    # And that the value runs against the upper bound
    model = dc.Tissue(vb=0.001, **params)
    model.set_free(vb = [0., 0.01])
    model.train(time, roi)
    pars = model.export_params()
    assert np.round(pars['vb'][1],3)==0.01
    assert model.cost(time, roi) > cost0

    params['water_exchange'] = 'NN'
    model = dc.Tissue(**params)
    model.train(time, roi)
    assert model.cost(time, roi) < 1.5

    params['water_exchange'] = 'RR'
    model = dc.Tissue(**params)
    model.train(time, roi, xtol=1e-1)
    assert model.cost(time, roi) < 1
    assert model.params('PSe') > 1 # fast exchange
    assert model.params('PSc') > 1

    # Loop over all models
    # cost = {'U':20, 'NX':10, 'FX':20, 'WV':10, 'HFU':20, 'HF':6, '2CU':20, 
    #         '2CX':2}
    for k in ['U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        for e in ['R','F','N']:
            for c in ['R','F','N']:
                params['water_exchange'] = e+c
                params['kinetics'] = k
                model = dc.Tissue(**params)
                model.train(time, roi, xtol=1e-1)
                #assert model.cost(time, roi) < cost[k]

    # Display last result
    model.plot(time, roi, ref=gt, show=SHOW)
    


def test_ui_tissue_array():

    # Create data
    n=8
    time, signal, aif, gt = dc.fake_brain(n=n, verbose=VERBOSE)

    with np.errstate(divide='ignore'):
        R10 = np.where(gt['T1']==0, 0, 1/gt['T1'])

    # Define model parameters
    params = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
        'aif': aif,
        'dt': time[1],
        'r1': dc.relaxivity(3, 'blood', 'gadodiamide'),
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10a': 1/dc.T1(3.0,'blood'),
        'R10': R10,
        'parallel': False,
        'verbose': VERBOSE,
    }

    # Train array
    image = dc.TissueArray((n,n), **params)
    image.train(time, signal, xtol=1e-4)
    pars = image.export_params()
    #image.save(path=tmp(), filename='TissueArray')
    #image = dc.TissueArray().load(filename=file)

    # Plot array
    roi = dc.shepp_logan(n=n)
    vmin = {'S0':0, 'Fb':0, 'vb':0, 'PS':0, 'vi':0}
    vmax = {'S0':np.amax(gt['S0']), 'Fb':0.02, 'vb':0.2, 'PS':0.003, 'vi':0.5}
    image.plot(time, signal, vmax=vmax, ref=gt, show=SHOW)
    image.plot_params(roi=roi, ref=gt, vmin=vmin, vmax=vmax, show=SHOW)
    image.plot_fit(time, signal, ref=gt, roi=roi, show=SHOW,
#       hist_kwargs = {'bins':100, 'range':[0,10]},
   )

    # Compare against curve fit in one pixel.
    loc = dc.shepp_logan(n=n)['gray matter'] == 1
    loc = dc.shepp_logan(n=n)['bone'] == 1
    signal_loc = signal[loc,:][0,:]
    params['FA'] = 15
    params['R10'] = R10[loc][0]
    params.pop('parallel')
    params.pop('verbose')
    curve = dc.Tissue(**params)
    curve.train(time, signal_loc, xtol=1e-4)
    curve.plot(time, signal_loc, show=SHOW)

    nrmsc = curve.cost(time, signal_loc)
    nrmsa = image.cost(time, signal)[loc][0]
    assert np.abs(nrmsc-nrmsa) <= 0.1*nrmsc
    cp = curve.params('S0','Fb','vb','PS','vi')
    ip = image.params('S0','Fb','vb','PS','vi')
    for i in range(len(cp)):
        assert np.abs(cp[i] - ip[i][loc][0]) <= 0.1*cp[i]






if __name__ == "__main__":


    test_ui_tissue()
    # test_ui_tissue_array()

    print('All ui_tissue tests passed!!')