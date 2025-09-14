import os
import shutil
import numpy as np
import dcmri as dc


# Debugging mode
VERBOSE = 1
SHOW = True

# Production mode
VERBOSE = 0
SHOW = False



def test_ui_tissue_ls():

    # Create data
    time, aif, roi, gt = dc.fake_tissue()
    gt['ve'] = gt['vp'] + gt['vi'] if gt['PS'] > 0 else gt['vp']

    params = {
        'aif': aif,
        'dt': time[1],
        'sequence': 'SS',
        'r1': dc.relaxivity(3,'blood','gadodiamide'),
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10a': 1/dc.T1(3.0,'blood'),
        'R10': 1/dc.T1(3.0,'muscle'),
    }

    # Train model and check results
    model = dc.TissueLS(**params)
    model.train(roi, tol=0.01)
    model.plot(roi, show=SHOW)
    pars = model.params()
    
    assert np.abs(pars['Fb']-gt['Fb'])/gt['Fb'] < 0.1
    assert np.abs(pars['ve']-gt['ve'])/gt['ve'] < 0.2



def test_ui_tissue_ls_array():

    # Create ground truth data
    n=16
    time, signal, aif, gt = dc.fake_brain(n=n, verbose=VERBOSE)
    with np.errstate(divide='ignore'):
        R10 = np.where(gt['T1']==0, 0, 1/gt['T1'])
    vmin = {'Fb':0, 've':0, 'S0':0}
    vmax = {'Fb':0.02, 've':0.2, 'S0':np.amax(gt['S0'])}
    gt['ve'] = np.where(gt['PS'] > 0, gt['vp'] + gt['vi'], gt['vp'])
    gt['Te'] = np.where(gt['Fp']==0, 0, gt['ve']/gt['Fp'])

    # Define model parameters
    params = {
        'aif': aif,
        'dt': time[1],
        'sequence': 'SS',
        'r1': dc.relaxivity(3,'blood','gadodiamide'),
        'TR': 0.005,
        'FA': 15,
        'n0': 10,
        'R10a': 1/dc.T1(3.0,'blood'),
        'R10': R10,
    }

    # Train array and show results
    image = dc.TissueLSArray((n,n), **params)
    image.train(signal, tol=0.01)
    image.plot(signal, vmin=vmin, vmax=vmax, ref=gt, show=SHOW)
    assert np.linalg.norm(image.params('Fb')-gt['Fb'])/np.linalg.norm(gt['Fb']) < 0.1

    # Repeat with linear model (different than ground truth model -> less accurate result)
    params['sequence'] = 'lin'
    image = dc.TissueLSArray((n,n), **params)
    image.train(signal, tol=0.01)
    image.plot(signal, vmin=vmin, vmax=vmax, ref=gt, show=SHOW)
    assert np.linalg.norm(image.params('Fb')-gt['Fb'])/np.linalg.norm(gt['Fb']) < 1


if __name__ == "__main__":

    test_ui_tissue_ls()
    test_ui_tissue_ls_array()

    print('All ui_tissue tests passed!!')