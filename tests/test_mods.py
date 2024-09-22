import os
import shutil
import numpy as np
import dcmri as dc


# Debugging mode
# VERBOSE = 1
# SHOW = True

VERBOSE = 0
SHOW = False


def tmp():
    return os.path.join(os.getcwd(),'tmp')

def make_tmp():
    os.makedirs(tmp(), exist_ok=True)

def delete_tmp():
    shutil.rmtree(tmp())

class VoidModel(dc.Model):
    pass

class TestModel(dc.Model):
    def __init__(self):
        self.a=1
        self.b=2
        self.c=[3,4]
        self.d=5*np.ones((2,3))
        self.free = []
        self.bounds = [-np.inf,np.inf]

def test_model():

    t = VoidModel()
    try:
        t.predict(None)
    except:
        assert True

    t = TestModel()
    a = [1,2,3,4] + 6*[5]
    assert np.array_equal(t._getflat(), [])
    assert np.array_equal(t._getflat(['a','b','c','d']), a)
    f = t._getflat(['b','c','d'])
    f[-1] = 0
    t._setflat(f, ['b','c','d'])
    a = [1,2,3,4] + 5*[5] + [0]
    assert np.array_equal(t._getflat(['a','b','c','d']), a)

    make_tmp()
    t.save(path=tmp())
    t.a=2
    assert t.a==2
    t.load(path=tmp())
    assert t.a==1
    delete_tmp()


def test_mods_tissue():

    # Create data
    time, aif, roi, gt = dc.fake_tissue()

    params = {
        'kinetics': '2CX',
        'water_exchange': 'FF',
        'aif': aif,
        'dt': time[1],
        'agent': 'gadodiamide',
        'TR': 0.005,
        'FA': 15,
        'FAa': 15,
        'n0': 10,
        'R10b': 1/dc.T1(3.0,'blood'),
        'R10': 1/dc.T1(3.0,'muscle'),
    }

    model = dc.Tissue(**params)

    # Train model and check results
    model.train(time, roi)
    pars = model.export_params()
    assert model.cost(time, roi) < 1 # perc
    assert np.abs(pars['Fp'][1]-gt['Fp']) < 0.1*gt['Fp']
    assert np.abs(pars['vp'][1]-gt['vp']) < 0.1*gt['vp']

    params['water_exchange'] = 'NN'
    model = dc.Tissue(**params)
    model.train(time, roi)
    assert model.cost(time, roi) < 1

    params['water_exchange'] = 'RR'
    model = dc.Tissue(**params)
    model.train(time, roi, xtol=1e-2)
    assert model.cost(time, roi) < 1
    assert model.get_params('PSe') > 1 # fast exchange
    assert model.get_params('PSc') > 1

    # Loop over all models
    cost = {'U':20, 'NX':10, 'FX':20, 'WV':10, 'HFU':20, 'HF':6, '2CU':20, '2CX':2}
    for k in ['U', 'NX', 'FX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        for e in ['R','F','N']:
            for c in ['R','F','N']:
                params['water_exchange'] = e+c
                params['kinetics'] = k
                model = dc.Tissue(**params)
                model.train(time, roi, xtol=1e-2)
                assert model.cost(time, roi) < cost[k]

    # Display last result
    model.plot(time, roi, ref=gt, show=False)
    


def test_mods_tissue_array():

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
        'agent': 'gadodiamide',
        'TR': 0.005,
        'FA': np.full((n,n), 15),
        'FAa': 15,
        'n0': 10,
        'R10b': 1/dc.T1(3.0,'blood'),
        'R10': R10,
        'parallel': False,
        'verbose': VERBOSE,
    }

    # Train array
    image = dc.TissueArray((n,n), **params)
    image.train(time, signal, xtol=1e-3)
    #image.save(path=tmp(), filename='TissueArray')
    #image = dc.TissueArray().load(filename=file)

    # Plot array
    roi = dc.shepp_logan(n=n)
    vmin = {'S0':0, 'Fp':0, 'vp':0, 'Ktrans':0, 've':0}
    vmax = {'S0':np.amax(gt['S0']), 'Fp':0.01, 'vp':0.2, 'Ktrans':0.003, 've':0.5}
    image.plot(time, signal, vmax=vmax, ref=gt, show=SHOW)
    image.plot_params(roi=roi, ref=gt, vmin=vmin, vmax=vmax, show=SHOW)
    image.plot_fit(time, signal, ref=gt, roi=roi, show=SHOW,
#       hist_kwargs = {'bins':100, 'range':[0,10]},
   )

    # Compare against curve fit in one pixel.
    loc = dc.shepp_logan(n=n)['gray matter'] == 1
    signal_loc = signal[loc,:][0,:]
    params['FA'] = 15
    params['R10'] = R10[loc][0]
    curve = dc.Tissue(**params)
    curve.train(time, signal_loc, xtol=1e-3)
    curve.plot(time, signal_loc, show=SHOW)

    nrmsc = curve.cost(time, signal_loc)
    nrmsa = image.cost(time, signal)[loc][0]
    assert np.abs(nrmsc-nrmsa) <= 0.1*nrmsc
    assert np.abs(curve.S0 - image.S0[loc][0]) <= 0.1*curve.S0
    assert np.abs(curve.Fp - image.Fp[loc][0]) <= 0.1*curve.Fp
    assert np.abs(curve.vp - image.vp[loc][0]) <= 0.1*curve.vp
    assert np.abs(curve.Ktrans - image.Ktrans[loc][0]) <= 0.1*curve.Ktrans
    assert np.abs(curve.vi - image.vi[loc][0]) <= 0.1*curve.vi



def test_mods_aorta():

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
        R10 = 1/dc.T1(3.0,'blood'), 
        heartlung = 'chain',
    )
    aorta.train(time, aif, xtol=1e-3)
    aorta.plot(time, aif, show=SHOW)
    rec = aorta.export_params()
    rec_aif = aorta.predict(time)
    assert rec['BAT'][1] == aorta.get_params('BAT')
    assert np.linalg.norm(aif-rec_aif) < 0.1*np.linalg.norm(aif)
    assert np.abs(rec['BAT'][1]-truth['BAT']) < 0.2*truth['BAT']


def test_mods_aorta_liver():

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
        bounds = {'Th': [0, np.inf]},
    )
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 10
    assert model.get_params('Th', round_to=0) == 68


def test_mods_aorta_liver2scan():
    time, aif, roi, gt = dc.fake_tissue2scan(R10 = 1/dc.T1(3.0,'liver'))
    xdata = (time[0], time[1], time[0], time[1])
    ydata = (aif[0], aif[1], roi[0], roi[1])

    model = dc.AortaLiver2scan(
        dt = 0.5,
        weight = 70,
        agent = 'gadodiamide',
        dose = [0.2,0.2],
        rate = 3,
        field_strength = 3.0,
        t0 = 10,
        TR = 0.005,
        FA = 15,
        kinetics = 'stationary',
        bounds = {'Th':[0, np.inf]},
    )
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=SHOW)
    assert model.cost(xdata, ydata) < 5
    assert model.get_params('Th', round_to=0) == 73

def test_mods_liver():
    time, aif, roi, gt = dc.fake_tissue(
        agent='gadoxetate', 
        R10=1/dc.T1(3.0,'liver'),
    )
    model = dc.Liver(
        aif = aif,
        dt = time[1],
        Hct = 0.45,
        agent = 'gadoxetate',
        TR = 0.005,
        FA = 15,
        n0 = 10,
        kinetics = 'stationary',
        bounds = {'Th':[0, np.inf]},
    )
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1
    assert model.get_params('Th', round_to=0) == 69

def test_mods_kidney():
    time, aif, roi, gt = dc.fake_tissue(R10=1/dc.T1(3.0,'kidney'))
    #
    # Override the parameter defaults to match the experimental conditions of the synthetic test data:
    #
    params = {
        'aif':aif,
        'dt':time[1],
        'agent': 'gadodiamide',
        'TR': 0.005,
        'FA': 15,
        'R10': 1/dc.T1(3.0,'kidney'),
        'n0': 15,
    }
    #
    # Train a two-compartment filtration model on the ROI data and plot the fit:
    #
    params['kinetics'] = '2CFM'
    model = dc.Kidney(**params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1
    assert model.get_params('Tt', round_to=0) == 67
    #
    # Repeat the fit using a free nephron model:
    #
    params['kinetics'] = 'FN'
    model = dc.Kidney(**params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 1
    assert model.get_params('Fp', round_to=2) == 0.06


def test_mods_kidney_cort_med():

    time, aif, roi, gt = dc.fake_kidney_cortex_medulla()

    model = dc.KidneyCortMed(
        aif = aif,
        dt = time[1],
        agent = 'gadoterate',
        TR = 0.005,
        FA = 15,
        TC = 0.2,
        n0 = 10,
    )

    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)

    assert model.cost(time, roi) < 1
    assert model.get_params('Tdt', round_to=0) == 27



if __name__ == "__main__":

    # make_tmp()

    test_model()
    test_mods_tissue()
    test_mods_tissue_array()
    test_mods_aorta()
    test_mods_aorta_liver()
    test_mods_aorta_liver2scan()
    test_mods_liver()
    test_mods_kidney()
    test_mods_kidney_cort_med()

    print('All mods tests passed!!')