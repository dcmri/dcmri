import numpy as np
import dcmri as dc

SHOW = False

def test_mods_tissue():

    # Create data
    time, aif, roi, gt = dc.fake_tissue(CNR=np.inf)

    # Train model
    model = dc.Tissue(
        aif = aif,
        dt = time[1],
        agent = 'gadodiamide',
        TR = 0.005,
        FA = 20,
        kinetics='2CX',
        n0=10,
    )
    ypred = model.predict(time)
    model.train(time, roi)

    # Display results
    model.plot(time, roi, ref=gt, show=SHOW)
    assert np.abs(model.Fp-gt['Fp']) < 0.1*gt['Fp']


def test_mods_tissue_pixel():

    file = 'mods_tissue_pixel'

    # Create data
    n=8
    time, signal, aif, gt = dc.fake_brain(n=n, verbose=1)

    # Define model
    params = {
        'ca': gt['cp'],
        'dt': gt['t'][1],
        'agent': 'gadodiamide',
        'kinetics': '2CX',
        'TS': time[1],
        'TR': 0.005,
    }

    # Train array
    shape = (n,n)
    image = dc.TissueArray(
        shape = shape,
        FA = np.full(shape, 20),
        R10 = 1/gt['T1'], 
        parallel = False, 
        verbose = 1,
        **params, 
    )
    image.train(time, signal, xtol=1e-3)
    image.save(filename=file)
    #image = dc.TissueArray().load(filename=file)

    # Train pixel curve
    region = 'gray matter'
    #region = 'background'
    loc = dc.shepp_logan(n=n)[region] == 1
    curve = dc.Tissue(
        FA = 20,
        R10 = 1/gt['T1'][loc][0], 
        **params,
    )
    curve.train(time, signal[loc,:][0,:], xtol=1e-3)
    curve.plot(time, signal[loc,:][0,:], show=SHOW)

    # Check array against curve fit
    # print('NRMS', curve.cost(time, signal[loc,:][0,:]), image.cost(time, signal)[loc][0])
    # print('S0 ', curve.S0, image.S0[loc][0], gt['S0'][loc][0])
    # print('Fp ', curve.Fp, image.Fp[loc][0], gt['Fp'][loc][0])
    # print('vp ', curve.vp, image.vp[loc][0], gt['vp'][loc][0])
    # print('PS ', curve.PS, image.PS[loc][0], gt['PS'][loc][0])
    # print('ve ', curve.ve, image.ve[loc][0], gt['ve'][loc][0])

    nrmsc = curve.cost(time, signal[loc,:][0,:])
    nrmsa = image.cost(time, signal)[loc][0]
    assert np.abs(nrmsc-nrmsa) <= 0.1*nrmsc
    assert np.abs(curve.S0 - image.S0[loc][0]) <= 0.1*curve.S0
    assert np.abs(curve.Fp - image.Fp[loc][0]) <= 0.1*curve.Fp
    assert np.abs(curve.vp - image.vp[loc][0]) <= 0.1*curve.vp
    assert np.abs(curve.PS - image.PS[loc][0]) <= 0.1*curve.PS
    assert np.abs(curve.ve - image.ve[loc][0]) <= 0.1*curve.ve



def test_mods_tissue_pixel_doc():
    n=8
    time, signal, aif, gt = dc.fake_brain(n)
    shape = (n,n)
    model = dc.TissueArray(
        shape = shape,
        aif = aif,
        dt = time[1],
        agent = 'gadodiamide',
        TR = 0.005,
        FA = np.full(shape, 20),
        R10 = 1/gt['T1'], 
        n0 = 15,
        kinetics = '2CX',
    )
    model.train(time, signal)
    model.plot(time, signal, ref=gt, show=SHOW)
    loss = model.cost(time, signal, 'NRMS')
    assert np.nanmax(loss) < 1



def test_mods_tissue_pixel_plot():
    
    file = 'mods_tissue_pixel'

    # Load trained model
    model = dc.TissueArray().load(filename=file)

    # Get ground truth data
    n=8
    time, signal, aif, gt = dc.fake_brain(n=n, CNR=np.inf, verbose=1)
    roi = dc.shepp_logan(n=n)

    # Plot trained model
    vmin= {
        'S0':0,
        'Fp':0,
        'vp':0,
        'PS':0,
        've':0,
    }
    vmax = {
        'S0':np.amax(gt['S0']),
        'Fp':0.01,
        'vp':0.2,
        'PS':0.003,
        've':0.5,
    }
    show=False
    model.plot_params(roi=roi, ref=gt, vmin=vmin, vmax=vmax, show=SHOW)
    model.plot(time, signal, vmax=vmax, ref=gt, show=SHOW)
    model.plot_fit(time, signal, ref=gt, roi=roi, show=SHOW,
#       hist_kwargs = {'bins':100, 'range':[0,10]},
   )





def test_mods_aorta():
    time, aif, _, _ = dc.fake_tissue()
    aorta = dc.Aorta(
        dt = 1.5,
        weight = 70,
        agent = 'gadodiamide',
        dose = 0.2,
        rate = 3,
        field_strength = 3.0,
        TR = 0.005,
        FA = 20,
        R10 = 1/dc.T1(3.0,'blood'),
    )
    aorta.train(time, aif)
    aorta.plot(time, aif, show=False)
    aorta.print_params(round_to=3)

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
        FA = 20,
    )

    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=False)
    model.print_params(round_to=3)

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
        FA = 20,
    )
    model.train(xdata, ydata, xtol=1e-3)
    model.plot(xdata, ydata, show=False)
    model.print_params(round_to=3)

def test_mods_liver():
    time, aif, roi, gt = dc.fake_tissue(CNR=100, agent='gadoxetate', R10=1/dc.T1(3.0,'liver'))
    model = dc.Liver(
        aif = aif,
        dt = time[1],
        Hct = 0.45,
        agent = 'gadoxetate',
        TR = 0.005,
        FA = 20,
        n0 = 10,
    )
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=False)

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
        'FA': 20,
        'R10': 1/dc.T1(3.0,'kidney'),
        'n0': 15,
    }
    #
    # Train a two-compartment filtration model on the ROI data and plot the fit:
    #
    params['kinetics'] = '2CFM'
    model = dc.Kidney(**params).train(time, roi)
    model.plot(time, roi, ref=gt, show=False)
    #
    # Repeat the fit using a free nephron model:
    #
    params['kinetics'] = 'FN'
    model = dc.Kidney(**params).train(time, roi)
    model.plot(time, roi, ref=gt, show=False)

def test_mods_kidney_cort_med():
    #
    # Use `fake_kidney_cortex_medulla` to generate synthetic test data:
    #
    time, aif, roi, gt = dc.fake_kidney_cortex_medulla(CNR=100)
    #
    # Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:
    #
    model = dc.KidneyCortMed(
        aif = aif,
        dt = time[1],
        agent = 'gadoterate',
        TR = 0.005,
        FA = 20,
        TC = 0.2,
        n0 = 10,
    )
    #
    # Train the model on the ROI data and predict signals and concentrations:
    #
    model.train(time, roi)
    #
    # Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:
    #
    model.plot(time, roi, ref=gt, show=False)


if __name__ == "__main__":

    test_mods_tissue()
    test_mods_tissue_pixel()
    test_mods_tissue_pixel_doc()
    # test_mods_tissue_pixel_plot()
    # test_mods_aorta()
    # test_mods_aorta_liver()
    # test_mods_aorta_liver2scan()
    # test_mods_liver()
    # test_mods_kidney()
    # test_mods_kidney_cort_med()

    print('All mods tests passed!!')