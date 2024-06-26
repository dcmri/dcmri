import numpy as np
import dcmri as dc


def test_mods_tissue():
    time, aif, roi, gt = dc.fake_tissue(CNR=50)
    model = dc.Tissue(
        aif = aif,
        dt = time[1],
        agent = 'gadodiamide',
        TR = 0.005,
        FA = 20,
        n0 = 15,
    )
    model.train(time, roi)
    model.plot(time, roi, testdata=gt, show=False)


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
    model.plot(time, roi, testdata=gt, show=False)

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
    model.plot(time, roi, testdata=gt, show=False)
    #
    # Repeat the fit using a free nephron model:
    #
    params['kinetics'] = 'FN'
    model = dc.Kidney(**params).train(time, roi)
    model.plot(time, roi, testdata=gt, show=False)

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
    model.plot(time, roi, testdata=gt, show=False)


if __name__ == "__main__":

    test_mods_tissue()
    test_mods_aorta()
    test_mods_aorta_liver()
    test_mods_aorta_liver2scan()
    test_mods_liver()
    test_mods_kidney()
    test_mods_kidney_cort_med()

    print('All mods tests passed!!')