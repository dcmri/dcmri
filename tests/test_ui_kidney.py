import numpy as np
import dcmri as dc


# Debugging mode
VERBOSE = 1
SHOW = True

VERBOSE = 0
SHOW = False


def test_ui_kidney():
    time, aif, roi, gt = dc.fake_tissue(R10=1/dc.T1(3.0,'kidney'))
    #
    # Override the parameter defaults to match the experimental conditions of 
    # the synthetic test data:
    #
    params = {
        'aif':aif,
        'dt':time[1],
        'agent': 'gadodiamide',
        'TR': 0.005,
        'FA': 15,
        'TS': time[1],
        'R10': 1/dc.T1(3.0,'kidney'),
        't0': 10*time[1],
    }
    #
    # Train a two-compartment filtration model on the ROI data and plot the fit:
    #
    model = dc.Kidney(kinetics='2CF', **params)
    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)
    assert model.cost(time, roi) < 0.5
    assert 80 < model.params('Tt', round_to=0) < 90


def test_ui_kidney_cort_med():

    time, aif, roi, gt = dc.fake_kidney()

    model = dc.KidneyCortMed(
        aif = aif,
        dt = time[1],
        agent = 'gadoterate',
        TR = 0.005,
        FA = 15,
        TC = 0.2,
        TS = time[1],
        n0 = 10,
    )

    model.train(time, roi)
    model.plot(time, roi, ref=gt, show=SHOW)

    assert model.cost(time, roi) < 1
    assert model.params('Tdt', round_to=0) == 27


def test_ui_aorta_kidneys():

    datafile = dc.fetch('minipig_renal_fibrosis')
    dmr = dc.read_dmr(datafile, 'nest')
    rois, pars = dmr['rois']['Pig']['Test'], dmr['pars']['Pig']['Test']

    aorta_kidneys = dc.AortaKidneys(

        # Configuration
        sequence='SSI',
        heartlung='chain',
        organs='comp',
        agent="gadoterate",

        # General parameters
        field_strength=pars['B0'],
        t0=pars['TS']*pars['n0'], 

        # Injection protocol
        weight=pars['weight'],
        dose=pars['dose'],
        rate=pars['rate'],

        # Sequence parameters
        TR=pars['TR'],
        FA=pars['FA'],
        TS=pars['TS'],

        # Aorta parameters
        CO=60,  
        R10a=1/dc.T1(pars['B0'], 'blood'),

        # Kidney parameters
        vol_lk=85,
        vol_rk=85,
        R10_lk=1/dc.T1(pars['B0'], 'kidney'),
        R10_rk=1/dc.T1(pars['B0'], 'kidney'),
    )

    # Define time and signal data
    time = pars['TS'] * np.arange(len(rois['Aorta']))
    t = (time, time, time)
    signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])

    # Train model and show result
    aorta_kidneys.train(t, signal)
    aorta_kidneys.plot(t, signal, show=SHOW)
    aorta_kidneys.print_params(round_to=4)
    Fp_lk = aorta_kidneys.params('Fp_lk') # 0.032
    assert 0.030 < np.round(Fp_lk, 3) < 0.033



if __name__ == "__main__":

    # test_ui_kidney()
    # test_ui_kidney_cort_med()
    test_ui_aorta_kidneys()

    print('All ui_kidney tests passed!!')

