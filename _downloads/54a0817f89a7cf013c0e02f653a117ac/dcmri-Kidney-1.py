import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data:
#
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
params['kinetics'] = '2CF'
model = dc.Kidney(**params).train(time, roi)
model.plot(time, roi, ref=gt)
