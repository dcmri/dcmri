import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_liver` to generate synthetic test data:
#
time, aif, vif, roi, gt = dc.fake_liver()
#
# Build a tissue model and set the constants to match the experimental
# conditions of the synthetic test data. Note the default model is the
# dual-inlet model for extracellular agents (2I-EC). Since the
# synthetic data are generated with an intracellular agent, the default
# for the kinetic model needs to be overwritten:
#
model = dc.Liver(
    kinetics = '2I-IC',
    aif = aif,
    vif = vif,
    dt = time[1],
    agent = 'gadoxetate',
    field_strength = 3.0,
    TR = 0.005,
    FA = 15,
    n0 = 10,
    R10 = 1/dc.T1(3.0,'liver'),
    R10a = 1/dc.T1(3.0, 'blood'),
    R10v = 1/dc.T1(3.0, 'blood'),
)
#
# Train the model on the ROI data:
#
model.train(time, roi)
#
# Plot the reconstructed signals (left) and concentrations (right) and
# compare the concentrations against the noise-free ground truth. Since
# the data are analysed with an exact model, and there are no other data
# errors present, this should fior the data exactly.
#
model.plot(time, roi, ref=gt)
