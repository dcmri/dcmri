import dcmri as dc
#
# Use `fake_kidney` to generate synthetic test data:
#
time, aif, roi, gt = dc.fake_kidney(CNR=100)
#
# Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:
#
model = dc.KidneyCortMed(
    aif = aif,
    dt = time[1],
    agent = 'gadoterate',
    TR = 0.005,
    FA = 15,
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
model.plot(time, roi, ref=gt)
