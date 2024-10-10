import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data:
#
time, aif, roi, gt = dc.fake_tissue(CNR=50)
#
# Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:
#
model = dc.Tissue(
    aif = aif,
    dt = time[1],
    agent = 'gadodiamide',
    TR = 0.005,
    FA = 15,
    n0 = 15,
)
#
# Train the model on the ROI data:
#
model.train(time, roi)
#
# Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:
#
model.plot(time, roi, ref=gt)