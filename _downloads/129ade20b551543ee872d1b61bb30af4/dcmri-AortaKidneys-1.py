import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `fake_tissue` to generate synthetic test data from experimentally-derived concentrations:
#
time, aif, roi, gt = dc.fake_tissue()
xdata, ydata = (time,time,time), (aif,roi,roi)
#
# Build an aorta-kidney model and parameters to match the conditions of the fake tissue data:
#
model = dc.AortaKidneys(
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
)
#
# Train the model on the data:
#
model.train(xdata, ydata, xtol=1e-3)
#
# Plot the reconstructed signals and concentrations and compare against the experimentally derived data:
#
model.plot(xdata, ydata)
