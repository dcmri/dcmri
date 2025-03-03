import dcmri as dc
#
# Get the data for the baseline visit of the first subject in the study:
#
data = dc.fetch('tristan_rifampicin')
data = data[0]
#
# Initialize the AortaLiver model with the available data:
#
model = dc.AortaLiver(
    weight = data['weight'],
    agent = data['agent'],
    dose = data['dose'][0],
    rate = data['rate'],
    field_strength = data['field_strength'],
    t0 = data['t0'],
    TR = data['TR'],
    FA = data['FA'],
    R10a = data['R10b'],
    R10l = data['R10l'],
    H = data['Hct'],
    vol = data['vol'],
)
#
# We are only fitting here the first scan data, so the xdata are the
# aorta- and liver time points of the first scan, and the ydata are
# the signals at these time points:
#
xdata = (data['time1aorta'], data['time1liver'])
ydata = (data['signal1aorta'], data['signal1liver'])
#
# Train the model using these data and plot the results to check that
# the model has fitted the data:
#
model.train(xdata, ydata, xtol=1e-3)
model.plot(xdata, ydata)
