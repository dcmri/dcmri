import matplotlib.pyplot as plt
import dcmri as dc
#
# Use `make_tissue_2cm_ss` to generate synthetic test data:
#
time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=100, R10=1/dc.T1(3.0,'liver'))
#
# Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:
#
model = dc.LiverPCC(gt['cb'],
    dt = gt['t'][1],
    Hct = 0.45,
    agent = 'gadoxetate',
    field_strength = 3.0,
    TR = 0.005,
    FA = 20,
    R10 = 1/dc.T1(3.0,'liver'),
    t0 = 15,
)
#
# Train the model on the ROI data:
#
model.train(time, roi)
#
# Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:
#
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
ax0.set_title('Prediction of the MRI signals.')
ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
ax0.set_xlabel('Time (min)')
ax0.set_ylabel('MRI signal (a.u.)')
ax0.legend()
ax1.set_title('Reconstruction of concentrations.')
ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
ax1.plot(gt['t']/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Concentration (mM)')
ax1.legend()
plt.show()