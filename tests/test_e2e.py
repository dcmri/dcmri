
# import matplotlib.pyplot as plt
# import dcmri as dc
# import numpy as np

# #
# # Use `dro_aif_2` to generate synthetic test data from experimentally-derived concentrations:
# #
# time, aif, roi, gt = dc.make_tissue_2cm()
# #
# # Build a tissue model with the appropriate constants:
# #
# model = dc.ToftsSS(aif,
#     dt = time[1],
#     Hct = 0.45, 
#     agent = 'gadodiamide',
#     field_strength = 3.0,
#     TR = 0.005,
#     FA = 20.0,
#     R10 = 1/dc.T1(3.0,'muscle'),
#     R10b = 1/dc.T1(3.0, 'blood'),
#     t0 = 10,
# )
# #
# # Train the model on the data, and predict concentrations:
# #
# model.train(time, roi)
# #
# # Plot the reconstructed signals and concentrations and compare against the experimentally derived data:
# #
# fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
# ax0.set_title('Prediction of the MRI signals.')
# ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
# ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
# ax0.set_xlabel('Time (min)')
# ax0.set_ylabel('MRI signal (a.u.)')
# ax0.legend()
# ax1.set_title('Reconstruction of concentrations.')
# ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue data')
# ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
# ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial data')
# ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
# ax1.set_xlabel('Time (min)')
# ax1.set_ylabel('Concentration (mM)')
# ax1.legend()
# plt.show()


# model.print(round_to=3)


