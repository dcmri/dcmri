"""
========================================
An exploration of water exchange effects
========================================

This example will explore the effect of water exchange in a standard two-compartmental tissue with extended Tofts kinetics. Data acquisition is modelled using the steady-state model of a spoiled gradient echo sequence.

The three tissue compartments involved are the blood, interstitium and tissue cells. The effect of water exchange between blood and interstitium (**transendothelial** water exchange) and between intersitium and tissue cells (**transcytolemmal** water exchange) is separately investigated. The water exchange in the blood compartment, particularly between plasma and red blood cells, is assumed to be in the fast exhange limit throughout. 
"""

# %%
# Simulation setup
# ----------------
# First we set up the simulation by importing the necessary packages and defining the constants that will be fixed throughout. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc

# The constants defining the signal model and simulation settings
const = {
    'TR': 0.005,                # Repetition time (sec)
    'FA': 15.0,                 # Flip angle (deg)
    'agent': 'gadoxetate',      # Contrast agent
    'field_strength': 3.0,      # Magnetic field strength (T)
    'R10': 1/dc.T1(3.0, 'muscle'),
    'R10b': 1/dc.T1(3.0, 'blood'), 
}

tacq, aif, _, _ = dc.fake_tissue(tacq=300, **const)

# The ground-truth kinetic parameters of the extended Tofts model
ptruth = {
    'S0':1,        # S0 (a.u.)
    'vp':0.05,       # vp (mL/mL)
    'PS':0.1/60,     # Ktrans (mL/sec/mL)
    've':0.30,        # ve (mL/mL)
} 
truth = np.array(list(ptruth.values()))

# %% 
# Visualising water exchange effects
# ----------------------------------
# We'll start by exploring how the level of water exchange affects the measured signal. As a point of reference we will predict signals using explicit models in the limits of fast water exchange and no water exchange:

# %% 

# Signal in the fast water exchange limit (all barriers fully transparent to water)
ffx = dc.Tissue(aif=aif, t=tacq, **(const | ptruth)).predict(tacq)

# Signal in the no water exchange limit (all barriers impermeable to water)
nnx = dc.Tissue(aif=aif, t=tacq, water_exchange='none', **(const | ptruth)).predict(tacq)

# %% 
# In order to simulate intermediate regimes, we need the more general model (AWX) that allows us to vary the values of the water permeabilities ``PSe`` and ``PSc`` across the endothelium and the membrane of the tissue cells, respectively. 
#
# In the first instance we consider a (hypothetical) tissue without transendothelial water exchange, but fast transcytolemmal water exchange. In other words, the endothelium is impermeable to water (``PSe = 0``) and the cell membrane is fully transparent. The symbolic value ``PSc = np.inf`` is not allowed but we can set ``PSc`` to the very high value of 1000 mL water filtered per second by 1mL of tissue. This is indistinguishable from the fast water exchange limit ``PSc = np.inf`` (as could be verified by increasing the value even higher):

# %%

# Signal without transendothelial water exchange, but fast transcytolemmal water exchange
nfx = dc.Tissue(aif=aif, t=tacq, water_exchange='any', PSe=0, PSc=1000, **(const | ptruth)).predict(tacq) 

# %% 
# Next we consider the alternative scenario where the endothelium is transparent to water (``PSe = np.inf``, approximated as ``PSe = 1000``) and the cell membrane is impermeable (``PSc = 0``):

# %%

# Signal with fast transendothelial water exchange, but without transcytolemmal water exchange
PSe, PSc = 1000, 0
fnx = dc.Tissue(aif=aif, t=tacq, water_exchange='any', PSe=1000, PSc=0, **(const | ptruth)).predict(tacq)

# %%
# An intermediate situation arises if neither of the water permeabilities is either very high or close to zero. Trial and error shows that a choice of ``PSe = 1`` mL/sec/mL and ``PSc = 2`` mL/sec/mL produces a curve that lies in between the extremes:

# %%

# Signal with intermediate transendothelial and transcytolemmal water exchange
iix = dc.Tissue(aif=aif, t=tacq, water_exchange='any', PSe=1, PSc=2, **(const | ptruth)).predict(tacq)

# %%
# We now plot the different results, using fast- and no-exchange limits for visual reference:

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(15,5))

ax0.set_title('No transendothelial exchange \n Fast transcytolemmal exchange')
ax0.plot(tacq, ffx, 'g-', label='Fast exchange')
ax0.plot(tacq, nnx, 'b-', label='No exchange')
ax0.plot(tacq, nfx, 'r--', label='Intermediate exchange')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()

ax1.set_title('Fast transendothelial exchange \n No transcytolemmal exchange')
ax1.plot(tacq, ffx, 'g-', label='Fast exchange')
ax1.plot(tacq, nnx, 'b-', label='No exchange')
ax1.plot(tacq, fnx, 'r--', label='Intermediate exchange')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Signal (a.u.)')
ax1.legend()

ax2.set_title('Intermediate transendothelial exchange \n Intermediate transcytolemmal exchange')
ax2.plot(tacq, ffx, 'g-', label='Fast exchange')
ax2.plot(tacq, nnx, 'b-', label='No exchange')
ax2.plot(tacq, iix, 'r--', label='Intermediate exchange')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Signal (a.u.)')
ax2.legend()

plt.show()

# %%
# These figures show the expected observations: 
#
# 1. Water exchange levels have a measureable effect on signals, as shown by the clear difference between fast and slow exchange scenarios (blue vs. green curves). 
#
# 2. With fast transcytolemmal exchange but impermeable endothelium (left panel), the slowy changing extravascular part of the signal aligns with the fast exchange curve, and the first pass aligns with the no-exchange curve  
#
# 3. Without transcytolemmal exchange the extravascular curve lines up with the no-exchange model (middle panel). The first pass is closer to the fast-exchange signal but does not align with it completely as it is partly obscured by already extravasated indicator.
#
# 4. When both exchange levels are intermediate (right panel), then the signal is also intermediate between the extremes of fast and no exchange.
#
# **Note** while the effect of water exchange is detectable, it is comparatively small considering the difference between the blue and green curves represent the extremes of zero to maximal levels of water exchange. It is easily verified that changing kinetic parameters such as Ktrans over their entire range (zero to infinity) has a much larger impact on the signal. Water exchange is in that sense a second order effect.


# %% 
# Understanding water exchange bias
# ---------------------------------
# Since the level of water exchange affects the signal, making inaccurate assumptions on the level of water exchange will create a bias in any measurement of the kinetic parameters. 
#
# One way to explore the scale of the water exchange bias is by generating data for a tissue in the fast exchange limit and analysing them making the opposite assumption that water exchange is negligible:

# Launch a no-exchange model with default settings for the free parameters
model = dc.Tissue(aif=aif, t=tacq, water_exchange='none', **const)

# Predict the signal using the untrained model as a reference
nnx0 = model.predict(tacq)

# Train the model using data for a fast-exchange tissue
model.train(tacq, ffx)

# Predict the signal using the trained model
nnx1 = model.predict(tacq)

# Calculate the bias in the fitted parameters in %
pars = model.get_params('S0','vp','PS','ve')
bias = 100*(np.array(pars)-truth)/truth

# Plot the model fits
fig, ax0 = plt.subplots(1,1,figsize=(6,5))
ax0.set_title('Water exchange bias')
ax0.plot(tacq, ffx, 'g-', linewidth=3, label='Signal data (fast exchange tissue)')
ax0.plot(tacq, nnx0, 'b-', label='Prediction (before training)')
ax0.plot(tacq, nnx1, 'b--', label='Prediction (after training)')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()
plt.show()

# Print the parameter bias
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],1), '%')
print('ve error:', round(bias[3],1), '%')
print('Ktrans error:', round(bias[2],1), '%')

# %%
# The plot shows that the trained model predicts the data with high accuracy, despite the inaccurate assumption of no water exchange. However the false assumption does lead to fitted parameters that are severely biased.

# %% 
# Removing water exchange bias
# ----------------------------
# The model bias can be removed by generalizing the model to allow for any level of water exchange, avoiding the risk of making a false assumption on this point:

# Launch a general water exchange model with default settings for all free parameters
model = dc.Tissue(aif=aif, t=tacq, water_exchange='any', **const)

# Predict the signal using the untrained model as a reference
iix0 = model.predict(tacq)

# Train the model using fast-exchange data and predict the signal again.
# Note: we reduce here the x-tolerance from its default (1e-08) to speed up convergence. 
iix1 = model.train(tacq, ffx, xtol=1e-2).predict(tacq)

# Calculate the bias in the fitted parameters
pars = model.get_params('S0','vp','PS','ve')
bias = 100*(np.array(pars)-truth)/truth

# Plot the model fits
fig, ax0 = plt.subplots(1,1,figsize=(6,5))
ax0.set_title('Water exchange bias')
ax0.plot(tacq, ffx, 'g-', linewidth=3, label='Signal data (fast exchange tissue)')
ax0.plot(tacq, iix0, 'r-', label='Prediction (before training)')
ax0.plot(tacq, iix1, 'r--', label='Prediction (after training)')
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Signal (a.u.)')
ax0.legend()
plt.show()

# Print the parameter bias
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],2), '%')
print('ve error:', round(bias[3],2), '%')
print('Ktrans error:', round(bias[2],2), '%')

# Print the water permeability estimates
print('')
print('Water permeability estimates')
print('----------------------------')
print('PSe:', round(model.PSe,0), 'mL/sec/mL')
print('PSc:', round(model.PSc,0), 'mL/sec/mL')


# %%
# Plotting the results now shows a practically perfect fit to the data, and the measurements of the kinetic parameters are effectively unbiased. 
# 
# As a bonus the water-exchange sensitive model also estimates the water permeability, which as expected produces values in the fast-exchange range. As the actual PS-values are infinite the estimates can never approximate the ground truth, but at this level the predicted data are effectively indistinguishable from fast-exchange signals. 

# %% 
# Additional sources of bias
# --------------------------
# The results show that small residual errors remain in the kinetic parameters, even after removing the model bias. While the error may be negligible for practical purposes, it is useful and illustrative to explore its origin further.
# 
# Any remaining bias must be due to one or more of the three remaining sources of error: (1) *sampling bias* - temporal undersampling in the data used for training, which at 1.5s creates a small mismatch with the exact (pseudo)continuous signals; (2) *convergence bias* - imperfect convergence of the model training; (3) *numerical bias* - numerical errors in the computation of the model solutions. 
#
# We can get some insight by fitting the data with an unbiased model, i.e. fitting the data with the same model that was used to generate it. This is a simple model that is likely to be much less susceptible to convergence or numerical bias, so this analysis exposes the sampling bias (alternatively we can generate data with much smaller temporal sampling intervals):

# Train a fast-exchange model on the fast exchange data
model = dc.Tissue(aif=aif, t=tacq, **const).train(tacq, ffx)

# Calculate the bias relative to the ground truth
pars = model.get_params('S0','vp','PS','ve')
bias = 100*(np.array(pars)-truth)/truth

# Print the bias for each kinetic parameter
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],2), '%')
print('ve error:', round(bias[3],2), '%')
print('Ktrans error:', round(bias[2],2), '%')

# %%
# Any remaining bias is smaller than 0.01%, which shows that temporal undersampling in this case only causes a minor error, and the residual errors observed with the more general model are due to imperfect convergence or numerical error. We can test for convergence bias by retraining the model with tighter convergence criteria: 

# Train a general water exchange model to fast exchange data:
model = dc.Tissue(aif=aif, t=tacq, water_exchange='any', **const).train(tacq, ffx, xtol=1e-9)

# Calculate the bias in the fitted parameters
pars = model.get_params('S0','vp','PS','ve')
bias = 100*(np.array(pars)-truth)/truth

# Print the parameter bias
print('')
print('Bias in kinetic model parameters')
print('--------------------------------')
print('vp error:', round(bias[1],2), '%')
print('ve error:', round(bias[3],2), '%')
print('Ktrans error:', round(bias[2],2), '%')

# Print the water permeability estimates
print('')
print('Water permeability estimates')
print('----------------------------')
print('PSe:', round(model.PSe,0), 'mL/sec/mL')
print('PSc:', round(model.PSc,0), 'mL/sec/mL')

# %%
# The result is almost exactly the same as before, which indicates that the model has indeed converged and the residual bias is likely due to numerical error. This is plausible, since the general water exchange model is implemented using linear algebra involving operations such as matrix exponentials and numerical matrix inversion, which are likely to come with some numerical error. The exercise here verifies that the impact of these errors on the measurements of the kinetic parameters is negligible - as it should be.  

# %% 
# Bias versus precision
# ---------------------
# 