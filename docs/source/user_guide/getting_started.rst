***************
Getting started
***************

``dcmri`` is designed as an intuitive interface for measuring tissue properties by from DC-MRI data. 

Basic usage
-----------

To show the basic usage of `dcmri`, let's import the package and generate some synthetic ROI data using `dcmri.fake_tissue`. We want these to look realistic, so we are adding noise with a contrast-to-noise ratio (CNR) of 50:

.. code-block:: python

    import dcmri as dc
    
    time, aif, roi, gt = dc.fake_tissue(CNR=50)

Here *time* is an array of time points, *aif* is a signal-time curve measured in a feeding artery at those times, and *roi* is a signal-time curve measured in a region of interest. *gt* is a dictionary with ground truth values that we can use as a sanity check of the approach.

Next we find a suitable tissue type from the :ref:`tissue bank <end-to-end models>` and initialize it. For most common applications, this will be `dcmri.Tissue`:

.. code-block:: python

    tissue = dc.Tissue(aif=aif, t=time)

At this point this is a generic tissue that is unlikely to predict the data well. The next step is to train the tissue using the ROI data:

.. code-block:: python  

    tissue.train(time, roi)

We can check that the tissue now correctly predicts the data using the plot function:

.. code-block:: python

    tissue.plot(time, roi)

.. image:: tissue.png
  :width: 600

Now we are satisfied that the tissue is properly trained, we can inspect its parameter values:

.. code-block:: python

    tissue.print_params(round_to=2)

.. code-block:: console

    --------------------------------
    Free parameters with their stdev
    --------------------------------

    Plasma volume (vp): 0.02 (0.002) mL/cm3
    Interstitial volume (vi): 0.243 (0.007) mL/cm3
    Volume transfer constant (Ktrans): 0.003 (0.0) mL/sec/cm3

    ------------------
    Derived parameters
    ------------------

    Extracellular volume (ve): 0.263 mL/cm3

The standard deviations of the free parameters are orders of magnitude smaller than the value itself, which adds confidence that the tissue properties are well determined. 

In this case, since the data are derived from synthetic ground truth values, we can also check that the results are close to the ground truths.