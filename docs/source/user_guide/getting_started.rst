***************
Getting started
***************


Installation
------------

``dcmri`` can be installed using pip:

.. code-block:: console

   pip install dcmri


Basic usage
-----------

Let's import the package and generate 
some synthetic region-of-interest (ROI) data using `dcmri.fake_tissue`. 
We want these to look realistic, so we are adding noise with a 
contrast-to-noise ratio (CNR) of 50:

.. code-block:: python

    import dcmri as dc
    
    time, aif, roi, _ = dc.fake_tissue(CNR=50)

Here *time* is an array of time points, *aif* is a signal-time curve measured 
in a feeding artery at those times, and *roi* is a signal-time curve measured 
in a region of interest. 

Next we find a suitable tissue type from the 
:ref:`tissue bank <end-to-end models>` and initialize it. For most common 
applications, this will be `dcmri.Tissue`:

.. code-block:: python

    tissue = dc.Tissue(aif=aif, t=time)

At this point this is a generic tissue with a default configuration. 
The next step is to train the tissue using the ROI data:

.. code-block:: python  

    tissue.train(time, roi)

And that's it. We can now display the measured tissue parameters:

.. code-block:: python

    tissue.print_params(round_to=2)

.. code-block:: console

    --------------------------------
    Free parameters with their stdev
    --------------------------------

    Blood volume (vb): 0.017 (0.002) mL/cm3
    Interstitial volume (vi): 0.161 (0.004) mL/cm3
    Permeability-surface area product (PS): 0.002 (0.0) mL/sec/cm3

    ----------------------------
    Fixed and derived parameters
    ----------------------------

    Plasma volume (vp): 0.009 mL/cm3
    Interstitial mean transit time (Ti): 82.839 sec

The standard deviations of the free parameters are orders of magnitude 
smaller than the value itself, which offers confidence that the tissue 
properties are well determined by the data. We should also verify that the 
trained tissue does indeed predict the data correctly:

.. code-block:: python

    tissue.plot(time, roi)

.. image:: tissue.png
  :width: 600

The signal plot on the left shows that the model correctly predicts the 
measured data, except for the noise. The plot on the right shows that 
the reconstructed concentrations in blood and tissue show the expected 
profiles and that values are in an expected range for a standard contrast 
agent injection (0-5mM in blood).

