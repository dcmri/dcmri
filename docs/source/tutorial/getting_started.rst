***************
Getting started
***************

For standard analysis tasks, ``dcmri`` includes a collection of :ref:`end-to-end models <end-to-end models>` that provide an easy way to fit data. 

To illustrate how these work, we generate some synthetic data using one of the built-in functions `dcmri.synth_1`. The following code shows how to fit this to a standard extended Tofts model, and print out the results.

.. exec_code::

    import dcmri as dc

    # Generate synthetic data: 
    #   time: an array of time points
    #   aif: MRI signals vs time measured in a feeding artery
    #   roi: MRI signals vs time measured in a region of interest.
    #   gt: a dictionary with ground-truth values for reference
    time, aif, roi, gt = dc.synth_1(CNR=50)

    # Build a tissue model and set all the constants to the correct value:
    model = dc.TissueSignal3c(aif,
        dt = time[1],
        Hct = 0.45, 
        agent = 'gadodiamide',
        field_strength = 3.0,
        TR = 0.005,
        FA = 20,
        R10 = 1.0,
        R10a = 1/dc.T1(3.0,'blood'),
        t0 = 15,
    )

    # Train the model on the ROI data:
    model.train(time, roi)

    # Display the optimized model parameters:
    model.print(round_to=2)

To visualise this in more detail we can plot the fits against data:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
    #
    ax0.set_title('Prediction of the MRI signals.')
    ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
    ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('MRI signal (a.u.)')
    ax0.legend()
    #
    ax1.set_title('Reconstruction of concentrations.')
    ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
    ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
    ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
    ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Concentration (mM)')
    ax1.legend()
    #
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import dcmri as dc

    time, aif, roi, gt = dc.synth_1(CNR=50)

    model = dc.TissueSignal3c(aif,
        dt = time[1],
        Hct = 0.45, 
        agent = 'gadodiamide',
        field_strength = 3.0,
        TR = 0.005,
        FA = 20,
        R10 = 1.0,
        R10a = 1/dc.T1(3.0,'blood'),
        t0 = 15,
    )
    model.train(time, roi)
    model.print(round_to=2)

    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
    #
    ax0.set_title('Prediction of the MRI signals.')
    ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
    ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('MRI signal (a.u.)')
    ax0.legend()
    #
    ax1.set_title('Reconstruction of concentrations.')
    ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
    ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
    ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
    ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Concentration (mM)')
    ax1.legend()
    #
    plt.show()


