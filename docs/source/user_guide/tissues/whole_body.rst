.. _whole-body-tissues:

Whole body
----------

Whole body pharmacokinetic models can be built by assembling suitable building 
blocks (see section :ref:`basics-pharmacokinetics`) into a closed model 
of the circulation. We will illustrate the approach by using a whole-body 
model to predict the indicator flux in the aorta 
(see :ref:`diagram <whole-body-image>` below). 

.. _whole-body-image:

.. image:: whole_body.png
  :width: 600

A model of this type is available through the function `dcmri.flux_aorta`, 
which has some flexibility to use different models for the organs 
in the model. For the purpose of this section we will model the organs as 
follows:

- Heart-lung system: chain (`dcmri.flux_chain`).
- Organs: two-compartment exchange model (`dcmri.flux_2cxm`).
- Kidneys: one-compartment (`dcmri.flux_comp`).
- Liver-gut system: plug-flow compartment (`dcmri.flux_pfcomp`).

Note that, since our aim here is to model the flux in the aorta, we only 
need to model the extracellular spaces of kidney and liver as filtered 
indicator is not returned to the blood stream.

Assume a known injection *Jv_0(t)* into the venous system. With a standard bolus 
injection this can be approximated by a step function between the start and 
end of the injection period (see `dcmri.ca_injection`). The first-pass *Ja_1* 
through the aorta can be computed by propagating the input through the 
heart-lung system:

.. code-block:: python

    Ja_1 = flux_chain(Jv_0, Thl, Dhl, t)        # First-pass through the aorta

The flow through the aorta then splits up into 3 parts: a fraction *Fl* enters 
the liver, a fraction *Fk* enters the kidneys and a fraction *Fo* enters the 
other organs. Note that these flow fractions must add up to 1, so one is 
dependent on the other two: :math:`F_o = 1 - F_k - F_l`. 

Each of the three components then pass through their respective organ systems:

.. code-block:: python

    Jk_1 = flux_comp(Fk*Ja_1, Tk, t)            # First-pass kidney outflux
    Jl_1 = flux_pfcomp(Fl*Ja_1, Tgl, Dgl, t)    # First-pass liver outflux
    Jo_1 = flux_2cxm(Fo*Ja_1, [Tb, Te], Eo, t)  # First-pass organs outflux

These three now join again and add up to produce the first pass in the venous 
system. At this stage we also correct for the extraction fraction *Ekl* by the 
combined effect of liver and kidneys:

.. code-block:: python

    Jv_1 = (1-Ekl)*(Jk_1 + Jl_1 + Jo_1)         # First pass through the veins

We can now iterate these steps to produce the second pass:

.. code-block:: python

    Ja_2 = flux_chain(Jv_1, Thl, Dhl, t)        # Second pass through the aorta
    Jk_2 = flux_comp(Fk*Ja_2, Tk, t)            # Second pass kidney outflux
    Jl_2 = flux_pfcomp(Fl*Ja_2, Tgl, Dgl, t)    # Second pass liver outflux
    Jo_2 = flux_2cxm(Fo*Ja_2, [Tp, Te], Eo, t)  # Second pass organs outflux
    Jv_2 = (1-Ekl)*(Jk_2 + Jl_2 + Jo_2)         # Second pass through the veins

These steps are iterated to produce the third pass *Ja_3*, fourth pass *Ja_4* 
and so on. 
At each iteration the residual dose in the system reduces because of the 
extraction by liver and kidneys, and because dispersion gradually shifts the 
bolus beyond the acquisition window. 
At some point the total dose becomes too small to have any measureable impact 
on the result and we can stop iterating. For instance at the 4th pass:

.. code-block:: python

    injected_dose = numpy.trapezoid(Jv_0, t)
    residual_dose = numpy.trapezoid(Ja_4, t)
    stop = residual_dose < 1e-3 * injected_dose
    # exit the iteration if stop is True

After the last iteration, the total flux through the aorta can be 
determined by adding up the contributions of each pass. For instance, if the 
iteration has stopped after the 4th pass, the result would be:

.. code-block:: python

    Ja = Ja_1 + Ja_2 + Ja_3 + Ja_4

The analysis shows that this models the aorta flux in terms of 11 independent 
parameters: 5 mean transit times (*Thl*, *Tk*, *Tgl*, *Tp*, *Te*), 2
dispersions (*Dhl*, *Dgl*) and 4 independent fractions 
(*Fk*, *Fl*, *Eo*, *Ekl*). The result can be derived in one step by 
using the function `dcmri.flux_aorta` with the following arguments:

.. code-block:: python

    Ja = flux_aorta(Jv_0, t, 
        E = Ekl, 
        FFkl = Fk + Fl, 
        FFk = Fk / (Fk + Fl), 
        heartlung = ['chain', (Thl, Dhl)], 
        organs = ['2cxm', ([Tp, Te], Eo)], 
        kidneys = ['comp', (Tk,)], 
        liver = ['pfcomp', (Tgl, Dgl)], 
        tol = 1e-3, 
    )