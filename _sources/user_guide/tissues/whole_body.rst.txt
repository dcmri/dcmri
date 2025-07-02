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

Assume a known injection :math:`J_{v,0}(t)` into the venous system. With a standard bolus 
injection this can be approximated by a step function between the start and 
end of the injection period (see `dcmri.ca_injection`). The first-pass  :math:`J_{a,1}(t)` 
through the aorta can be computed by propagating the input through the 
heart-lung system. Using the :ref:`propagator for a chain <define-chain>`:

.. math::

    J_{a,1} = h(T_{hl}, D_{hl}) * J_{v,0}       

The flow through the aorta then splits up into 3 parts: a fraction :math:`F_l` enters 
the liver, a fraction :math:`F_k` enters the kidneys and a fraction :math:`F_o` enters the 
other organs. These flow fractions must add up to 1, so one is 
dependent on the other two: 

.. math::
    
    F_o = 1 - F_k - F_l

Each of the three components then pass through their respective organ systems, 
to produce the first pass outflux :math:`J_{k,1}, J_{l,1}, J_{o,1}` out of 
kidney, liver and other organs, respectively. The passage can be computed 
by convolution with the appropriate propagator. For the kidnyes this is the 
:ref:`compartment propagator <define-compartment>`:

.. math::

    J_{k,1} = h(T_k) * F_kJ_{a,1}  
    
For the liver it is the :ref:`plug-flow compartment <define-pfcomp>`:

.. math::

    J_{l,1} = h(T_{gl}, D_{gl}) * F_lJ_{a,1}  
    
and for the other organs this is the 
:ref:`2-compartment exchange propagator <define-2comp>`:

.. math::

    J_{o,1} = h(T_b, T_e, E_o) * F_oJ_{a,1}  

These three now join again and add up to produce the first pass in the venous 
system. At this stage we also correct for the extraction fraction :math:`E_{kl}` by the 
combined effect of liver and kidneys:

.. math:: 

    J_{v,1} = (1-E_{kl})(J_{k,1} + J_{l,1} + J_{o,1})

We can now repeat these steps to produce the second pass :math:`J_{v,2}`:

.. math::

    J_{a,2} &= h(T_{hl}, D_{hl}) * J_{v,1}  \\
    \\
    J_{k,2} &= h(T_k) * F_kJ_{a,2} \\
    J_{l,2} &= h(T_{gl}, D_{gl}) * F_lJ_{a,2} \\
    J_{o,2} &= h(T_b, T_e, E_o) * F_oJ_{a,2}  \\
    \\
    J_{v,2} &= (1-E_{kl})(J_{k,2} + J_{l,2} + J_{o,2}) \\


These steps are iterated to produce the third pass :math:`J_{a,3}`, fourth 
pass :math:`J_{a,4}` and so on. The total flux through the aorta can be 
determined by adding up the contributions of each pass:

.. math::

    J_a(t) = \sum_{i=1}^\infty J_{a,i}(t)

At each iteration the residual dose in the system reduces because of the 
extraction by liver and kidneys, and because dispersion gradually shifts the 
bolus beyond the acquisition window. 
At some point the total dose becomes too small to have any measureable impact 
on the result and we can stop iterating. 

In practice we compute the 
residual dose :math:`r_i` at each iteration (here :math:`t_\max` is the total 
acquisition time):

.. math::

    r_i = \frac{\int_0^{t_\max} J_{v,i}(t)dt}{\int_0^{t_\max} J_{v,0}(t)dt} 

And then truncate the sum as soon as the residual dose drops below a user-
defined tolerance.

The analysis shows that this models the aorta flux in terms of 11 independent 
parameters: 5 mean transit times :math:`\{T_{hl}, T_k, T_{gl}, T_p, T_e\}`, 2
dispersions :math:`\{D_{hl}, D_{gl}\}` and 4 independent fractions 
:math:`\{F_k, F_l, E_o, E_{kl}\}`. The result can be derived in one step by 
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

Other models can be derived with the same function, for instance a simplified 
model where all organs including kidney and liver are modelled by a single 
two-compartment exchange model:

.. code-block:: python

    Ja = flux_aorta(Jv_0, t, 
        E = Ekl, 
        heartlung = ['chain', (Thl, Dhl)], 
        organs = ['2cxm', ([Tp, Te], Eo)], 
        tol = 1e-3, 
    )