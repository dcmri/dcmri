.. _liver-tissues:

Liver
-----

Pharmacokinetic models in DCMRI assume that the liver consists of at most 
three spaces, as shown in the :ref:`diagram <liver-image>` below.  
A plasma compartment recieves inflow of plasma from the hepatic artery (*Fa*) 
and the portal vein (*Fv*), which collects venous blood that has passed 
through the gut. 
The plasma exchanges indicator with the liver's interstitium with a 
bi-directional exchange rate (*PS*). The interstitium exchanges with the 
hepatocytes which evacuate indicator to the gall bladder. Any indicator 
that is not evacuated in this way will be cleared from the plasma compartment
through the venous outlet. 

.. _liver-image:

.. figure:: liver.png
  :width: 600

  A diagram of the indicator transport into and out of the liver compartments, 
  showing their inputs and outputs. The free model parameters are shown in red.


All models currently implemented in ``dcmri`` assume that the hepatic artery 
can be modelled as a :ref:`plug flow system <define-plug-flow>` with mean 
transit time *Ta*. They also assume that the walls of 
the liver capillaries are highly permeable to contrast agent so that plasma and 
interstitium are well-mixed (:math:`PS\to\infty`). Future versions will 
generalize to include models with finite *PS*. 


Extracellular agents
^^^^^^^^^^^^^^^^^^^^

Dual inlet
++++++++++

If the contrast agent is extracellular, then by definition :math:`k_{he}=0` 
and the hepatocyte compartment plays no role in the analysis. Since we assume 
fast exchange of indicator across the capillary wall, the liver can then 
be modelled as a single extracellular 
:ref:`compartment <define-compartment>` with volume fraction *ve*. A
physical interpretation for the rate constant can be derived by expressing 
mass conservation for the compartment in terms of the extracellular 
concentration *c_e*. Since the indicator is evacuated by the total plasma flow 
we have:

.. math::
    :label: liver-extracellular-mc

    v_e\frac{dc_e}{dt}(t) = J_\mathrm{in}(t) - (F_a+F_v)c_e(t)

Writing this in terms of the tissue concentration :math:`C_e=v_ec_e` we get an 
equation as in section :ref:`define-compartment` with mean transit time:

.. math::
    :label: liver-extracellular-mtt

    T_e = \frac{v_e}{F_a+F_v}

If a reliable concentration can be measured in the aorta (*ca*) and in the 
portal vein (*cv*), then:

.. math::
    :label: liver-influx

    J_\mathrm{in}(t) = F_ac_a(t-T_a) + F_vc_v(t)

And the solution is:

.. math::
    :label: liver-extracellular

    C_e(t) = e^{-t/T_e}*\left(F_ac_a(t-T_a) + F_vc_v(t)\right)
 
In this case the model is fully defined by the four parameters *ve*, *Fa*, 
*Fv* and *Ta*. An alternative parametrization uses the total plasma flow *Fp* 
and the arterial flow fraction *fa*:

.. math::
    :label: arterial-flow-fraction

    F_p = F_a+F_v
    \qquad\textrm{and}\qquad
    f_a = \frac{F_a}{F_a+F_v}

Eq. :eq:`liver-extracellular` is the *dual-inlet model for 
extracellular tracer*, or **2I-EC**.

If the plasma flow is very high then the bolus dispersion in the liver is not 
separately measureable. The extracellular space is then a simple 
:ref:`pass <define-pass>`:

.. math::
    :label: liver-extracellular-hf

    C_e(t) = v_e\left(f_ac_a(t-T_a) + (1-f_a)c_v(t)\right)
    
This is the *dual-inlet extracellular high-flow* model, or **2I-EC-HF**.


Single inlet
++++++++++++

If the acquisition is not optimized for data collection in the portal vein, 
then a portal-venous concentration may not be available. In theory this can 
be addressed by modelling the passage through the gut. If we model the gut as 
a compartment then Eq. :eq:`liver-extracellular` becomes:

.. math::
    :label: liver-extracellular-1i

    C_e(t) = e^{-t/T_e}*\left(F_ac_a(t-T_a) 
           + F_v\;\frac{e^{-t/T_g}}{T_g}*c_a(t)\right)

This is a single-inlet extracellular model (**1I-EC**). If the dispersion in 
liver and gut cannot be separated, an alternative approach is to 
simplify the model by considering the extracellular space of liver and gut 
as a single combined space, and model it for instance as a 
:ref:`plug-flow compartment<define-pfcomp>`: with mean transit time *Te* and 
dispersion *De*. Using the propagator :math:`h` of a plug-flow 
compartment the solution is a 3-parameter model:

.. math::
    :label: liver-extracellular-disp

    C_e = v_e\; h(T_e, D_e) * c_a

This is the single-inlet extracellular dispersion model 
(**1I-EC-D**). Out of the 3 parameters, only the extracellular volume 
:math:`v_e` is a liver characteristic. 
The other two (:math:`T_e` and :math:`D_e`) are determined by 
the properties of the gut and the liver. In particular the blood flow into the 
liver is not measureable under these conditions. 


Intracellular agents
^^^^^^^^^^^^^^^^^^^^

For contrast agents that enter the hepatocytes, the models must be extended 
with a hepatocyte compartment. In the case where backflux from hepatocytes is 
negligible (:math:`k_{hb}=0`) this is a straightforward extension of the 
models for extracellular agents. The hepatocytes are modelled as a 
compartment (see :ref:`define-compartment`) and an interpretation of the rate 
constants can be found from the conservation of indicator mass in terms of 
the concentration :math:`c_h`:

.. math::
    :label: liver-hepatocytes-mc

    v_h\frac{dc_h}{dt} = k_{he}c_e - k_{bh}c_h

Expressing this in terms of the tissue concentration :math:`C_h` we find the 
mean transit time of the hepatocytes:

.. math::
    :label: liver-hepatocytes-mtt

    T_h = \frac{v_h}{k_{bh}}

Since there is no backflux into the extracellular space, the solution for 
:math:`c_e` can be used as an input function to the hepatocellular 
compartment:

.. math::
    :label: liver-hepatocytes

    C_h(t) = e^{-t/T_h}*k_{he}c_e(t)

When acquisition times are short, the excretion from the hepatocytes is 
negligible and the hepatocytes are modelled as a trap with uptake only:

.. math::
    :label: liver-hepatocytes-uptake

    C_h(t) = \int_0^{\infty} k_{he}c_e(t)

The total tissue concentration is :math:`C=C_e+C_h`. The intracellular models 
**2I-IC**, **2I-IC-HF**, **2I-IC-U**, **1I-IC-D** and **1I-IC-DU** are direct 
extensions of the extracellular models, with the additional parameters 
:math:`k_{he}` and :math:`T_h` (see table :ref:`table-liver-models`). 

Passage through the hepatocytes is a slow process, especially when the 
excretion rate is substantially impaired by disease or drugs. Measuring 
excretion rates reliably under such conditions is challenging and requires 
very long acquisition times (hours). Under those conditions the actual state of 
the liver may change during the acquisition, requiring a non-stationary model 
for the hepatocyte compartment. In the simplest scenario this can be modelled 
by interpolating linearly between *initial* values for the parameters 
:math:`k_{he, i}` and :math:`T_{h, i}` and *final* values 
:math:`k_{he, f}` and :math:`T_{h, f}`.

Definitions and notations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _table-liver-params:
.. list-table:: **Tissue parameters**
    :widths: 15 25 40 20
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - Ta
      - Arterial mean transit time
      - Time for blood to travel through the hepatic artery
      - sec
    * - fa
      - Arterial flow fraction
      - Arterial fraction of the total blood flow into the liver
      - None
    * - Fp
      - Liver plasma flow
      - Total flow of plasma into the liver tissue, per unit tissue volume.
      - mL/sec/cm3
    * - ve
      - liver extracellular volume fraction
      - Part of the liver tissue taken up by the extracellular space
      - mL/cm3
    * - De 
      - Extracellular dispersion
      - Bolus broadening in the extracellular space of the liver
      - None
    * - Te
      - Extracellular mean transit time
      - Average time to for an indicator molecule to travel through the liver 
        extracellular space
      - sec
    * - khe
      - Intracellular uptake rate
      - volume of extracellular fluid fully cleared of indicator per unit 
        time and tissue
      - mL/sec/cm3
    * - kbh
      - Biliary excretion rate
      - volume of intracellular fluid fully cleared of indicator per unit 
        time and tissue, by transport to bile
      - mL/sec/cm3
    * - Th
      - Hepatocellular mean transit time
      - Average time for an indicator molecule to travel through the 
        hepatocytes
      - sec.


.. _table-liver-models:
.. list-table:: **Kinetic models for the liver**
    :widths: 20 40 20 20
    :header-rows: 1

    * - Short name
      - Full name
      - Parameters
      - Solution
    * - **Dual-inlet extracellular**
      - 
      -
      - 
    * - 2I-EC
      - Dual-inlet extracellular
      - ve, Fp, fa, Ta
      - Eq. :eq:`liver-extracellular`
    * - 2I-EC-HF
      - Dual-inlet extracellular high-flow
      - ve, fa, Ta
      - Eq. :eq:`liver-extracellular-hf`
    * - **Single-inlet extracellular**
      - 
      -
      - 
    * - 1I-EC
      - Single-inlet extracellular
      - ve, Fp, fa, Ta, Tg
      - Eq. :eq:`liver-extracellular-1i`
    * - 1I-EC-D
      - Single-inlet extracellular dispersion
      - ve, Te, De
      - Eq. :eq:`liver-extracellular-disp`
    * - **Dual-inlet intracellular**
      - 
      - 
      - 
    * - 2I-IC
      - Dual-inlet intracellular
      - ve, Fp, fa, Ta, khe, Th
      - Eqs. :eq:`liver-extracellular` and :eq:`liver-hepatocytes`
    * - 2I-IC-HF
      - Dual-inlet intracellular high-flow
      - ve, fa, Ta, khe, Th
      - Eqs. :eq:`liver-extracellular-hf` and :eq:`liver-hepatocytes`
    * - 2I-IC-U
      - Dual-inlet intracellular uptake
      - ve, Fp, fa, Ta, Th
      - Eqs. :eq:`liver-extracellular` and :eq:`liver-hepatocytes-uptake`
    * - **Single-inlet intracellular**
      - 
      - 
      - 
    * - 1I-IC-D
      - Single-inlet intracellular dispersion
      - ve, Te, De, khe, Th
      - Eqs. :eq:`liver-extracellular-disp` and :eq:`liver-hepatocytes`
    * - 1I-IC-DU
      - Single-inlet intracellular dispersion uptake
      - ve, Te, De, khe
      - Eqs. :eq:`liver-extracellular-disp` and :eq:`liver-hepatocytes-uptake`



