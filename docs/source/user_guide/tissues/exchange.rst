.. _two-site-exchange:

Exchange tissues
----------------

This is the most common tissue type relevant for the majority of applications 
currently in DC-MRI, including brain, cancer, prostate, muscle, and more. 
Most common approaches to modelling the signal from two-site exchange tissues 
are available through `dcmri.Tissue`.  

Indicator kinetics
^^^^^^^^^^^^^^^^^^

The most general kinetic model implemented in `dcmri.Tissue` is the 2-compartment exchange model (2CX). It assumes that the indicator distributes over two compartments: the plasma *p* and the interstitium *i*, with volume fractions *vp* and *vi*, respectively. Together they form the extracellular space *e* with volume fraction *ve*: 

.. math::
    v_e = v_p + v_i

The rate of exchange of indicator across the endothelium (the barrier 
separating *p* and *i*) is the same in either direction and is quantified by 
the indicator's permeability-surface area product *PS*. *Fp* is the flow of 
plasma into *p* via the arterial inlet, which equals the flow of plasma out 
of the venous outlet. Any leakage from the interstitium via lymphatic flow or 
otherwise is ignored. 

The indicator extraction fraction *E* and the volume transfer constant 
*Ktrans* measure the uptake of indicator into the interstitium. *E* is the 
fraction of the indicator that enters the interstitium at least once in a 
transit through the tissue. *Ktrans* is the rate at which indicator is 
delivered to the interstitium. *E* and *Ktrans* are related to the other 
parameters:

.. math::
    E=\frac{PS}{PS+F_p} \qquad \textrm{and} \qquad K^{\mathrm{trans}}=EF_p

The other kinetic models available through `dcmri.Tissue` are all special 
cases of 2CX. For more details see table :ref:`two-site-exchange-kinetics`. 
`dcmri` uses a naming convention that references the assumptions made, but 
the table also lists common alternative names.



Water kinetics
^^^^^^^^^^^^^^

For two-compartment exchange tissues, the three tissue compartments exchanging 
water are the blood, interstitium and tissue cells. Water exchange refers to 
the transport of water across the barriers between them: *transendothelial* 
water exchange between blood and interstitium, and *transcytolemmal* water 
exchange between interstitium and tissue cells. The water exchange in the 
blood compartment between plasma and red blood cells is assumed to be in the 
fast exchange limit throughout.

Since water occupies intracellular spaces, water exchange models introduce a 
dependence on the intracellular volumes. The volume fraction of the tissue 
cells is measured by the parameter *vc*, and the volume of the blood 
compartment is *vb*. The volume fraction of red blood cells in blood is the 
tissue hematocrit *Hct*. The following constraints are always valid:

.. math::
    v_b + v_i + v_c = 1 \qquad\textrm{and}\qquad v_p = (1-Hct)v_b

The rate of water exchange across a barrier is quantified by the 
permeability-surface area (PS) of water, a quantity in units of mL/sec/cm3. 
*PSe* is the transendothelial water exchange rate and *PSc* is the 
transcytolemmal water rate.

Water exchange across either of these two barriers can be in the 
fast-exchange limit (F), restricted (R), or there may be no water exchange at 
all (N). Since there are two barriers involved this leads to 3x3=9 possible 
water exchange regimes. `dcmri.Tissue` denotes these 9 regimes by a 
combination of the letters F, R and N: the first letter refers to the water 
exchange across the endothelium, and the second to the water exchange across 
the cell wall. Examples of possible water exchange regimes are:

- *RF*: Restricted water exchange across the endothelium 
  (:math:`0\lt PS_e\lt\infty`) and fast water exchange across the tissue cell 
  wall (:math:`PS_c=\infty`). 
- *NF*: No water exchange across the endothelium (:math:`PS_e=0`) and fast 
  water exchange across the tissue cell wall (:math:`PS_c=\infty`). 
- *FR*: Fast water exchange across the endothelium (:math:`PS_e=\infty`) and 
  restricted water exchange across the tissue cell wall 
  (:math:`0\lt PS_c\lt\infty`).


Tissue models
^^^^^^^^^^^^^

Any kinetic model can be combined with any water exchange model to build a 
complete tissue model. In the fast water exchange limit (FF) the entire tissue 
acts as a single well-mixed water compartment and the parameters are the 
kinetic parameters listed in table :ref:`two-site-exchange-kinetics`. For 
other combinations, the free parameters are listed in section 
:ref:`two-site-exchange-tables`, but they can also be printed using 
`dcmri.Tissue` directly. For instance, to find the free parameters for a 
high-flow tissue with restricted transcytolemmal water exchange:

.. code-block:: python

   # Build a tissue with the required configuration
   tissue = dcmri.Tissue('HF', 'RF')
   
   # Print the free parameters
   print(tissue.params())

This will print all tissue parameters with their initial values:

.. code-block:: console

   {'PSe': 0.03, 'Hct': 0.45, 'vp': 0.055, 'vi': 0.5, 'Ktrans': 0.003}


.. _two-site-exchange-tables:

Reference tables
^^^^^^^^^^^^^^^^

Table :ref:`two-site-tissue-params` provides a complete list of possible 
tissue parameters including symbols, full name, and units. 

Table :ref:`two-site-exchange-kinetics` lists all kinetic models for exchange 
tissues, along with alternative names and their free parameters. 

Table :ref:`kinetic-regimes` list the water compartments and free parameters 
for all tissues with water exchange regimes FF, FR, RF, and RR. Regimes 
without water exchange across one or both of the barriers are not listed 
explicitly (FN, NF, FR, NR and NN). They differ from restricted water 
exchange only in that they fix the respective water permeabilities 
(*PSe* or *PSc*) to zero. 

.. _two-site-tissue-params:
.. list-table:: **Tissue parameters**
    :widths: 15 25 40 20
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - Fp
      - Plasma flow
      - Flow of plasma into the vascular space of a unit tissue
      - mL/sec/cm3
    * - Ktrans
      - Volume transfer constant
      - Volume of arterial plasma cleared of indicator per unit time and per 
        unit tissue
      - mL/sec/cm3
    * - E 
      - Extraction fraction
      - The fraction of entering particles that will pass through the 
        interstitum at least once
      - None
    * - vp  
      - Plasma volume
      - Volume fraction of the plasma space
      - mL/cm3
    * - vb
      - Blood volume fraction
      - Volume fraction of the blood space
      - mL/cm3
    * - H
      - Hematocrit
      - Volume fraction of the red blood cells in whole blood
      - None
    * - vi
      - Interstitial volume
      - Volume fraction of the interstitial space
      - mL/cm3
    * - ve
      - Extracellular volume 
      - Combined volume fraction of plasma and interstitium
      - mL/cm3
    * - vc
      - Cellular volume
      - Volume fraction of the tissue cells
      - mL/cm3
    * - PSe
      - Endothelial water permeability
      - Flow of water across the endothelium per unit tissue volume
      - mL/sec/cm3
    * - PSc
      - Cytolemmal water permeability
      - Flow of water across the cell wall per unit tissue volume
      - mL/sec/cm3


.. _two-site-exchange-kinetics:
.. list-table:: **Kinetic models**
    :widths: 10 40 20 20
    :header-rows: 1

    * - Short name
      - Full name
      - Parameters
      - Assumptions
    * - 2CX
      - Two-compartment exchange
      - vp, vi, Fp, PS
      - *See text*
    * - 2CU
      - Two-compartment uptake
      - vp, Fp, PS
      - :math:`PS \approx 0`
    * - HF
      - High-flow, *AKA* extended Tofts model, extended Patlak model, 
        general kinetic model.
      - vp, vi, PS
      - :math:`F_p = \infty`
    * - HFU
      - High flow uptake, *AKA* Patlak model
      - vp, PS
      - :math:`F_p = \infty`, :math:`PS \approx 0`
    * - FX
      - Fast indicator exchange
      - ve, Fp
      - :math:`PS = \infty`  
    * - NX
      - No indicator exchange
      - vp, Fp
      - :math:`PS = 0`      
    * - U
      - Uptake
      - Fp
      - :math:`F_p \approx 0`    
    * - WV
      - Weakly vascularized, *AKA* Tofts model.
      - vi, Ktrans
      - :math:`v_p = 0`


.. _kinetic-regimes:
.. list-table:: **Parameters in each regime** 
    :widths: 15 15 30 40
    :header-rows: 1 

    * - Water exchange
      - Indicator exchange
      - Water compartments
      - Free parameters
    * - **FF**
      - 
      - 
      - 
    * - FF
      - 2CX
      - vb + vi + vc
      - vp, vi, Fp, PS
    * - FF
      - 2CU
      - vb + vi + vc
      - vp, Fp, PS
    * - FF
      - HF
      - vb + vi + vc
      - vp, vi, PS
    * - FF
      - HFU
      - vb + vi + vc
      - vp, PS
    * - FF
      - FX
      - vb + vi + vc 
      - ve, Fp  
    * - FF
      - NX
      - vb + vi + vc
      - vp, Fp  
    * - FF
      - U
      - vb + vi + vc
      - Fp 
    * - FF
      - WV
      - vb + vi + vc
      - vi, Ktrans
    * - **RR**
      - 
      - 
      - 
    * - RR
      - 2CX
      - vb, vi, vc
      - PSe, PSc, H, vb, vi, Fp, PS
    * - RR
      - 2CU
      - vb, vi, vc
      - PSe, PSc, H, vb, vi, Fp, PS
    * - RR
      - HF
      - vb, vi, vc
      - PSe, PSc, H, vb, vi, PS
    * - RR
      - HFU
      - vb, vi, vc
      - PSe, PSc, H, vb, vi, PS
    * - RR
      - FX
      - vb, vi, vc 
      - PSe, PSc, H, vb, vi, Fp  
    * - RR
      - NX
      - vb, vi, vc 
      - PSe, H, vb, vi, Fp  
    * - RR
      - U
      - vb, vi, vc 
      - PSe, vb, vi, Fp 
    * - RR
      - WV
      - vi, vi+vc
      - PSc, vi, Ktrans
    * - **RF**
      - 
      - 
      - 
    * - RF
      - 2CX
      - vb, vi+vc
      - PSe, H, vb, vi, Fp, PS
    * - RF
      - 2CU
      - vb, vi+vc
      - PSe, H, vb, Fp, PS
    * - RF
      - HF
      - vb, vi+vc
      - PSe, H, vb, vi, PS
    * - RF
      - HFU
      - vb, vi+vc
      - PSe, H, vb, PS
    * - RF
      - FX
      - vb, vi+vc
      - PSe, H, vb, vi, Fp
    * - RF
      - NX
      - vb, vi+vc
      - PSe, H, vb, Fp
    * - RF
      - U
      - vb, vi+vc
      - PSe, vb, Fp  
    * - RF
      - WV
      - vi+vc
      - vi, Ktrans
    * - **FR**
      - 
      - 
      -  
    * - FR
      - 2CX
      - vb+vi, vc
      - PSc, H, vb, vi, Fp, PS
    * - FR
      - 2CU
      - vb+vi, vc
      - PSc, vc, vp, Fp, PS
    * - FR
      - HF
      - vb+vi, vc
      - PSc, H, vb, vi, PS
    * - FR
      - HFU
      - vb+vi, vc
      - PSc, vc, vp, PS
    * - FR
      - FX
      - vb+vi, vc 
      - PSc, vc, ve, Fp
    * - FR
      - NX
      - vb+vi, vc 
      - PSc, vc, vp, Fp
    * - FR
      - U
      - vb+vi, vc 
      - PSc, vc, Fp
    * - FR
      - WV
      - vi, vc
      - PSc, vi, Ktrans

        





