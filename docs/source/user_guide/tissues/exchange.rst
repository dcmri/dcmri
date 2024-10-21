.. _two-site-exchange:

Exchange tissues
----------------

The *two-site exchange tissue*, or *exchange tissue* for short, is the most 
common tissue type used in applications such as brain, cancer, prostate, 
muscle, and more. It models tissues that are defined by a vascular bed which 
receives blood flow and exchanges indicator with an extravascular space.

Data measured on exchange tissues can be analysed most conveniently using the 
user interface in `dcmri.Tissue`. Developers can access the core 
functionality through the functions listed in :ref:`ref-exchange-tissues`.

Configurations
^^^^^^^^^^^^^^

The configuration of an exchange tissue is fully defined by: 
a *tracer-kinetic model* (see section :ref:`tissue-indicator-kinetics`), 
which describes the transport of indicator through the tissue; 
and a *water-exchange model* (see section :ref:`tissue-water-exchange`), 
which describes the transport of water and magnetization. Any tracer-kinetic 
model can be combined with any water exchange model to build a complete 
tissue model. 

The parameters that characterise an exchange tissue depend on the 
configuration. Table :ref:`two-site-tissue-params` lists all relevant 
parameters, and a list of parameters for each configuration 
can be found through the function `dcmri.params_tissue` or in its 
documentation. The most commonly used tissue types have fast water exchange 
across all barriers, in which case the parameters 
are those listed in table :ref:`two-site-exchange-kinetics`.

.. _two-site-tissue-params:
.. list-table:: **Tissue parameters**
    :widths: 15 25 40 20
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
    * - Fb
      - Blood flow
      - Flow of blood into the vascular space of a unit tissue
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


.. _tissue-indicator-kinetics:

Indicator kinetics
^^^^^^^^^^^^^^^^^^ 

The most general kinetic model implemented in `dcmri.Tissue` is the 
2-compartment exchange model (2CX). It assumes that the indicator distributes 
over two compartments: the plasma *p* and the interstitium *i*, with volume 
fractions *vp* and *vi*, respectively. The plasma volume is related to the 
blood volume *vb* by the hematocrit *H*; plasma and interstitium 
combined form the extracellular space *e* with volume fraction *ve*: 

.. math::
    v_p = (1-H)v_b \qquad \textrm{and} \qquad v_e = v_p + v_i

*Fp* is the flow of plasma into *p* via the arterial inlet, which equals the 
flow of plasma out of the venous outlet. It is related to the blood flow *Fb*
via the hematocrit H:

.. math::
    F_p = (1-H)F_b

The transport of indicator across the endothelium (the barrier 
separating *p* and *i*) is quantified by 
the indicator's permeability-surface area product *PS*. The most general 2CX 
model assumes that the transport of indicator across the endothelium is 
the same in each direction, i.e. the PS from interstitium to plasma (*PSi*) 
is the same as the PS from plasma to interstitium. 
Any leakage from the interstitium via lymphatic flow or otherwise is ignored. 

The indicator extraction fraction *E* and the volume transfer constant 
*Ktrans* measure the uptake of indicator into the interstitium. *E* is the 
fraction of the indicator that enters the interstitium at least once in a 
transit through the tissue. *Ktrans* is the rate at which indicator is 
delivered to the interstitium. *E* and *Ktrans* are related to the other 
parameters:

.. math::
    E=\frac{PS}{PS+F_p} \qquad \textrm{and} \qquad K^{\mathrm{trans}}=EF_p

The other kinetic models available through `dcmri.Tissue` are all special 
cases of 2CX. An overview can be found in table 
:ref:`two-site-exchange-kinetics`. 
The *two-compartment uptake model* (2CU) applies when the 
acquisition time is short so that the return of indicator to the 
vasculature is not detectable. The *high-flow model* (HF) applies when the 
temporal resolution of the measurement is too low to see any dispersion in the 
vasculature, in which case the blood flow is above the detection limit. The 
*high-flow uptake model* (HFU) combines the assumptions of the above. The 
*fast-exchange model* (FX) assumes that the transport across the endothelium is 
so rapid that plasma and interstitium are effectively well-mixed. The 
*no-exchange model* (NX) applies when the opposite is the case, when no 
measureable amounts of indicator leak out of the vasculature. The *uptake 
model* (U) applies when data are truncated to the first seconds after 
indicator arrival when any effect of venous outflow (*Fv*) is not detectable. 
The *weakly vascularised* model (WV) applies when the indicator in the 
vasculature is below the detection limit.


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
      - H, vb, vi, Fb, PS
      - :math:`PS_i = PS`
    * - 2CU
      - Two-compartment uptake
      - H, vb, Fb, PS
      - :math:`PS_i = 0`
    * - HF
      - High-flow, *AKA* extended Tofts model, extended Patlak model, 
        general kinetic model.
      - H, vb, vi, PS
      - :math:`F_b = \infty`
    * - HFU
      - High flow uptake, *AKA* Patlak model
      - H, vb, PS
      - :math:`F_b = \infty`, :math:`PSi = 0`
    * - FX
      - Fast indicator exchange
      - H, ve, Fb
      - :math:`PS = \infty`  
    * - NX
      - No indicator exchange
      - vb, Fb
      - :math:`PS = 0`      
    * - U
      - Uptake
      - Fb
      - :math:`F_v = 0`    
    * - WV
      - Weakly vascularized, *AKA* Tofts model.
      - H, vi, Ktrans
      - :math:`v_b = 0`


.. _tissue-water-exchange:

Water exchange
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
cells is measured by the parameter *vc*, and it is assumed that cells, blood 
and interstitium compse the entire tissue:

.. math::
    v_b + v_i + v_c = 1 

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





