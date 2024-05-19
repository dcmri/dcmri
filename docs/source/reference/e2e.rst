.. _end-to-end models:

***************
Model catalogue
***************

An overview of all end-to-end models built into `dcmri`. 
End-to-end models relate tissue parameters directly to measured data 
and have in-built functionality to fit the models to the data.

.. currentmodule:: dcmri


Tissue models - measured input
==============================

Models for generic 2-site exchange tissues where the AIF is measured and available as signal-time curves.


Fast water exchange
-------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   OneCompFXSS
   PatlakFXSS
   ToftsFXSS
   EToftsFXSS
   TwoCompUptSS
   TwoCompExchFXSS


No water exchange
-----------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   OneCompNXSS
   PatlakNXSS
   ToftsNXSS
   EToftsNXSS
   TwoCompUptNXSS
   TwoCompExchNXSS


Any water exchange
------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   UptSS
   OneCompSS
   PatlakSS
   ToftsSS
   EToftsSS
   TwoCompUptSS
   TwoCompExchSS


Tissue models - known input
===========================

Models for generic 2-site exchange tissues where the concentrations at the inlet are known, such as when population-average input concentrations are used, or when the concentration at the inlet has been derived from data by a separate procedure.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   EToftsFXSSC
   EToftsNXSSC
   EToftsSSC


Aorta models
============

Whole-body models for arterial signals.

Steady-state sequence
---------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   AortaSignal6
   AortaSignal8
   AortaSignal8b


Saturation-recovery sequence
----------------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   AortaSignal8c


Two-scan acquisition
--------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   AortaSignal10


Kidney models
=============

Models for kidney signals.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   KidneySignal6
   KidneySignal9
   KidneyCMSignal9


Liver models
============

Models for liver signals.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   LiverSignal5
   LiverSignal9


