.. _end-to-end models:

***************
Model catalogue
***************

An overview of all end-to-end models built into `dcmri`. 
End-to-end models relate tissue parameters directly to measured data 
and have in-built functionality to fit the models to the data.

.. currentmodule:: dcmri


Tissue models
=============

Models for generic 2-site exchange tissues where the AIF is measured and available as signal-time curves.

Fast/no water exchange
----------------------

Models that assume either very fast water exchange (``water_exchange=True``) or none at all (``water_exchange=False``).

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   OneCompSS
   PatlakSS
   ToftsSS
   EToftsSS
   TwoCompUptSS
   TwoCompExchSS


Any water exchange
------------------

Tissue models that allow for any intermediate level of water exchange.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   UptSS
   OneCompWXSS
   PatlakWXSS
   ToftsWXSS
   EToftsWXSS
   TwoCompUptWXSS
   TwoCompExchWXSS


Organ models
============

Models for specific tissue types

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   Aorta
   Liver
   AortaLiver
   Kidney2CFXSR
   Kidney2CFXSS
   KidneyPFFXSS
   KidneyCortMedSR


Special models
==============

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   AortaLiver2scan
   


