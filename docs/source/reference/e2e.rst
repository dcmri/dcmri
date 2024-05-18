.. _end-to-end models:

*********
Model zoo
*********

An overview of all end-to-end models built into `dcmri`. 
End-to-end models relate tissue parameters directly to measured data 
and have in-built functionality to fit the models to the data.

.. currentmodule:: dcmri


Tissue models
=============

This is a collection of models for generic tissue types.

Fast water exchange
-------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   TissueSignal3
   TissueSignal4
   TissueSignal5b

No water exchange
-----------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   TissueSignal3b


Intermediate water exchange
---------------------------

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   TissueSignal5
   TissueSignal5c
   TissueSignal7


Aorta models
============

This is a collection of models for aorta data.

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

This is a collection of models for kidney data.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   KidneySignal6
   KidneySignal9
   KidneyCMSignal9


Liver models
============

This is a collection of models for liver data.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   LiverSignal5
   LiverSignal9


