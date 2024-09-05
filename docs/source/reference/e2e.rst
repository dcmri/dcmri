.. _end-to-end models:

***************
Model catalogue
***************

An overview of all end-to-end models built into `dcmri`. 

End-to-end models relate tissue parameters directly to measured data 
and have built-in functionality to determine parameter values from data. 

The models are organised based on the type of data they can predict.

.. currentmodule:: dcmri


One region of interest
======================

Theae models assume the data represent a single 1D time curve - typically the signal for a pixel, or an average over a region of interest. Some, but not all, of these models require a separately measured input, or have to assume a standardised input.


.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   Tissue
   Liver
   Kidney
   Aorta


Pixel-based models
==================

These models assume the data are arrays with multiple signals from the same subject. Typically these will be taken from different pixels, or all pixels, in the same volume, image or region.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   TissueArray


Multiple regions of interest
============================

These models assume the data are measured over two or more tissue regions, typically whole organs, or different substructures of the same organ. Some, but not all, of these models require a separately measured input, or have to assume a standardised input.

.. autosummary::
   :toctree: ../generated/api/
   :template: custom-class-template.rst
   :recursive:

   KidneyCortMed
   AortaKidneys
   AortaLiver
   AortaLiver2scan