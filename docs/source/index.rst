#######################
**dcmri** documentation
#######################

A python toolbox for dynamic contrast MRI

.. note::

   ``dcmri`` is under construction. At this stage, the API may still change and 
   features may be deprecated without warning.


*******
Mission
*******

``dcmri`` aims to simplify the analysis of dynamic-contrast MRI (DC-MRI) data, 
and to facilitate the development and distribution of new methods. 


********
Features
********

- A :ref:`tissue bank <end-to-end models>` with an intuitive user interface 
  to analyse data from different tissues.
- Building blocks to simplify the creation and testing of new methods:

  - :ref:`signal models <signal mods>` for common MRI sequences;
  - basic :ref:`pharmacokinetic building blocks <PK blocks>` such as 
    multi-compartment models; and
  - functions to generate signals for specific 
    :ref:`tissue types <tissue-types>`.

- A library of :ref:`real data <real-data>`,  
  :ref:`synthetic data <synthetic-data>` and 
  :ref:`synthetic images <synthetic-images>` to simplify testing or comparison 
  of methods. 

- A library of basic utilities such as common 
  :ref:`input functions <input-functions>` and 
  :ref:`constants <useful-constants>`, and functions for performing 
  :ref:`convolutions <convolution-functions>` and 
  :ref:`data sampling <sampling-functions>`. 


***************
Getting started
***************

``dcmri`` offers methods for a range of application areas including  
vascular-interstitial tissues (brain, cancer, prostate, muscle, ...) but also 
atypical tissues such as liver and kidney, and whole-body 
models to analyse vascular signals.

To get started, have look at the :ref:`user guide <user-guide>`, 
look for an example in the list of :ref:`tutorials <tutorials>` 
and :ref:`use cases <use-cases>`, or find a suitable method in the 
:ref:`reference guide <reference-guide>` and dive straight in.

******
Citing
******

When you use ``dcmri``, please cite: 

Ebony Gunwhy, Eve Shalom and Steven Sourbron. dcmri: an open-source python 
package for dynamic contrast MRI. European Society of Magnetic Resonance in 
Medicine and Biology (Barcelona, Spain), pp 491, Oct 2024.

************
Contributing
************

``dcmri`` is open to any type of contributions, but at this stage we are 
particularly interested in contributions of :ref:`use cases <use-cases>`. 
Please see the :ref:`contributor guide <contributor-guide>` for more detail.

*******
License
*******

``dcmri`` is distributed under the 
`Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ license - a 
permissive, free license that allows users to use, modify, and 
distribute the software without restrictions.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   user_guide/index
   reference/index
   generated/examples/index
   contribute/index
   releases/index
   about/index

