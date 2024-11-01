****************
How to use dcmri
****************

Analyzing data
--------------

To analyse data with ``dcmri``, select the appropriate tissue type from the 
:ref:`tissue bank <end-to-end models>` and train it on your data. For examples 
of usage, consult the documentation of the model or look for a 
similar application in the list of :ref:`tutorials <tutorials>` or 
:ref:`use cases <use-cases>`. 

The models in the tissue bank are high-level implementations that can be 
customized to run a wide range of different models, parameter settings or 
methods. If no configuration options are specified by the user, 
they will always run the most conventional models. For more background on 
what models are available, or what their parameters are, have a look 
at the relevant :ref:`background section <tissue-types>`. 

Developing models
-----------------

Apart from the end-to-end models in the :ref:`tissue bank <end-to-end models>`, 
``dcmri`` also includes a library of more generic basic methods that can 
be used to build custom-models more easily, and facilitate the creation of new 
models to extend the functionality of ``dcmri``. These basic methods are 
implemented as simple python functions for maximal transparency and modularity. 

The are organised in a hierarchical fashion:

- The :ref:`tissue module <tissue mods>` contains high-level implementations 
  of tissue-specific functions such as concentrations or signals. For more 
  background, see the section on :ref:`tissue types <tissue-types>`.

- The :ref:`signal module <signal mods>` provides generic signal models for 
  MRI sequences. For more background on these models, see the section on 
  :ref:`imaging sequences <imaging-sequences>`.

- The :ref:`pharmacokinetic building blocks <PK blocks>` implement generic 
  pharmacokinetic models that can be assembled to build more complex models. 
  For more background on these models, see the section on 
  :ref:`basic pharmacokinetics <basics-pharmacokinetics>`.

- A library of :ref:`utilities <utilities>` that can be used to build new 
  functions, test them or demonstrate their usage. This includes:
  
  - A library of **real data** taken from published studies. These are all 
    available through the `dcmri.fetch` function (see also 
    section :ref:`use cases <use-cases>`). 
  - A library of functions to generate **synthetic data** for testing or 
    demonstration of models.
  - **Synthetic images** the can be used to build digital reference objects. 
  - A library of published **input functions**.
  - A library of useful **constants** taken from literature, such as standard 
    dosages, concentrations and relaxivities of common contrast agents, and 
    common MRI parameters and perfusion parameters for different field strengths 
    or tissue types.
  - A collection of basic functions for performing **convolutions**, or 
    **sampling data**.