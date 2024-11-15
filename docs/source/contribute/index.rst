.. _contributor-guide:

##########
Contribute
##########

.. note::

   dcmri is under construction. At this stage, the API may still change and 
   features may be deprecated without warning.

The ``dcmri`` source code is freely accessible on 
`github <https://github.com/dcmri/dcmri>`_ and contributions can be made via 
`pull-request <https://github.com/dcmri/dcmri/pulls>`_. 

If you would like to make a contribution, please open an 
`issue <https://github.com/dcmri/dcmri/issues>`_ 
in the first instance and include some detail on what you would like to 
contribute, and why. ``dcmri`` is open to contributions of any kind, 
including new features and enhancements to documentation. 

*********
Use cases
*********

Since the only real test for software is its applicability in real-world 
applications, we are at this stage particularly interested in contributions of 
new :ref:`examples <examples>`. If you are using ``dcmri`` to analyse 
your data, please consider packaging this up as a use case. 

As the examples show, we define a use-case loosely as a 
replication of published work with real 
data and sufficient power to show an effect (or absence thereof) on meaningful 
endpoints - such as effect of treatment, differences between groups, changes 
over time or predictions of outcomes. 

Every use case consists of two elements: 

- a self-contained python script 
  that generates your main results, including any statistics and figures; 
- the data used as input to this script accessible via the `~dcmri.fetch` 
  function. Note while currently data are distributed as part of the package's 
  source code, these will move to a dedicated `OSF <https://osf.io/>`_ 
  respository in future versions.

Writing up your result as a ``dcmri`` use case is of great value to the wider 
community but will also increase the visibility and impact of your work, 
increase the confidence in your findings by allowing independent replication, 
and improve the value of your work by allowing secondary usage and further 
development by others. 

If you are planning to submit a use case, please consider that ``dcmri``, 
including any data that come with the package, is distributed under a 
permissive `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ license. 
Please check that the conditions of the license are 
compatible with any requirements of the funders, data owners or sponsors of 
your study. 


