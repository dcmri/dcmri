.. _contributor-guide:

##########
Contribute
##########


The ``dcmri`` source code is freely accessible on 
`github <https://github.com/dcmri/dcmri>`_. We welcome contributions of any 
kind, large or small, including new features, bug fixes or improvements to 
documentation. 

If you have a suggestion for something that could be improved, but are 
not able to contribute yourself, please post an
`issue <https://github.com/dcmri/dcmri/issues>`_ describing your idea, problem 
or request, and perhaps someone else will pick it up.


**************************
How to make a contribution
**************************

You can make a contribution by submitting a 
`pull request <https://github.com/dcmri/dcmri/pulls>`_. 
The methodology for contributing to open source projects on github 
is well explained in many other resources, so we will not go through it here 
and assume you have some basic familiarity with these ideas. There is some 
guidance in the 
`github documentation <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_
for instance.

If you want to make a larger contribution, it is best to open an 
`issue <https://github.com/dcmri/dcmri/issues>`_ 
in the first instance and include some detail on what you would like to 
contribute, and why, so we can make a plan before diving in. This will 
avoid code conflicts and duplicate work. 


*********************
Contributing examples
*********************

By far the easiest way to get started with a new software package 
is to look for an example that does something similar to what you have in mind, 
and then build on that. 

As `dcmri` is a new package, we are at this stage particularly interested in 
contributions of new :ref:`examples <examples>` to help others get started 
and act as independent tests of the package. If you are using ``dcmri`` to 
analyse your data, develop or compare some 
methods please consider packaging this up as an example. 

Writing up your result as a ``dcmri`` example is of great value to the wider 
community but will also increase the visibility and impact of your work, 
increase the confidence in your findings by allowing independent replication, 
and improve the value of your work by allowing secondary usage and further 
development by others. 

Types of examples
-----------------

Examples can be 
different things but we generally consider the following types:

- **Tutorial**: an educational notebook on an aspect of DCMRI that is 
  explained using methods in the package.
- **Methodological study**: this can be an aspect of method development or 
  evaluation, such as sensitivity analysis, model comparisons etc.
- **Use case**: this is an application of `dcmri` in a real world problem, 
  typically an analysis of real data with sufficient power to show an effect 
  (or absence thereof) on meaningful endpoints - such as effect of treatment, 
  differences between groups, changes over time or predictions of outcomes. 

Methological studies or use cases can also be replications of published work, 
or serve as supplementary material with published papers or abstracts. 

How to write an example
-----------------------

The source code for all existing examples can be found in the folder 
dcmri/docs/examples. Each example is a single python script, saved in a file 
that must start with *plot_*. To write a new example, simply drop a script in 
the same folder. The next time the documentation is built, your example will 
appear under the `examples tab <https://dcmri.org/generated/examples>`_ on 
the website, in a notebook format. 

In order to test what your use case will look like before committing it, 
you can build the documentation locally:

1. In your terminal, cd to the folder *dcmri/doc*
2. Then type the command *./make html*
3. When the execution finishes, double-click on the file *index.html* in the 
   folder *dcmri/docs/build/html/*

In your python script, you must follow proper formatting of strings and 
titles to make sure your example shows up properly on the website in notebook 
format. Easiest is to look at some other examples in the source 
code, and copy the formatting.

How to include data?
--------------------

Examples often use real data as input, and these have to be made accessible to
package users via the `~dcmri.fetch` function. 

For this purpose, the data must be saved as a pickle 
file in the datafiles folder (dcmri/src/dcmri/datafiles). If they are named
'mydata.pkl', then they will be available to users by calling 
`dcmri.fetch(mydata)`. 

The data themselves must be a list of dictionaries, 
one dictionary for each scan. The items in the dictionary are the data and 
any relevant 
metadata such as sequence parameters. You must also add an entry in the 
dostring of the `dcmri.fetch()` function with some background on the dataset, 
where it comes from and what the dictionary items represent. 
  
Note while currently data are distributed as part of the package's source 
code, these will move to a dedicated ` <https://osf.io/>`_ repository in 
future versions.

If you are planning to include data with your example, please consider 
that ``dcmri``, including any data that come with the package, is distributed 
under a permissive `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ 
license. Please check that the conditions of the license are 
compatible with any requirements of the funders, data owners or sponsors of 
your study. 


