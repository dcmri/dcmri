********************
The .dmr data format
********************

The .dmr (dynamic mr) data format is a simple csv-based format for storing
dynamic region-of-interest based data. It is a zipped folder with some 
or all of the following csv files:

- rois.csv: ROI based time curves for one or more subjects and studies
- pars.csv: Parameters such as sequence parameters or subject properties
- sdev.csv: Standard deviations for the parameters in pars.csv
- data.csv: Data dictionary for all parameters in the above files 
  with full description, units, data type and other characteristics.

A single .dmr folder can contain data for one or more subject, and 
for one or more studies per subject.

Why another data format?
------------------------

Region-of-interest based data in dynamic MRI are typically saved and 
shared in spreadsheets - mostly excel. The type of data is always very 
similar (time curves and parameters) but they can be encoded 
in an infinite variety of ways (rows, columns, multiple or single 
tabs, multiple or single files, etc.). 

When sharing data between 
collaborators, even within the same team, significant amount of time 
is therefore spent reformatting excel sheets from one collaborator 
to a format suitable as input for the pipelines written by another. 
This costs substantial amounts of time and effort - especially because 
the differences in data formats also propagate into different 
code interfaces, which also need to be harmonized.

This variability and effort spent refactoring data and code is wholly 
unnecessary as the different ways of encoding these data in excel are 
equivalent - there is seldom a good reason for the differences. 
Significant time and effort can therefore be saved at no cost at all 
by deciding on a single standardised way of encoding ROI-based 
data in spreadsheet form, and writing an API for easy reading and 
writing in python.

Why a csv format?
-------------------

A csv-based format was chosen because csv is simple and 
non-proprietary format that can be read and written with built-in 
python packages. 

csv data can also be read with excel or 
other common programs such as google sheets, and they can also be 
written in the same way. This is important because not everyone codes 
and ROI-based data are often inspected and further analysed with 
graphical user interfaces. 

What is the .dmr python API?
----------------------------

The `dcmri` package includes functions `dcmri.read_dmr` and `dcmri.write_dmr` 
for easily creating and reading dmr files programmatically. These 
can also be used to check existing data for compliance with the 
.dmr standard. 

`dcmri.read_dmr` will throw an error with an informative error 
message when reading a file that is not properly formatted. Equivalently, 
`dcmri.write_dmr` will raise an informative exception when attempting to write 
an improperly formatted dictionary ot csv. 

`dcmri.read_dmr` also produces 
resulst in a form that can easily be recast as a pandas DataFrame for 
further processing.

Where can I find examples of .dmr data?
---------------------------------------

The `dcmri.fetch` function retrieves data in .dmr format from online 
repositories. See the `examples <https://dcmri.org/generated/examples/index.html>`_ 
for notebooks where these files are read and the data are processed.

Details of the .dmr format
--------------------------

A .drm file is simply a zipped folder with one or more csv files. 
data.csv is a required file but rois.csv, pars.csv and sdev.csv are 
optional and may or may not be present.

rois.csv
^^^^^^^^

Signal-time curves for regions-of-interest can be found in the 
rois.csv file of the dmr folder. The top row lists the subjects, 
the second row lists the studies, and the third row names the data 
series. Then below this header information the data are listed as 
columns. They do not need to have the same length, and not all subjects 
need to have the same studies and series.

Consider the example of two subjects S1, S2 with two studies 
each (visit V1 and V2), and two ROI curves per study (tumor-T, artery-A). 
For the sake of illustration we wil assume each curve has 3 time points. 
The file rois.csv would contain the following table (ROI values randomly chosen):

.. list-table:: rois.csv example
    :widths: 5 5 5 5 5 5 5 5
    :header-rows: 3

    * - S1
      - S1
      - S1
      - S1
      - S2
      - S2
      - S2
      - S2
    * - V1
      - V1
      - V2
      - V2
      - V1
      - V1
      - V2
      - V2
    * - T
      - A 
      - T 
      - A 
      - T
      - A
      - T
      - A 
    * - 120
      - 320
      - 226
      - 158
      - 95
      - 185
      - 417
      - 260
    * - 160
      - 301
      - 255
      - 198
      - 83
      - 120
      - 450
      - 213
    * - 178
      - 398
      - 201
      - 123
      - 65
      - 122
      - 496
      - 201

pars.csv
^^^^^^^^

The pars.csv file contains any parameters that are provided with the ROI data. 
The parameter values are listed in long format with columns subject, 
study, parameter name, and parameter value.

Consider the previous example of two subjects and two visits. If 
sequence parameters TR and FA are provided for each subject and each 
visit, the pars.csv file looks like this:

.. list-table:: pars.csv example 
    :widths: 5 5 5 5
    :header-rows: 1

    * - subject
      - study
      - parameter
      - value
    * - S1
      - V1
      - TR
      - 5
    * - S1
      - V1
      - FA
      - 15
    * - S1
      - V2
      - TR
      - 5.5
    * - S1
      - V2
      - FA
      - 12
    * - S2
      - V1
      - TR
      - 6
    * - S2
      - V1
      - FA
      - 14
    * - S2
      - V2
      - TR
      - 4
    * - S2
      - V2
      - FA
      - 16


sdev.csv
^^^^^^^^

The sdev.csv file is similar to pars.csv, but it provides standard 
deviations of parameters rather than their value. These are not 
required and should only be provided when known. 

Consider the previous example of two subjects and two visits. Assume 
uncertainties in the flip angles due to B1-effects are known. Then this 
can be encoded in the sdev.csv file:

.. list-table:: sdev.csv example
    :widths: 5 5 5 5
    :header-rows: 1

    * - subject
      - study
      - parameter
      - value
    * - S1
      - V1
      - FA
      - 2
    * - S1
      - V2
      - FA
      - 1.5
    * - S2
      - V1
      - FA
      - 1.8
    * - S2
      - V2
      - FA
      - 1.6


data.csv
^^^^^^^^

data.csv is the only required file in a dmr folder. It is a data 
dictionary that provides detail on the parameters listed in 
rois.csv, pars.csv and sdev.csv. 

data.csv must list for each parameter a more explicit *description*, 
a *unit* and a *type*. Possible types are *str*, *float*, *int*, 
*bool*, and *complex*. 

Other properties can be added to the data.csv dictionary - for instance 
keywords that group parameters, codes such as osipi or DICOM codes, etc. 

Each series and parameter in rois.csv and pars.csv must be listed in 
the data.csv dictionary. So for the example above a minimal data.csv 
file would look like this:


.. list-table:: data.csv example
    :widths: 5 15 5 5
    :header-rows: 1

    * - parameter
      - description
      - unit
      - type
    * - T
      - Signal in a tumour region of interest
      - arbitrary units
      - float
    * - A
      - Signal in an arterial region of interest
      - arbitrary units
      - float
    * - TR
      - Repetition time
      - msec
      - float
    * - FA
      - Flip angle
      - deg
      - float
