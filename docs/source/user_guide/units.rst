*****
Units
*****

``dcmri`` uses a standardized system of units for all input and output 
quantities. The system is **internally consistent**: as long as data and 
parameters are entered in these units, all return values will automatically 
also be in the same units. Hence it is not needed to track units through 
intermediate computations, which greatly reduces the risk of unit conversion 
errors in complex analysis pipelines.

Table of units
--------------

The following table shows for each quantity the `dcmri` unit, and wehere 
relevant also other units that are commonly found in the literature on DC-MRI:

+-------------------------------+------------+--------------------------------------+
| Quantity                      | dcmri unit | common alternative units             |
+===============================+============+======================================+
| Time                          | sec        | min                                  |
+-------------------------------+------------+--------------------------------------+
| Length                        | cm         |                                      | 
+-------------------------------+------------+--------------------------------------+
| Area                          | cm2        |                                      | 
+-------------------------------+------------+--------------------------------------+
| Volume                        | cm3        | mL                                   | 
+-------------------------------+------------+--------------------------------------+
| Angle                         | deg        | rad                                  | 
+-------------------------------+------------+--------------------------------------+
| Fluid volume                  | mL         |                                      | 
+-------------------------------+------------+--------------------------------------+
| Weight                        | kg         | g                                    | 
+-------------------------------+------------+--------------------------------------+
| Amount of indicator           | mmol       | mol                                  | 
+-------------------------------+------------+--------------------------------------+
| Concentration                 | M          | mmol/L, mM                           | 
+-------------------------------+------------+--------------------------------------+
| Tissue concentration          | mmol/cm3   | mmol/L, mM                           |
+-------------------------------+------------+--------------------------------------+
| Indicator flux                | mmol/sec   |                                      |
+-------------------------------+------------+--------------------------------------+
| MRI relaxation time           | sec        | msec                                 | 
+-------------------------------+------------+--------------------------------------+
| MRI relaxation rate           | 1/sec      | 1/msec                               | 
+-------------------------------+------------+--------------------------------------+
| Magnetization                 | A/cm       | A/m                                  | 
+-------------------------------+------------+--------------------------------------+
| Contrast agent relaxivity     | 1/sec*M    | 1/mM/sec                             | 
+-------------------------------+------------+--------------------------------------+
| Perfusion                     | mL/sec/cm3 | mL/100mL/min, 1/sec, mL/sec/mL       | 
+-------------------------------+------------+--------------------------------------+
| Permeability-surface area     | mL/sec/cm3 | 1/sec, 1/min                         | 
+-------------------------------+------------+--------------------------------------+
| (Fluid) volume fraction       | mL/cm3     | mL/100mL, mL/100g, %, dimensionless  | 
+-------------------------------+------------+--------------------------------------+
| Fluid flow                    | mL/sec     |                                      | 
+-------------------------------+------------+--------------------------------------+

Rationale
---------

We have chosen to deviate from the system of SI units in order 
to align more closely to the natural units of most of the quantities, and 
historical conventions in the field. However we have also chosen to deviate 
from conventions in some places in order the arrive at an internally 
consistent set of units.

The use of a single systematic system of units minimises the risk of errors due 
to incorrect conversions between units in code, which are hard to track 
especially when different functions from different modules are assembled. 

It does mean that some of the quantities are not in their natural units - 
for instance tissue concentrations are typically in the order of 0.001 M and 
are therefore commonly expressed in units of mM. Equally, contrast agent 
relaxivities are conventionally expressed in units of 1/mM*sec. In `dcmri`, 
concentrations and relaxivities will nevertheless be required and returned in 
units of M and 1/M*sec, respectively. 

It is the responsability of the user to convert any output in more 
conventional units for publication or presentation purposes. Equally, it 
is the user's responsability to convert any input data into `dcmri` units 
before applying any of the functions in the package. 


Units of volume
---------------

Fluid volumes of mL and volumes of cm3 are of course numerically 
interchangeable. `dcmri` nevertheless distinguishes explicitly between 
(tissue) volumes (cm3) or voxel volumes (cm3) and fluid volumes (mL) to avoid 
confusion in the interpretation of important quantities such as perfusion or 
concentration.

In the `dcmri` system of units, perfusion is expressed in mL/sec/cm3. This 
directly reflects the physical interpretation as volume of blood (mL) 
delivered per unit of time (sec) to a unit of tissue (cm3). If tissue 
volumes were to be expressed in units of mL instead of cm3, the units of 
perfusion would be mL/sec/mL. This looks odd and is often reduced to units 
of 1/sec, cancelling out the mL's. In this form the units suggest that 
perfusion is a rate, which is physically confusing. 

The units of mL/sec/cm3 also align more directly to the historical units of 
perfusion as mL/sec/g. They reflect the original practice of first measuring 
whole-organ blood flow (mL/min), excising and weighing the organ, and dividing 
out the weight to arrive at organ perfusion. In an imaging context normalising 
to weight is awkward as voxel volume is known, but voxel weight is not. 
Converting to historical units of mL/min/g would require inserting of 
literature values for tissue density in all models, and keeping track of 
these throughout computations. This is cumbersome and unnecessary, since 
conversion to mL/sec/g can always be done at the end of computations on the 
final output.

The use of cm3 for tissue volumes also ensures that important distinctions 
such as that between concentration and tissue concentration, are explicit. 
The term *concentration* in `dcmri` documentation is consistently used to 
designate amounts of indicator (mmol) relative to fluid volumes (mL), whereas 
the term *tissue concentration* refers to indicator amounts (mmol) per unit 
tissue volume (cm3). They are different physical quantities and in an imaging 
context the distinction is important. 


Units in plots
--------------

While `dcmri` rigorously adheres to standard units for all arguments and return 
values in functions - exceptions are made in plots or other presentations of 
data and results. Here `dcmri` will choose the units that are most natural and 
intuitive in conveying the key messages. For instance concentrations in plots 
will typically be shown in units of mM.

