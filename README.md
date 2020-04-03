# covid19_age_adjusted_mortality
The code to fit an analytically expressed function to model dependency of COVID-19 mortality on age.

The code of the Python script and of the affiliated C shared library may help you fit an analytically expressed function to the COVID-19 mortality data to model age-adjusted mortality risk using maximum likelihood point estimates.

This software was desinged in two parts: a Python wrapper script to help more easily import and transform the input tabular data and finally plot the fitted curves, and a C shared library to dramatically speed up the calculation.

The C shared library provides the opportunity to test arbitraty functions on condition that, in the domain between 0 and 120+, they return natural logarithms of values between 0.0 and 1.0 - otherwise, in this scenario, it will make no sense.

The C library interface fitFunction function accepts two arrays and 3 other parameters:
- an array of subjects' ages (of the data type 'double' to accomodate data that specify full dates of birth instead of years of birth)
- an array of subjects' outcomes, where '1' is death and '0' is a more positive outcome (of the data type 'int')
- the length of those two arrays (of the data type 'int')
- an array of 16 double-precision floats to accomodate the calculated parameters the interface function returns
- a pointer to an 'int' variable to return the number of the second best fitted function, if necessary.
So the Python script can be modified to supply case-by-case data I don't yet have access to or failed to find.
