# covid19_age_adjusted_mortality

The code to fit an analytically expressed function to model dependency of COVID-19 mortality on age.

The code of the Python script and of the affiliated shared C library may help you fit an analytically expressed function to the COVID-19 mortality data to model age-adjusted mortality risk using maximum likelihood point estimates.

This software was desinged in two parts: a Python wrapper script to help more easily import and transform the input tabular data and finally plot the fitted curves, and a shared C library to dramatically speed up the calculation.

The shared C library provides the opportunity to test arbitraty functions on condition that, in the domain between 0 and 120+, they return values between 0.0 and 1.0 - otherwise, in this scenario, it will make no sense.

## C interface function
The C library interface "fitFunction" function accepts three arrays and two other parameters:
- an array of subjects' ages (of the data type 'double' to accomodate data that specify full dates of birth instead of years of birth)
- an array of subjects' outcomes, where '1' is death and '0' is a more positive outcome (of the data type 'int')
- the length of those two arrays (of the data type 'int')
- an array of 16 double-precision floats to accomodate the calculated parameters the interface function returns
- a pointer to an 'int' variable to return the number of the second best fitted function, if necessary.

So the Python script can be modified to supply case-by-case data I don't yet have access to or have failed to find.

## Fitting outcomes for other acute conditions
Imortantly, outcome data for acute conditions, for which the infant mortality is higher than the toddler mortality or the mortality in older children, shouldn't be fed into this C interface function without changes to the tested functions inside the shared C library, because currently four out of the five tested functions are monotonic increasing functions. However, you may substitute them with "smile-shaped" functions for other acute conditions if you so desire.

## Compatibility
### C code:
    In order to speed up calculation, the C code uses POSIX threads. Therefore, this code is designed for MacOS and Linux environment, not natively for Windows.
### Python script:
    It needs the following non-standard modules: numpy, pandas, and packages scipy and matplotlib.

## Environment to launch the Python script
The Python script is best launched in the MacOS or Linux terminal window. Otherwise, your Python shell or IDE may not print the progress messages from the imported shared C library.

## Attached binary shared library
The attached deathcurve.so shared library binary file has been compiled from the attached deathcurve.c file for MacOS Catalina x86-64 using the attached Makefile.

## Compilation
If you work on MacOS or Linux, you may download deathcurve.py, deathcurve.c, and Makefile into the same directory, then open the Terminal window, proceed to that directory with 'cd' commands, and, if you already have the clang compiler installed, you may want to enter the command 'make' and press 'Enter'. This will compile deathcurve.c into the shared library deathcurve.so. If you don't have the clang compiler installed, you may want to install it or, alternatively, change clang to whatever compiler you wish (e.g., gcc) in Makefile before launching 'make'.

## Output formulas formats
The Python script, when finishes, should both plot the best fitted curve and save into the same directory the report file that contains the formula for the best fitted function in the formats for:
* Python
* Microsoft Excel
* WolframAlpha

## Remaining terms in the formula
For the sake of simplicity of the algorithm, the terms of the equation that were minimized virtualy to zero during the fitting are left in the formula. You are free to remove all close-to-zero terms manually if it doesn't change the curve.
