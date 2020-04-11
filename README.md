# COVID-19 Age-adjusted Mortality

The code to fit analytically expressed functions to model dependency of COVID-19 mortality on age.

The code of the Python scripts and of the affiliated shared C library may help you fit analytically expressed functions to the COVID-19 mortality data to model age-adjusted mortality risk using maximum likelihood point estimates.

This software was desinged in two parts: a shared C library to dramatically speed up the calculation and a Python wrapper script to feed the input data into the affiliated shared C library and finally plot the fitted curves. A sample Python script.py file, which helps more easily import and transform the input tabular data and imports and communicates with the Python wrapper module, is also attached.

The shared C library provides the opportunity to test arbitrary functions on condition that, in the domain between 0 and 120+, they return values between 0.0 and 1.0 - otherwise, in this scenario, it will make no sense.

## Python wrapper interface function
The Python wrapper interface function "fitFunctionWrapper" accepts a two-column pandas DataFrame with:
- the first column 'age' of the numpy numerical data type, e.g., numpy.float64 or numpy.intc (the float datatype allows to accomodate data that specify full dates of birth instead of years of birth),
- the second column 'outcome' of the numpy numerical data type, e.g., numpy.intc, where non-zero (e.g., 1) means death and zero means a more positive outcome.

It return a tuple of two objects of the class "bestFit" defined in the same wrapper module. The first object contains the calculated parameters and the number of the best fitted function.

The attached script.py sample can be modified to supply case-by-case data I don't yet have access to or have failed to find.

## Fitting outcomes for other acute conditions
Importantly, outcome data for acute conditions, for which the infant mortality is higher than the toddler mortality or the mortality in older children, shouldn't be fed into this Python wrapper function without changes to the tested functions inside the shared C library, because currently four out of the five main tested functions are monotonic increasing functions. However, you may substitute them with "smile-shaped" functions for other acute conditions if you so desire.

Each of the five main functions is supplemented with a version elevated above the x-axis and squeezed down from the asymptote y = 1.0 to account for the posibility that the risk of death doesn't start from zero at birth and never asymptotically achieves 1.0. Those variants, however, don't seem to model COVID-19 mortality better, though may become useful for other acute conditions.

## Compatibility
### C code:
    In order to speed up calculation, the C code uses POSIX threads. Therefore, this code is designed for MacOS and Linux environment, not natively for Windows.
### Python script:
    It needs the following non-standard modules and packages: numpy, pandas, scipy, and matplotlib.

## Environment to launch the Python script
script.py is best launched in the MacOS or Linux terminal window. Otherwise, your Python shell or IDE may not print the progress messages from the imported shared C library.

## Attached binary shared library
The attached libdeathcurve.so shared library binary file in the root folder was compiled from the attached deathcurve.c file for MacOS Catalina x86-64 using the attached Makefile. A shared library binary file that was compiled for Ubuntu can be found in a separate folder.

## Compilation
If you work on MacOS or Linux (tested on Ubuntu only), you may download script.py, deathcurve.py, deathcurve.c, and Makefile into the same directory, then open the Terminal window, proceed to that directory with 'cd' commands, and, if you already have the clang compiler installed, you may want to enter the command 'make' and press 'Enter'. This will compile deathcurve.c into the shared library libdeathcurve.so. If you don't have the clang compiler installed, you may want to install it or, alternatively, change clang to whatever compiler you wish (e.g., gcc) in Makefile before launching 'make'. After compilation, you may want to fetch the suggested test csv file and launch script.py in the terminal window.

## Output formulas formats
The Python script, when finishes, should both plot the best fitted curve and save into the same directory the report file that contains the formula for the best fitted function in the formats for:
* Python
* Microsoft Excel
* WolframAlpha

## Remaining terms in the formula
For the sake of simplicity of the algorithm, the terms in the equation that were minimized virtualy to zero during the fitting are left in the formula. You are free to remove all close-to-zero terms manually if it doesn't change the curve.
