# COVID-19 Age-adjusted Mortality

The code to fit analytically expressed functions to model dependency of COVID-19 mortality on age.

The code of the Python scripts and of the affiliated shared C library may help you fit analytically expressed functions to the COVID-19 mortality data to model age-adjusted mortality risk using maximum likelihood point estimates.

This software was desinged in two parts: a shared C library to dramatically speed up the calculation and a Python wrapper script to feed the input data into the affiliated shared C library and finally plot the fitted curves. A sample Python *script.py* file, which helps more easily import and transform the input tabular data and imports and communicates with the Python wrapper module, is also attached.

The shared C library provides the opportunity to test arbitrary functions on condition that, in the domain between 0 and 120+, they return values between 0.0 and 1.0 — otherwise, in this scenario, it will make no sense.

## Python wrapper interface function
The Python wrapper interface function *fitFunctionWrapper()* accepts up to five arguments:
- a two-column *pandas DataFrame* (the only mandatory argument) with:
  - the first column 'age' of the numpy numerical data type, e.g., *numpy.float64* or *numpy.intc* (the float datatype allows to accomodate data that specify full dates of birth instead of years of birth)
  - the second column 'outcome' of the numpy numerical data type, e.g., *numpy.intc*, where non-zero (e.g., 1) means death and zero means a more positive outcome
- a string with signs for the up to eight coefficients (without specifying, only positive coefficients are going to be fitted, as in the package versions below 2.0), e.g., "++++++++", "-", "-+-+"
- a boolean argument specifying if you want to fit the coefficients with the signs starting from those specified in the previous parameter all the way to "--------" (*False*) or the signs specified in the previous parameter only (*True*). The defaule is 'False'
- a tuple of integers with the numbers of functions you want to fit (starting at zero): e.g., (0,), (0, 3), (5, 2), (0, 1, 4, 5, 6, 7, 8, 9)
- an integer with the order of the internal polymonial, which can be in the range from 2 to 7 (restricting the order for the functions' variants with floor and ceiling (functions 1, 3, 5, 7, 9) currently is not supported)

It returns an object of the class *bestFit* defined in the same wrapper module.

The attached *script.py* sample can be modified to supply case-by-case data I don't yet have access to or have failed to find.

Launching the fitting of all functions and for all coeficients' signs will occupy your laptop, workstation or server for many hours (if not days). In order to stop the execution and report the best fitted function so far, you may save the file with the name *stop.txt* empty or with any content in the same directory. The C code checks the presence of this file in the working directory at each round and properly finishes if that file is found. The *stop_script.py* file serves that purpose. Alternatively you may enter 'touch stop.txt' command in the terminal while in the working directory to create that file. Please do not forget to delete it afterward.

## Fitting outcomes for other acute conditions
Importantly, outcome data for acute conditions, for which the infant mortality is higher than the toddler mortality or the mortality in older children, shouldn't be fed into this Python wrapper function with all positive signs of the coefficients "++++++++" or without changes to the tested functions inside the shared C library, because currently four out of the five main tested functions are monotonic increasing functions. You may substitute them with "smile-shaped" functions for other acute conditions if you so desire.

Each of the five main sigmoid functions is supplemented with a version elevated above the x-axis and squeezed down from the asymptote y = 1.0 to account for the posibility that the risk of death doesn't start from zero at birth and never asymptotically achieves 1.0. Those variants, however, don't seem to model COVID-19 mortality better, though may become useful for other acute conditions or fit better with larger datasets.

## Compatibility
### C code:
    In order to speed up calculation, the C code uses POSIX threads. Therefore, this code is designed for MacOS and Linux environment, not natively for Windows.
### Python script:
    It needs the following non-standard modules and packages: numpy, pandas, scipy, and matplotlib.

## Environment to launch the Python script
*script.py* is best launched in the MacOS or Linux terminal window. Otherwise, your Python shell or IDE may not print the progress messages from the imported shared C library.

## Attached binary shared library
The attached *libdeathcurve.so* shared library binary file in the root folder was compiled from the attached *deathcurve.c* file for MacOS Catalina x86-64 using the attached *Makefile*. A shared library binary file that was compiled for Ubuntu can be found in a separate folder.

## Compilation
If you work on MacOS or Linux (tested on Ubuntu only), you may download *script.py*, *deathcurve.py*, *deathcurve.c*, and *Makefile* into the same directory, then open the Terminal window, proceed to that directory with 'cd' commands, and, if you already have the clang compiler installed, you may want to enter the command 'make' and press 'Enter'. This will compile *deathcurve.c* into the shared library *libdeathcurve.so*. If you don't have the clang compiler installed, you may want to install it or, alternatively, change clang to whatever compiler you wish (e.g., gcc) in *Makefile* before launching 'make'. After compilation, you may want to fetch the suggested test *csv* file (or put your dataset and rewrite *ingestData()* function in *script.py*) and launch *script.py* in the terminal window.

If you experience any difficulty doing that, those steps are shown in the demo video at https://youtu.be/HKwlgA16MF4

It demonstrates one of the previous versions 1.+ of this package.

## Output formulas formats
The Python script, when finishes, should both plot the best fitted curve and save into the same directory the report file that contains the formula for the best fitted function in the formats for:
* Python
* Microsoft Excel
* WolframAlpha

## What's new in the version 2
- The most radical change is that the package now fits the functions with arbitrary signs of coefficients. No need in hard coding those signs anymore. However, exhaustive fitting all of them will require significantly longer computation time. In the case of using this package for business application or adequately funded academic research, I highly recommend launching the calculation in the clould on a virtual machine with the largest number of CPU cores possible.
- The terms in the equation minimized virtualy to zero during the fitting used to be left in the formula. Now they are removed automatically and the best fitted equations are reported without them.
- The *fitFunctionWrapper()* function of the *deathcurve.py* module now returns one object of the *bestFit* class, not a tuple of two objects.
- *reportModel()* and *plotModel()* turned into methods of the *bestFit* class from standalone functions.
