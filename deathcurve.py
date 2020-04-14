'''
This code, the code of the affiliated shared C library and the sample
script.py may help you fit analytically expressed functions to the
COVID-19 mortality data to model age-adjusted mortality risk using
maximum likelihood point estimates.

This software was desinged in two parts: a shared C library to
dramatically speed up the calculation and a Python wrapper script
to feed the input data into the affiliated shared C library and
finally plot the fitted curves. A sample Python script.py file,
which helps more easily import and transform the input tabular data
and imports and communicates with the Python wrapper module, is also
attached.

The shared C library provides the opportunity to test arbitrary
functions on condition that, in the domain between 0 and 120+, they
return values between 0.0 and 1.0 - otherwise, in this scenario, it
will make no sense.

The Python wrapper interface function "fitFunctionWrapper" accepts
a two-column pandas DataFrame with:
- the first column 'age' of the numpy numerical data type, e.g.,
  numpy.float64 or numpy.intc (the float datatype allows to accomodate
  data that specify full dates of birth instead of years of birth),
- the second column 'outcome' of the numpy numerical data type, e.g.,
  numpy.intc, where non-zero (e.g., 1) means death and zero means a
  more positive outcome.

It return a tuple of two objects of the class "bestFit" defined in the
same wrapper module. The first object contains the calculated
parameters and the number of the best fitted function.

The attached script.py sample can be modified to supply case-by-case
data I don't yet have access to or have failed to find.

Copyright (C) 2020  Alexander Yuryatin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

print('''
Copyright (C) 2020  Alexander Yuryatin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
''')


import numpy as np
import pandas as pd
from scipy.special import erf
from ctypes import cdll, c_void_p, c_int
import matplotlib.pyplot as plt
from os.path import abspath
from typing import Tuple


class bestFit():
        
    def output(self):
        return self.outputText.format(np.power(10.0, self.b0),np.power(10.0, self.b1),np.power(10.0, self.b2),np.power(10.0, self.b3),np.power(10.0, self.b4),np.power(10.0, self.b5),np.power(10.0, self.b6),np.power(10.0, self.b7))

    def internalLogL(x, b0, b1, b2, b3, b4, b5, b6, b7):
        return np.log(np.power(10.0, b0) + np.power(10.0, b1) * x + np.power(10.0, b2) * (x ** 2) + np.power(10.0, b3) * (x ** 3) + np.power(10.0, b4) * (x ** 4) + np.power(10.0, b5) * (x ** 5) + np.power(10.0, b6) * (x ** 6) + np.power(10.0, b7) * (x ** 7))
    
    def internalLogS(x, b0, b1, b2, b3, b4, b5):
        return np.log(np.power(10.0, b0) + np.power(10.0, b1) * x + np.power(10.0, b2) * (x ** 2) + np.power(10.0, b3) * (x ** 3) + np.power(10.0, b4) * (x ** 4) + np.power(10.0, b5) * (x ** 5))

    def erf(self, x):
        return erf(bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)) * 0.5 + 0.5
    
    def erfFC(self, x):
        return erf(bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6
    
    def hyperbTan(self, x):
        return np.tanh(bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)) * 0.5 + 0.5
    
    def hyperbTanFC(self, x):
        return np.tanh(bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6

    def GudFunc(self, x):
        return np.arctan(np.tanh(bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7))) * 2.0 / np.pi + 0.5
    
    def GudFuncFC(self, x):
        return np.arctan(np.tanh(bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5))) * 4.0 * (0.5 - self.b7) / np.pi + 0.5 - self.b7 + self.b6

    def xOverX2(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        return temp * np.power(1 + np.power(temp, 2), -0.5) * 0.5 + 0.5
    
    def xOverX2FC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        return temp * np.power(1 + np.power(temp, 2), -0.5) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6

    def xOverAbs(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        return 0.5 * temp / (1 + np.abs(temp)) + 0.5
    
    def xOverAbsFC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        return (0.5 - self.b7) * temp / (1 + np.abs(temp)) + 0.5 - self.b7 + self.b6
        
    testFuncs = [ erf, erfFC, hyperbTan, hyperbTanFC, GudFunc, GudFuncFC, xOverX2, xOverX2FC, xOverAbs, xOverAbsFC ]     # list of functions to facilitate their calls by numbers
    
    testFuncsNames = ['Erf-derived function',
                      'Erf-derived function with floor and ceiling',
                      'Logistic-derived function',
                      'Logistic-derived function with floor and ceiling',
                      'Gudermannian-derived function',
                      'Gudermannian-derived function with floor and ceiling',
                      'Algebraic function derived from x over sqrt(1 + x^2)',
                      'Algebraic function derived from x over sqrt(1 + x^2) with floor and ceiling',
                      'Algebraic function derived from x over (1 + abs(x))',
                      'Algebraic function derived from x over (1 + abs(x)) with floor and ceiling']
        
    testFuncsReports = [
                        '\n\n\tPython\n\tfrom scipy.special import erf\n\terf(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) )) * 0.5 + 0.5\n\n\t' +
                        'Microsoft Excel\n\tERF(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | erf(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n',
                        '\n\n\tPython\n\tfrom scipy.special import erf\n\terf(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'Microsoft Excel\n\tERF(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'WolframAlpha\n\tplot | erf(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e} | x = 0 to 100\n\n',
                        '\n\n\tPython\n\tmath.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))/2 + 0.5\n\n\t' +
                        'Microsoft Excel\n\tTANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n',
                        '\n\n\tPython\n\tmath.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'Microsoft Excel\n\tTANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'WolframAlpha\n\tplot | tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 )) * (0.5 - {7:e}) + 0.5 - {7:e} + {6:e} | x = 0 to 100\n\n',
                        '\n\n\tPython\n\t2.0 * math.atan(math.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) )))/math.pi + 0.5\n\n\t' +
                        'Microsoft Excel\n\t2.0 * ATAN(TANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) )))/ PI() + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 2.0 * atan(tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 )))/pi + 0.5 | x = 0 to 100\n\n\n',
                        '\n\n\tPython\n\t4.0 * (0.5 - {7:e}) * math.atan(math.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) )))/math.pi + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'Microsoft Excel\n\t4.0 * (0.5 - {7:e}) * ATAN(TANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) )))/ PI() + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'WolframAlpha\n\tplot | 4.0 * (0.5 - {7:e}) * atan(tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 )))/pi + 0.5 - {7:e} + {6:e} | x = 0 to 100\n\n\n',
                        '\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/((1 + (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))**2)**0.5) + 0.5\n\n\t' +
                        'Microsoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/((1 + (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))^2)^0.5) + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 0.5 * (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ) )/((1 + (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))^2)^0.5) + 0.5 | x = 0 to 100\n\n',
                        '\n\n\tPython\n\t(0.5 - {7:e}) * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/((1 + (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) ))**2)**0.5) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'Microsoft Excel\n\t(0.5 - {7:e}) * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/((1 + (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) ))^2)^0.5) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'WolframAlpha\n\tplot | (0.5 - {7:e}) * (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ) )/((1 + (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 ))^2)^0.5) + 0.5 - {7:e} + {6:e} | x = 0 to 100\n\n',
                        '\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/(1 + abs(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))) + 0.5\n\n\t' +
                        'Microsoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/(1 + ABS(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))) + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 0.5 * log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7)/(1 + abs(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))) + 0.5 | x = 0 to 100\n\n',
                        '\n\n\tPython\n\t(0.5 - {7:e}) * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/(1 + abs(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) ))) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'Microsoft Excel\n\t(0.5 - {7:e}) * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/(1 + ABS(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) ))) + 0.5 - {7:e} + {6:e}\n\n\t' +
                        'WolframAlpha\n\tplot |  (0.5 - {7:e}) * log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7)/(1 + abs(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 ))) + 0.5 - {7:e} + {6:e} | x = 0 to 100\n\n' ]
                        
    def function(self, x) -> float:
        return self.testFuncs[self.best](self, x)
                        
    def __init__(self, parameters: np.ndarray, functionNumber: int):
        self.b0 = parameters[0]
        self.b1 = parameters[1]
        self.b2 = parameters[2]
        self.b3 = parameters[3]
        self.b4 = parameters[4]
        self.b5 = parameters[5]
        self.b6 = parameters[6]
        self.b7 = parameters[7]
        self.ml = parameters[8]
        self.best = functionNumber
        self.bestName = bestFit.testFuncsNames[functionNumber]
        self.outputText = bestFit.testFuncsReports[functionNumber]


def fitFunctionWrapper(df: pd.DataFrame) -> Tuple[bestFit, bestFit]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError('function fitFunctionWrapper accepts only pandas DataFrames')
    if df.shape[1] != 2:
        raise ValueError('function fitFunctionWrapper accepts pandas DataFrames with only two columns: age and outcome')
    df.columns = ['age', 'outcome']
    if 'float' not in df['age'].dtype.__str__() and 'int' not in df['age'].dtype.__str__():
        raise TypeError('the 1st column in the pandas DataFrame that function fitFunctionWrapper() accepts may only contain data of the numeric types, not {}'.format(df['age'].dtype.__str__()))
    if 'float' not in df['outcome'].dtype.__str__() and 'int' not in df['outcome'].dtype.__str__():
        raise TypeError('the 2nd column in the pandas DataFrame that function fitFunctionWrapper() accepts may only contain data of the numeric types, not {}'.format(df['outcome'].dtype.__str__()))

    # when working with these arrays from the shared C library, it is critical for them be continuous in memory
    age = np.ascontiguousarray(df['age'], dtype=np.float64)
    outcome = np.ascontiguousarray(df['outcome'], dtype=np.intc)
    output = np.ascontiguousarray(np.zeros(18, dtype=np.float64))   # the array to collect fitted parameters and ML estimates from the shared C library
    # the first 9 values are reserved for the best fitted function
    # the second 9 values are reserved for the second best fitted function
    # among those 9, the first 8 are for the fitted coefficients and the 9th is for the (logarithmic) ML point estimate
    res = 0     # the variable to collect the number of the best fitted function from the C interface function
    secondRes = np.zeros(1, dtype=np.intc)     # the variable to pass by reference to the C interface function to collect the number of the second best fitted function if necessary
    clib = cdll.LoadLibrary(abspath('libdeathcurve.so'))      # loading the compiled binary shared C library, which should be located in the same directory as this Python script; absolute path is more important for Linux — not necessary for MacOS
    f = clib.fitFunction       # assigning the C interface function to this Python variable "f"
    f.arguments = [ c_void_p, c_void_p, c_int, c_void_p, c_void_p]         # declaring the data types for C function arguments
    f.restype = c_int      # declaring the data types for C function return value
    res = f(c_void_p(age.ctypes.data), c_void_p(outcome.ctypes.data), age.size, c_void_p(output.ctypes.data), c_void_p(secondRes.ctypes.data))   # calling the C interface function
    return bestFit(output[:9], res), bestFit(output[9:], int(secondRes))


def reportModel(models: Tuple[bestFit, bestFit]) -> None:
    text_output = '\nBest fit is:' + models[0].output()
    if 'Gudermannian' in models[0].bestName:
        text_output += '\nSecond best fit is:' + models[1].output()

    print(text_output)

    # Saving the report text file in the same directory
    with open('report.txt', 'w') as f:
        f.write(text_output)


def plotModel(models: Tuple[bestFit, bestFit]) -> None:
    # Plotting the graph
    x = np.arange(0.01, 200.0, 0.01)
    y1 = np.array([100.0 * models[0].function(i) for i in x])

    if 'Gudermannian' in models[0].bestName:
        y2 = np.array([100.0 * models[1].function(i) for i in x])
    
        fig, subpl = plt.subplots( 1, 2, figsize=(12,6))
        fig.suptitle('Age-adjusted COVID-19 mortality', fontsize=16)
        fig.set_figheight(6)
        fig.set_figwidth(12)
        subpl[0].plot(x, y1)
        subpl[0].set_xlim(0.0, 100.0)
        subpl[0].set_ylim(0.0, max(100.0 * models[0].function(100.0), 100.0 * models[1].function(100.0) ) )
        subpl[0].set_xlabel('Age (years)')
        subpl[0].set_ylabel('Risk of death (%)')
        subpl[0].set_title(models[0].bestName)
        subpl[0].grid()
    
        subpl[1].plot(x, y2)
        subpl[1].set_xlim(0.0, 100.0)
        subpl[1].set_ylim(0.0, max(100.0 * models[0].function(100.0), 100.0 * models[1].function(100.0) ) )
        subpl[1].set_xlabel('Age (years)')
        subpl[1].set_title(models[1].bestName)
        subpl[1].grid()
    else:
        fig, subpl = plt.subplots( 1, 1, figsize=(6,6))
        fig.suptitle('Age-adjusted COVID-19 mortality', fontsize=16)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        subpl.plot(x, y1)
        subpl.set_xlim(0.0, 100.0)
        subpl.set_ylim(0.0, 100.0 * models[0].function(100.0))
        subpl.set_xlabel('Age (years)')
        subpl.set_ylabel('Risk of death (%)')
        subpl.set_title(models[0].bestName)
        subpl.grid()

    # Saving the graph image file in the same directory
    fig.savefig("result.png")
    plt.show()
