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
from math import ceil
from ctypes import cdll, c_void_p, c_int
import matplotlib.pyplot as plt
from os.path import abspath
from typing import Tuple


class bestFit():

    formats = dict(Python=['*x','*(x**{:d})'],
                   Excel=['*A1','*(A1**{:d})'],
                   Wolfram=[' x',' x^{:d}'])

    def outputLog(self, format, short = False):
        length = 5 if short else 7
        output = ''
        startSign = False
        for i, par in enumerate(self.params[:length]):
            if par:
                if par > 0.0:
                    if startSign:
                        output += ' + {:.6e}'.format(par)
                        if i == 1:
                            output += bestFit.formats[format][0]
                        elif i > 1:
                            output += bestFit.formats[format][1].format(i)
                    else:
                        startSign = True
                        output += '{:.6e}'.format(par)
                        if i == 1:
                            output += bestFit.formats[format][0]
                        elif i > 1:
                            output += bestFit.formats[format][1].format(i)
                if par < 0.0:
                    if startSign:
                        output += ' - {:.6e}'.format(abs(par))
                        if i == 1:
                            output += bestFit.formats[format][0]
                        elif i > 1:
                            output += bestFit.formats[format][1].format(i)
                    else:
                        startSign = True
                        output += '{:.6e}'.format(par)
                        if i == 1:
                            output += bestFit.formats[format][0]
                        elif i > 1:
                            output += bestFit.formats[format][1].format(i)
                        startSign == True
        return output
        
    def output(self):
        return self.outputText.format(self.outputLog('Python', bestFit.testFuncs[self.best].short), self.outputLog('Excel', bestFit.testFuncs[self.best].short), self.outputLog('Wolfram', bestFit.testFuncs[self.best].short), self.b6, self.b7, '0.0 if x < {:.6f} else '.format(ceil(-self.b0 * 1e6) * 1e-6) if self.b0 < 0.0 else '', format(ceil(-self.b0 * 1e6) * 1e-6))

    def internalLogL(x, b0, b1, b2, b3, b4, b5, b6, b7):
        return b0 + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b4 * (x ** 4) + b5 * (x ** 5) + b6 * (x ** 6) + b7 * (x ** 7)
    
    def internalLogS(x, b0, b1, b2, b3, b4, b5):
        return b0 + b1 * x + b2 * (x ** 2) + b3 * (x ** 3) + b4 * (x ** 4) + b5 * (x ** 5)

    def erf(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return erf(temp) * 0.5 + 0.5
    erf.short = False
    
    def erfFC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return erf(temp) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6
    erfFC.short = True
    
    def hyperbTan(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return np.tanh(temp) * 0.5 + 0.5
    hyperbTan.short = False
    
    def hyperbTanFC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return np.tanh(temp) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6
    hyperbTanFC.short = True

    def GudFunc(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return np.arctan(np.tanh(temp)) * 2.0 / np.pi + 0.5
    GudFunc.short = False
    
    def GudFuncFC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return np.arctan(np.tanh(temp)) * 4.0 * (0.5 - self.b7) / np.pi + 0.5 - self.b7 + self.b6
    GudFuncFC.short = True

    def xOverX2(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return temp * np.power(1 + np.power(temp, 2), -0.5) * 0.5 + 0.5
    xOverX2.short = False
    
    def xOverX2FC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return temp * np.power(1 + np.power(temp, 2), -0.5) * (0.5 - self.b7) + 0.5 - self.b7 + self.b6
    xOverX2FC.short = True

    def xOverAbs(self, x):
        temp = bestFit.internalLogL(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return 0.5 * temp / (1 + np.abs(temp)) + 0.5
    xOverAbs.short = False
    
    def xOverAbsFC(self, x):
        temp = bestFit.internalLogS(x, self.b0, self.b1, self.b2, self.b3, self.b4, self.b5)
        if(temp) <= 0.0: temp = -np.inf
        else: temp = np.log(temp)
        return (0.5 - self.b7) * temp / (1 + np.abs(temp)) + 0.5 - self.b7 + self.b6
    xOverAbsFC.short = True
        
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
                        '\n\n\tPython\n\tfrom scipy.special import erf\n\t{5:s}erf(math.log({0:s})) * 0.5 + 0.5\n\n\t' +
                        'Microsoft Excel\n\tERF(LN({1:s}))/2 + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | erf(log({2:s}))/2 + 0.5 | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\tfrom scipy.special import erf\n\t{5:s}erf(math.log({0:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'Microsoft Excel\n\tERF(LN({1:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'WolframAlpha\n\tplot | erf(log({2:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e} | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}math.tanh(math.log({0:s}))/2 + 0.5\n\n\t' +
                        'Microsoft Excel\n\tTANH(LN({1:s}))/2 + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | tanh(log({2:s}))/2 + 0.5 | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}math.tanh(math.log({0:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'Microsoft Excel\n\tTANH(LN({1:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'WolframAlpha\n\tplot | tanh(log({2:s})) * (0.5 - {4:e}) + 0.5 - {4:e} + {3:e} | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}2.0 * math.atan(math.tanh(math.log({0:s})))/math.pi + 0.5\n\n\t' +
                        'Microsoft Excel\n\t2.0 * ATAN(TANH(LN({1:s})))/ PI() + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 2.0 * atan(tanh(log({2:s})))/pi + 0.5 | x = {6:s} to 100\n\n\n',
                        '\n\n\tPython\n\t{5:s}4.0 * (0.5 - {4:e}) * math.atan(math.tanh(math.log({0:s})))/math.pi + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'Microsoft Excel\n\t4.0 * (0.5 - {4:e}) * ATAN(TANH(LN({1:s})))/ PI() + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'WolframAlpha\n\tplot | 4.0 * (0.5 - {4:e}) * atan(tanh(log({2:s})))/pi + 0.5 - {4:e} + {3:e} | x = {6:s} to 100\n\n\n',
                        '\n\n\tPython\n\t{5:s}0.5 * (math.log({0:s}) )/((1 + (math.log({0:s}))**2)**0.5) + 0.5\n\n\t' +
                        'Microsoft Excel\n\t0.5 * (LN({1:s}) )/((1 + (LN({1:s}))^2)^0.5) + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 0.5 * (log({2:s}) )/((1 + (log({2:s}))^2)^0.5) + 0.5 | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}(0.5 - {4:e}) * (math.log({0:s}) )/((1 + (math.log({0:s}))**2)**0.5) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'Microsoft Excel\n\t(0.5 - {4:e}) * (LN({1:s}) )/((1 + (LN({1:s}))^2)^0.5) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'WolframAlpha\n\tplot | (0.5 - {4:e}) * (log({2:s}) )/((1 + (log({2:s}))^2)^0.5) + 0.5 - {4:e} + {3:e} | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}0.5 * (math.log({0:s}) )/(1 + abs(math.log({0:s}))) + 0.5\n\n\t' +
                        'Microsoft Excel\n\t0.5 * (LN({1:s}) )/(1 + ABS(LN({1:s}))) + 0.5\n\n\t' +
                        'WolframAlpha\n\tplot | 0.5 * log({2:s})/(1 + abs(log({2:s}))) + 0.5 | x = {6:s} to 100\n\n',
                        '\n\n\tPython\n\t{5:s}(0.5 - {4:e}) * (math.log({0:s}) )/(1 + abs(math.log({0:s}))) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'Microsoft Excel\n\t(0.5 - {4:e}) * (LN({1:s}) )/(1 + ABS(LN({1:s}))) + 0.5 - {4:e} + {3:e}\n\n\t' +
                        'WolframAlpha\n\tplot |  (0.5 - {4:e}) * log({2:s})/(1 + abs(log({2:s}))) + 0.5 - {4:e} + {3:e} | x = {6:s} to 100\n\n' ]
                        
    def function(self, x) -> float:
        return self.testFuncs[self.best](self, x)
        
    def reportModel(self) -> None:
        text_output = '\nBest fit is:' + self.output()
        print(text_output)

        # Saving the report text file in the same directory
        with open('report.txt', 'w') as f:
            f.write(text_output)

    def plotModel(self) -> None:
        # Plotting the graph
        x = np.arange(0.01, self.maxAge, 0.01)
        y = np.array([100.0 * self.function(i) for i in x])

        fig, subpl = plt.subplots( 1, 1, figsize=(7,6))
        fig.suptitle('Age-adjusted COVID-19 mortality', fontsize=16)
        fig.set_figheight(6)
        fig.set_figwidth(7)
        subpl.plot(x, y)
        subpl.set_xlim(0.0, self.maxAge)
        subpl.set_ylim(0.0, 100.0 * self.function(self.maxAge))
        subpl.set_ylabel('Risk of death (%)')
        subpl.set_xlabel('Age (years)')
        subpl.set_title(self.bestName)
        plt.subplots_adjust(left=0.17)
        subpl.grid()

        # Saving the graph image file in the same directory
        fig.savefig("result.png")
        plt.show()
                        
    def __init__(self, parameters: np.ndarray, functionNumber: int, sign: int, maxAge: float):
        self.b0 = parameters[0]
        self.b1 = parameters[1]
        self.b2 = parameters[2]
        self.b3 = parameters[3]
        self.b4 = parameters[4]
        self.b5 = parameters[5]
        self.b6 = parameters[6]
        self.b7 = parameters[7]
        self.params = tuple(parameters[:8])
        self.ml = parameters[8]
        self.signs = sign
        self.best = functionNumber
        self.bestName = bestFit.testFuncsNames[functionNumber]
        self.outputText = bestFit.testFuncsReports[functionNumber]
        self.maxAge = maxAge
        
        
def _strToSigns(signs: str) -> int:
    result = 0
    for i, letter in enumerate(signs):
        if letter == '-':
            result += 2 ** i
    return result


def fitFunctionWrapper(df: pd.DataFrame, signs: str = None, oneSignSet: bool = False, functions: Tuple = tuple(range(len(bestFit.testFuncs)))) -> bestFit:
    if not isinstance(df, pd.DataFrame):
        raise TypeError('function fitFunctionWrapper accepts only pandas DataFrames as a first parameter')
    if df.shape[1] != 2:
        raise ValueError('function fitFunctionWrapper accepts as a first parameter pandas DataFrames with only two columns: age and outcome')
    df.columns = ['age', 'outcome']
    if 'float' not in df['age'].dtype.__str__() and 'int' not in df['age'].dtype.__str__():
        raise TypeError('the 1st column in the pandas DataFrame that function fitFunctionWrapper() accepts may only contain data of the numeric types, not {}'.format(df['age'].dtype.__str__()))
    if 'float' not in df['outcome'].dtype.__str__() and 'int' not in df['outcome'].dtype.__str__():
        raise TypeError('the 2nd column in the pandas DataFrame that function fitFunctionWrapper() accepts may only contain data of the numeric types, not {}'.format(df['outcome'].dtype.__str__()))
    if any(df['age'] < 0.0) or any(df['age'] > 140.0):
        raise ValueError('function fitFunctionWrapper accepts as a first parameter pandas DataFrames with the first column \'age\' with the values between 0.0 and 140.0 only')
    if signs != None and not isinstance(signs, str):
        raise TypeError('argument \'signs\' of the function fitFunctionWrapper accepts only strings')
    if signs:
        if len(signs) > 8:
            raise ValueError('argument \'signs\' of the function fitFunctionWrapper accepts only strings of the length up to 8 (with \'+\' and \'-\')')
        for letter in signs:
            if letter != '+' and letter != '-':
                raise ValueError('argument \'signs\' of the function fitFunctionWrapper accepts only strings with \'+\' and \'-\'')
    if not isinstance(functions, Tuple):
        raise ValueError('argument \'functions\' of the function fitFunctionWrapper accepts only tuples of integers')
    for i in functions:
        if not isinstance(i, int):
            raise TypeError('argument \'functions\' of the function fitFunctionWrapper should contain only integers')
        if i not in range(len(bestFit.testFuncs)):
            raise ValueError('argument \'functions\' of the function fitFunctionWrapper should contain only integers between 0 and {} inclusive'.format(len(bestFit.testFuncs)))
    functions = set(functions)
    # when working with these arrays from the shared C library, it is critical for them be continuous in memory
    age = np.ascontiguousarray(df['age'], dtype=np.float64)
    outcome = np.ascontiguousarray(df['outcome'], dtype=np.intc)
    output = np.ascontiguousarray(np.zeros(18, dtype=np.float64))   # the array to collect fitted parameters and ML estimates from the shared C library
    # the first 9 values are reserved for the best fitted function
    # the second 9 values are reserved for the second best fitted function
    # among those 9, the first 8 are for the fitted coefficients and the 9th is for the (logarithmic) ML point estimate
    res = 0     # the variable to collect the number of the best fitted function from the C interface function
    secondRes = np.zeros(1, dtype=np.intc)     # the variable to pass by reference to the C interface function to collect the number of the second best fitted function if necessary
    sign1 = np.zeros(1, dtype=np.intc)
    sign2 = np.zeros(1, dtype=np.intc)
    if signs:
        sign1 = np.array(np.intc(_strToSigns(signs)))
    if signs and oneSignSet:
        sign2 = np.array(np.intc(1))
    functionsToFit = np.ascontiguousarray(np.zeros(10, dtype=np.intc))
    for i in range(len(bestFit.testFuncs)):
        if i in functions: functionsToFit[i] = 1
    clib = cdll.LoadLibrary(abspath('libdeathcurve.so'))      # loading the compiled binary shared C library, which should be located in the same directory as this Python script; absolute path is more important for Linux — not necessary for MacOS
    f = clib.fitFunction       # assigning the C interface function to this Python variable "f"
    f.arguments = [ c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p ]         # declaring the data types for C function arguments
    f.restype = c_int      # declaring the data types for C function return value
    res = f(c_void_p(age.ctypes.data), c_void_p(outcome.ctypes.data), age.size, c_void_p(output.ctypes.data), c_void_p(secondRes.ctypes.data), c_void_p(sign1.ctypes.data), c_void_p(sign2.ctypes.data), c_void_p(functionsToFit.ctypes.data))   # calling the C interface function
    return bestFit(output[:9], res, int(sign1), float(df['age'].max()))
