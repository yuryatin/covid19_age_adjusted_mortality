'''
This code and the code of the affiliated shared C library may help
you fit an analytically expressed function to the COVID-19 mortality
data to model age-adjusted mortality risk using maximum likelihood
point estimates.

This software was desinged in two parts: a Python wrapper script to
help more easily import and transform the input tabular data and
finally plot the fitted curves, and a shared C library to dramatically
speed up the calculation.

The shared C library provides the opportunity to test arbitraty
functions on condition that, in the domain between 0 and 120+, they
return natural logarithms of values between 0.0 and 1.0 - otherwise
in this scenario, it will make no sense.

The C library interface fitFunction() function accepts three arrays
and two other parameters:
- array of subjects' ages (of the data type 'double' to accomodate
    data that specify full dates of birth instead of years of birth)
- array of subjects' outcomes, where '1' is death and '0' is a more
positive outcome (of the data type 'int')
- the length of those arrays (of the data type 'int')
- array of 16 double-precision floats to accomodate the calculated
    parameters the interface function returns
- a pointer to an 'int' variable to return the number of the second
    best fit function, if necessary
So the Python script can be modified to supply case-by-case data I
don't yet have access to.

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
import scipy.special as ss
import sys
from  ctypes import *
import matplotlib.pyplot as plt
import scipy.special as ss

def internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7):
    return np.log(np.power(10.0, b0) + np.power(10.0, b1) * x + np.power(10.0, b2) * (x ** 2) + np.power(10.0, b3) * (x ** 3) + np.power(10.0, b4) * (x ** 4) + np.power(10.0, b5) * (x ** 5) + np.power(10.0, b6) * (x ** 6) + np.power(10.0, b7) * (x ** 7))

def erf(x, b0, b1, b2, b3, b4, b5, b6, b7):
    return ss.erf(internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7)) * 0.5 + 0.5
    
def hyperbTan(x, b0, b1, b2, b3, b4, b5, b6, b7):
    return np.tanh(internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7)) * 0.5 + 0.5

def GudFunc(x, b0, b1, b2, b3, b4, b5, b6, b7):
    return np.arctan(np.tanh(internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7))) * 2.0 / np.pi + 0.5

def xOverX2(x, b0, b1, b2, b3, b4, b5, b6, b7):
    temp = internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7)
    return temp * np.power(1 +np.power(temp, 2), -0.5) * 0.5 + 0.5

def xOverAbs(x, b0, b1, b2, b3, b4, b5, b6, b7):
    temp = internalLog(x, b0, b1, b2, b3, b4, b5, b6, b7)
    return 0.5 * temp / (1 +np.abs(temp)) + 0.5

testFuncs = [ erf, hyperbTan, GudFunc, xOverX2, xOverAbs ]     # list of functions to facilitate their calls by numbers
testFuncsNames = ['Erf-derived function', 'Logistic-derived function', 'Gudermannian-derived function', 'Algebraic function derived from x over sqrt(1 + x^2)', 'Algebraic function derived from x over (1 + abs(x))' ]

# the section below is designed to import the case-by-case table from URL https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
df = pd.read_csv('PatientInfo.csv')[['birth_year','symptom_onset_date','confirmed_date','state','released_date','deceased_date']]
df[['symptom_onset_date']] = pd.to_datetime(df['symptom_onset_date'])
df[['confirmed_date']] = pd.to_datetime(df['confirmed_date'])
df[['deceased_date']] = pd.to_datetime(df['deceased_date'])
df[['released_date']] = pd.to_datetime(df['released_date'])
latestDate = df[['symptom_onset_date','confirmed_date','deceased_date','released_date']].max(skipna=True).max()
df['earliest_date'] = df[['symptom_onset_date','confirmed_date']].min(skipna=True, axis=1)
df['death_on_date'] = df['deceased_date'] - df['earliest_date']

df = df[(latestDate - df.earliest_date > df['death_on_date'].dropna().quantile(.98)) & (df.birth_year.isna()==False)]    # it was an arbitrary decision to remove all cases that are younger than 98%-percentile of days-to-death from first symptoms or diagnosis confirmation whichever is earlier. You can adjust it to make with a more rigorous rationale. If left all cases (especially on the rising pandemics), the mortality is expected to be underestimated
df['age'] = 2020 - df['birth_year']       # in case the dataset has full birth date instead of birth year, the data type for this variable was left float64/double both for numpy/pandas and the shared C library
df['outcome'] = np.where(df.state == 'deceased', 1, 0)
df['age'] = np.where(df.age == 0.0, 1e-8, df.age)
df = df[['age','outcome']]

# when working with these arrays from the shared C library, it is critical for them be continuous in memory
age = np.ascontiguousarray(df['age'], dtype=np.float64)
outcome = np.ascontiguousarray(df['outcome'], dtype=np.intc)

output = np.ascontiguousarray(np.zeros(18, dtype=np.float64))   # the array to collect fitted parameters and ML estimates from the shared C library
# the first 9 values are reserved for the best fitted function
# the second 9 values are reserved for the second best fitted function
# among those 9, the first 8 are for the fitted coefficients and the 9th is for the (logarithmic) ML point estimate

res = 0     # the variable to collect the number of the best fitted function from the C interface function
secondRes = np.zeros(1, dtype=np.intc)     # the variable to pass by reference to the C interface function to collect the number of the second best fitted function if necessary
clib = cdll.LoadLibrary('deathcurve.so')      # loading the compiled binary shared C library, which should be located in the same directory as this Python script
f = clib.fitFunction       # assigning the C interface function to this Python variable "f"
f.arguments = [ c_void_p, c_void_p, c_int, c_void_p, c_void_p]         # declaring the data types for C function arguments
f.restype = c_int      # declaring the data types for C function return value
res = f(c_void_p(age.ctypes.data), c_void_p(outcome.ctypes.data), age.size, c_void_p(output.ctypes.data), c_void_p(secondRes.ctypes.data))   # calling the C interface function

text_output = ''            # string to collect the text to save in the report text file and to print in the terminal

if(res==0):
    text_output += '\nBest fit is:\n\n\tPython\n\tscipy.special.erf(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))/2 + 0.5\n\n\tMicrosoft Excel\n\tERF(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\tWolframAlpha\n\tplot | erf(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[0]),np.power(10.0,output[1]),np.power(10.0,output[2]),np.power(10.0,output[3]),np.power(10.0,output[4]),np.power(10.0,output[5]),np.power(10.0,output[6]),np.power(10.0,output[7]))
elif(res==1):
    text_output += '\nBest fit is:\n\n\tPython\n\tmath.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))/2 + 0.5\n\n\tMicrosoft Excel\n\tTANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\tWolframAlpha\n\tplot | tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[0]),np.power(10.0,output[1]),np.power(10.0,output[2]),np.power(10.0,output[3]),np.power(10.0,output[4]),np.power(10.0,output[5]),np.power(10.0,output[6]),np.power(10.0,output[7]))
elif(res==3):
    text_output += '\nBest fit is:\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/((1 + (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))**2)**0.5) + 0.5\n\n\tMicrosoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/((1 + (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))^2)^0.5) + 0.5\n\n\tWolframAlpha\n\tplot | 0.5 * (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ) )/((1 + (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))^2)^0.5) + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[0]),np.power(10.0,output[1]),np.power(10.0,output[2]),np.power(10.0,output[3]),np.power(10.0,output[4]),np.power(10.0,output[5]),np.power(10.0,output[6]),np.power(10.0,output[7]))
elif(res==4):
    text_output += '\nBest fit is:\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/(1 + abs(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))) + 0.5\n\n\tMicrosoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/(1 + ABS(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))) + 0.5\n\n\tWolframAlpha\n\tplot | 0.5 * log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7)/(1 + abs(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))) + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[0]),np.power(10.0,output[1]),np.power(10.0,output[2]),np.power(10.0,output[3]),np.power(10.0,output[4]),np.power(10.0,output[5]),np.power(10.0,output[6]),np.power(10.0,output[7]))
elif(res==2):
    text_output += '\nBest fit is:\n\n\tPython\n\t2.0 * math.atan(math.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) )))/math.pi + 0.5\n\n\tMicrosoft Excel\n\t2.0 * ATAN(TANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) )))/ PI() + 0.5\n\n\tWolframAlpha\n\tplot | 2.0 * atan(tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 )))/pi + 0.5 | x = 0 to 100\n\n\n'.format(np.power(10.0,output[0]),np.power(10.0,output[1]),np.power(10.0,output[2]),np.power(10.0,output[3]),np.power(10.0,output[4]),np.power(10.0,output[5]),np.power(10.0,output[6]),np.power(10.0,output[7]))
    if(secondRes==0):
        text_output += '\nSecond best fit is:\n\n\tPython\n\tscipy.special.erf(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))/2 + 0.5\n\n\tMicrosoft Excel\n\tERF(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\tWolframAlpha\n\tplot | erf(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[9]),np.power(10.0,output[10]),np.power(10.0,output[11]),np.power(10.0,output[12]),np.power(10.0,output[13]),np.power(10.0,output[14]),np.power(10.0,output[15]),np.power(10.0,output[16]))
    elif(secondRes==1):
        text_output += '\nSecond best fit is:\n\n\tPython\n\tmath.tanh(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))/2 + 0.5\n\n\tMicrosoft Excel\n\tTANH(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))/2 + 0.5\n\n\tWolframAlpha\n\tplot | tanh(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))/2 + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[9]),np.power(10.0,output[10]),np.power(10.0,output[11]),np.power(10.0,output[12]),np.power(10.0,output[13]),np.power(10.0,output[14]),np.power(10.0,output[15]),np.power(10.0,output[16]))
    elif(secondRes==3):
        text_output += '\nSecond best fit is:\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/((1 + (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))**2)**0.5) + 0.5\n\n\tMicrosoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/((1 + (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))^2)^0.5) + 0.5\n\n\tWolframAlpha\n\tplot | 0.5 * (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ) )/((1 + (log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))^2)^0.5) + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[9]),np.power(10.0,output[10]),np.power(10.0,output[11]),np.power(10.0,output[12]),np.power(10.0,output[13]),np.power(10.0,output[14]),np.power(10.0,output[15]),np.power(10.0,output[16]))
    elif(secondRes==4):
        text_output += '\nSecond best fit is:\n\n\tPython\n\t0.5 * (math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ) )/(1 + abs(math.log({0:e} + {1:e}*x + {2:e}*(x**2) + {3:e}*(x**3) + {4:e}*(x**4) + {5:e}*(x**5) + {6:e}*(x**6) + {7:e}*(x**7) ))) + 0.5\n\n\tMicrosoft Excel\n\t0.5 * (LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ) )/(1 + ABS(LN({0:e} + {1:e}*A1 + {2:e}*(A1^2) + {3:e}*(A1^3) + {4:e}*(A1^4) + {5:e}*(A1^5) + {6:e}*(A1^6) + {7:e}*(A1^7) ))) + 0.5\n\n\tWolframAlpha\n\tplot | 0.5 * log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7)/(1 + abs(log({0:e} + {1:e} x + {2:e} x^2 + {3:e} x^3 + {4:e} x^4 + {5:e} x^5 + {6:e} x^6 + {7:e} x^7 ))) + 0.5 | x = 0 to 100\n\n'.format(np.power(10.0,output[9]),np.power(10.0,output[10]),np.power(10.0,output[11]),np.power(10.0,output[12]),np.power(10.0,output[13]),np.power(10.0,output[14]),np.power(10.0,output[15]),np.power(10.0,output[16]))

print(text_output)

#Saving the report text file in the same directory
with open('report.txt', 'w') as f:
    f.write(text_output)


# Plotting the graph
x = np.arange(0.01, 200.0, 0.01)
y1 = np.array([100.0 * testFuncs[res](i, *output[:8]) for i in x])


if(res==2):
    y2 = np.array([100.0 * testFuncs[int(secondRes)](i, *output[9:17]) for i in x])
    
    fig, subpl = plt.subplots( 1, 2, figsize=(12,6))
    fig.suptitle('Age-adjusted COVID-19 mortality', fontsize=16)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    subpl[0].plot(x, y1)
    subpl[0].set_xlim(0.0, 100.0)
    subpl[0].set_ylim(0.0, max(100.0 * testFuncs[res](100.0, *output[:8]) , 100.0 * testFuncs[int(secondRes)](100.0, *output[9:17]) ) )
    subpl[0].set_xlabel('Age (years)')
    subpl[0].set_ylabel('Risk of death (%)')
    subpl[0].set_title(testFuncsNames[res])
    subpl[0].grid()
    
    subpl[1].plot(x, y2)
    subpl[1].set_xlim(0.0, 100.0)
    subpl[1].set_ylim(0.0, max(100.0 * testFuncs[res](100.0, *output[:8]) , 100.0 * testFuncs[int(secondRes)](100.0, *output[9:17]) ) )
    subpl[1].set_xlabel('Age (years)')
    subpl[1].set_title(testFuncsNames[int(secondRes)])
    subpl[1].grid()
else:
    fig, subpl = plt.subplots( 1, 1, figsize=(6,6))
    fig.suptitle('Age-adjusted COVID-19 mortality', fontsize=16)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    subpl.plot(x, y1)
    subpl.set_xlim(0.0, 100.0)
    subpl.set_ylim(0.0, max(100.0 * testFuncs[res](100.0, *output[:8]) , 100.0 * testFuncs[int(secondRes)](100.0, *output[:8]) ) )
    subpl.set_xlabel('Age (years)')
    subpl.set_ylabel('Risk of death (%)')
    subpl.set_title(testFuncsNames[int(secondRes)])
    subpl.grid()

#Saving the graph image file in the same directory
fig.savefig("result.png")
plt.show()
