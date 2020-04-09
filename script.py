'''
This code and the code of the affiliated Python wrapper module and
shared C library may help you fit an analytically expressed function
to the COVID-19 mortality data to model age-adjusted mortality risk
using maximum likelihood point estimates.

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


import numpy as np
import pandas as pd
import deathcurve


# the function below is designed to import the case-by-case table from URL https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
# to preprocess other data sources, it should be modified
def ingestData(dataFile):
    df = pd.read_csv(dataFile)[['birth_year','symptom_onset_date','confirmed_date','state','released_date','deceased_date']]
    df[['symptom_onset_date']] = pd.to_datetime(df['symptom_onset_date'], errors='coerce')
    df[['confirmed_date']] = pd.to_datetime(df['confirmed_date'], errors='coerce')
    df[['deceased_date']] = pd.to_datetime(df['deceased_date'], errors='coerce')
    df[['released_date']] = pd.to_datetime(df['released_date'], errors='coerce')
    latestDate = df[['symptom_onset_date','confirmed_date','deceased_date','released_date']].max(skipna=True).max()
    df['earliest_date'] = df[['symptom_onset_date','confirmed_date']].min(skipna=True, axis=1)
    df['death_on_date'] = df['deceased_date'] - df['earliest_date']
    df = df[(latestDate - df.earliest_date > df['death_on_date'].dropna().quantile(.98)) & (df.birth_year.isna()==False)]    # it was an arbitrary decision to remove all cases that are younger than 98%-percentile of days-to-death from first symptoms or diagnosis confirmation whichever is earlier. You can adjust it with a more rigorous rationale. If all cases are left, (especially on the rising pandemics) the mortality is expected to be underestimated
    df['age'] = 2020 - df['birth_year']       # in case the dataset has full birth dates instead of birth years, the data type for this variable was left float64/double both in numpy/pandas and the shared C library
    df['outcome'] = np.where(df.state == 'deceased', 1, 0)
    df['age'] = np.where(df.age == 0.0, 1e-8, df.age)
    return df[['age','outcome']]


def main():
    # the function below is designed to import the case-by-case table from URL https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
    df = ingestData('PatientInfo.csv')
    bestFunction = deathcurve.fitFunctionWrapper(df)
    deathcurve.reportModel(bestFunction)
    deathcurve.plotModel(bestFunction)
    
    
if __name__ == '__main__':
    main()
