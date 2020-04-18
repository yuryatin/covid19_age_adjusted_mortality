'''
This code, the code of the affiliated Python script and shared C
library may help you fit analytically expressed functions to the
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


import numpy as np
import pandas as pd
import deathcurve


# the function below is designed to import the case-by-case table from URL https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
# to preprocess other data sources, it should be modified
def ingestData(dataFile: str) -> pd.DataFrame:
    df = pd.read_csv(dataFile)[['birth_year','symptom_onset_date','confirmed_date','state','released_date','deceased_date']]
    df[['symptom_onset_date']] = pd.to_datetime(df['symptom_onset_date'], errors='coerce')
    df[['confirmed_date']] = pd.to_datetime(df['confirmed_date'], errors='coerce')
    df[['deceased_date']] = pd.to_datetime(df['deceased_date'], errors='coerce')
    df[['released_date']] = pd.to_datetime(df['released_date'], errors='coerce')
    latestDate = df[['symptom_onset_date','confirmed_date','deceased_date','released_date']].max(skipna=True).max()
    df['earliest_date'] = df[['symptom_onset_date','confirmed_date']].min(skipna=True, axis=1)
    df['death_on_date'] = df['deceased_date'] - df['earliest_date']
    df = df[(latestDate - df.earliest_date > df['death_on_date'].dropna().quantile(.95)) & (df.birth_year.isna()==False)]    # it was an arbitrary decision to remove all cases that are younger than 95%-percentile of days-to-death from first symptoms or diagnosis confirmation whichever is earlier. You can adjust it with a more rigorous rationale. If all cases are left, (especially on the rising pandemics) the mortality is expected to be underestimated
    df['age'] = 2020 - df['birth_year']       # in case the dataset has full birth dates instead of birth years, the data type for this variable was left float64/double both in numpy/pandas and the shared C library
    df['outcome'] = np.where(df.state == 'deceased', 1, 0)
    df['age'] = np.where(df.age == 0.0, 1e-8, df.age)
    return df[['age','outcome']]


def main():
    # the function below is designed to import the case-by-case table from URL https://www.kaggle.com/kimjihoo/coronavirusdataset#PatientInfo.csv
    df = ingestData('PatientInfo.csv')
    bestFunction = deathcurve.fitFunctionWrapper(df, '-', False, functions = (0,))   # when passing custom 'functions' parameter consisting of one integer, please remember to add a comma after it to count for a tuple: (1,) instead of (1)
    bestFunction.reportModel()
    bestFunction.plotModel()
    
    
if __name__ == '__main__':
    main()
