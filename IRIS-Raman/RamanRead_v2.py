# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:43:17 2024

Reading CSV files from BWTek iRaman Pro 1064 with full output including
data preclean


@author: Armando Anzellini
"""

import os
import re
import numpy as np
import scipy as sp
import pandas as pd
import SpecCalcMod as scm # make sure the cwd is the correct one
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

#path = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Data\\ANDSIA_1064_Power71\\"
path = "C:\\Users\\ara622\\OneDrive\\Documents\\Academic\\Data\\ANDSIA_1064_Power71\\"

file = "ANDSIA_1.csv"

#read metadata bu specify only 2 columns so it doesn't get hung up on the colon in lines 67 and 68
metadata = pd.read_csv(path + file, skipfooter=513, usecols=range(2), header=None)

# read signal data from csv with all columns even if empty
df = pd.read_csv(path + file, skiprows=98)

df.plot("Raman Shift", "Dark Subtracted #1")

# clean spec by removing Raman shifts in negatives
df = df[df['Raman Shift'].between(250, 1800)]

# als baseline correction
_, spectra_arPLS, info = scm.baseline_arPLS(df['Raw data #1'], lam=1e4, niter=100, full_output=True)

plt.plot(spectra_arPLS)

# poly baseline on dark correct data
# find first three minima since that's what we see in the dark corrected data
mins = sp.signal.argrelextrema(df['Dark Subtracted #1'].values, np.less)[0][:4]

mins = np.append(0, mins) # add the first point, important for fit but not in func

df.plot("Raman Shift", "Dark Subtracted #1")
plt.scatter(df.iloc[mins]['Raman Shift'], df.iloc[mins]["Dark Subtracted #1"])

# Fit a spline to three minima and first point to correct upwards low raman shift #
x = df.iloc[mins]['Raman Shift']
y = df.iloc[mins]["Dark Subtracted #1"]

fit_x = df[df['Raman Shift'] < max(x)]['Raman Shift']

spl = sp.interpolate.UnivariateSpline(x, y)

# plot for example on baseline correction if needed in publication
# df.plot("Raman Shift", "Dark Subtracted #1")
# plt.plot(fit_x, spl(fit_x))
# plt.scatter(df.iloc[mins]['Raman Shift'], df.iloc[mins]["Dark Subtracted #1"])

# get the subtractive values into a pd array with corresponding x values
baseline_y = spl(fit_x)

base_y = np.append(baseline_y, [0]*(len(df)-len(fit_x)))

baseline = pd.DataFrame((df['Raman Shift'], baseline_y))

# add baseline values to analytic df
d = {'Raman Shift': df['Raman Shift'], 'Dark Subtracted' : df['Dark Subtracted #1'], 'Baseline' : base_y}

ramdf = pd.DataFrame(d)

#ramdf.plot('Raman Shift', ['Dark Subtracted', 'Baseline'])

ramdf['AdjIntensity'] = ramdf['Dark Subtracted'] - ramdf['Baseline']

ramdf.plot('Raman Shift', 'AdjIntensity')

# normalize to peak intensity of PO4 peak



