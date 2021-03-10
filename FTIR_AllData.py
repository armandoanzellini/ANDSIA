# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:18:39 2021

@author: Armando
"""
import os
import glob
import pandas as pd
import scipy.signal as ss
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

direct = 'C:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\Preliminary Study\\FTIR Results'

os.chdir(direct)

# Get list of files from directory and compile a list of only csv files
files = glob.glob('*.csv')

# Compile the data of each file into a pandas dataframe
data = []
for i in files:
    sampname = 'P' + ''.join([x for x in i if x.isdigit()])
    samp     = pd.read_csv(i ,names = ['wavelength', sampname], header=None,skiprows=2,index_col=False)
         
    data.append(samp)

df = pd.concat(data, axis = 1)

df = df.loc[:,~df.columns.duplicated()]

# Define X and Y to plot all spectra
X = 'wavelength'
#Y = df.columns.to_list()[1:]

for i in df.columns.to_list()[1:]:
    Y = i
    fig, ax = plt.subplots()
    ax.plot(df[X], df[Y])
    ax.invert_xaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(i)
    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)')
    ax.set_ylabel(r'Absorbance')

    plt.show()

# Do SG 2nd derivative and then plot again to see if they normalize
sgdf = df['wavelength'].to_frame()

for i in df.columns[1:]:
    sg      = ss.savgol_filter(df[i], window_length=9, polyorder=2, deriv=2)
    sgdf[i] = -sg
    
fig, ax = plt.subplots()
ax.plot(sgdf[X], sgdf[Y])
ax.invert_xaxis()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'Wavenumber ($cm^{-1}$)')
ax.set_ylabel(r'$-$A$^{\prime \prime}\/(\nu)$')
plt.show()

# Normalize data and plot again
norm = df['wavelength'].to_frame()
    
for i in df.columns[1:]:
    normspec        = df[i]/df[i].max()
    norm[i] = normspec
    
fig, ax = plt.subplots()
ax.plot(norm[X], norm[Y])
ax.invert_xaxis()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'Wavenumber ($cm^{-1}$)')
ax.set_ylabel(r'Normalized Absorbance')
plt.show()


# Now plot SG 2nd derivative of normalized data
sgnorm = df['wavelength'].to_frame()

for i in norm.columns[1:]:
    sg      = ss.savgol_filter(norm[i], window_length=9, polyorder=2, deriv=1)
    sgnorm[i] = -sg

fig, ax = plt.subplots()
ax.plot(sgnorm[X], sgnorm[Y])
ax.invert_xaxis()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'Wavenumber ($cm^{-1}$)')
ax.set_ylabel(r'Normalized $-$A$^{\prime \prime}\/(\nu)$')
plt.show()

# Run Bayesian PCA
