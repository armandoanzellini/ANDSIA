# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:00:47 2021

@author: Armanco Anzellini

This code reads CSV data from a single RAMAN spectrum of bone and completes
post-processing on the spectrum to reveal underlying data by removing the
fluorescence.
"""
import os
import re
import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy import sparse
from numpy.linalg import norm
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from pyspectra.readers.read_spc import read_spc_dir
from matplotlib.ticker import AutoMinorLocator

direct = 'C:\\Users\\aanzellini\\OneDrive\\Documents\\Academic\\Dissertation\\Patakfalva-Papdomb\\'

filedir = 'Mand Method Test\\'

samp    = 'Mand'

os.chdir(direct)

# Read the spc files from the directory
df_spc, dict_spc = read_spc_dir(direct + filedir)

# transpose df and add index
spc_df = df_spc.transpose()

# Select only rows that will provide valueable data
spc_df = spc_df.loc[(spc_df.index > 100) & (spc_df.index < 3295)]

# Normalize all spectra to 0
# find minimum for each spectrum and subtract to bring to zero
df = spc_df.apply(lambda x: x-x.min())

# Check that all scans have the same name when the number is removed
for i in range(len(df.columns)-1):
    col  = re.search(r'.+?(?=\d{1,2}.spc)', df.columns[i])[0]
    ncol = re.search(r'.+?(?=\d{1,2}.spc)', df.columns[i+1])[0]
    if ncol != col:
        print('Error in Scan Names, Check All Scans Are Equivalent')

# Change the column names from full values to just scan number
col_dict = {}

for column in df.columns:
    number           = re.findall(r'(\d{1,2}).spc', column)[0]
    col_dict[column] = int(number) # create dictionary for column change
    
df = df.rename(columns = col_dict) # rename columns to integers

df = df.reindex(sorted(df.columns), axis=1) # sort columns in ascending order

# Limit wavenumbers to expected values for bone
df = df[df.index > 700]

# Plot to see all scans together
f, ax =plt.subplots(1, figsize=(15,5))
ax.plot(df)
ax.grid()
ax.set_ylabel('Signal')
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_xlim([700, 3250])
plt.show()

# Average data together to see what comes out
df['mean'] = df.mean(axis=1)

# Create subplot
fig, ax = plt.subplots(1, figsize= [15, 5])

# Spectral plot
ax.plot(df['mean'], '-', color = 'k', linewidth=0.6)

# Fix axis direction, set title, add labels
ax.set_xlim([700, 3250])
ax.set_title('Average of all scans: ' + samp)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
ax.set_ylabel("Absorbance",family="serif",  fontsize=12)

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()

# Use arPLS (https://doi.org/10.1039/C4AN01061B) to remove fluorescence
def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z

_, spectra_arPLS, info = baseline_arPLS(df['mean'], lam=1e4, niter=100,
                                         full_output=True)

# add corrected spectrum to df
spectra_arPLS.name = 'arPLS'

df = df.join(spectra_arPLS)

# Plot baseline corrected spectra
# Create subplot
fig, ax = plt.subplots(1, figsize= [15, 5])

# Spectral plot
ax.plot(df['arPLS'], '-', color = 'k', linewidth=0.6)

# Fix axis direction, set title, add labels
ax.set_xlim([700, 3250])
ax.set_title('arPLS Corrected Spectrum: ' + samp)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
ax.set_ylabel("Absorbance",family="serif",  fontsize=12)

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()

# try SG smoothing
df['PLSsmooth'] = ss.savgol_filter(df['arPLS'], window_length=11, 
                                            polyorder=5, deriv=0)

# Plot smoothed baseline corrected spectra
# Create subplot
fig, ax = plt.subplots(1, figsize= [15, 5])

# Spectral plot
ax.plot(df['PLSsmooth'], '-', color = 'k', linewidth=0.6)

# Fix axis direction, set title, add labels
ax.set_xlim([700, 3250])
ax.set_title('Smoothed arPLS Corrected Spectrum: ' + samp)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
ax.set_ylabel("Absorbance",family="serif",  fontsize=12)

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()

# try a non arPLS baseline, instead a baseline following minima
minima = ss.find_peaks(-df['mean'], distance=37)

baseline = np.interp(df.index, df.iloc[minima[0]].index, df.iloc[minima[0]]['mean'])

corr_bl = baseline - (max(baseline - df['mean']))

for i,val in enumerate(corr_bl):
    if val < 0:
        corr_bl[i] = 0.0

# add new column to df with new baseline values and smooth
df['corr_signal'] = df['mean'] - corr_bl
df['BLsmooth'] = ss.savgol_filter(df['corr_signal'], window_length=11, 
                                            polyorder=5, deriv=0)

# plot arPLS and Baseline together to look for differences
# Create subplot
fig, ax = plt.subplots(1, figsize= [15, 5])

# Spectral plot
ax.plot(df['PLSsmooth'], '-', color = 'k', linewidth=0.6)
ax.plot(df['BLsmooth'], '--', color = 'k', linewidth=0.6)

# Fix axis direction, set title, add labels
ax.set_xlim([700, 3250])
ax.set_title('Baseline to arPLS Comparison: ' + samp)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
ax.set_ylabel("Absorbance",family="serif",  fontsize=12)

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.show()



