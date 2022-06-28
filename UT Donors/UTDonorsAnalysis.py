# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:47:46 2022

@author: Armando Anzellini

Analysis of the skeleltal remains representing indivdiuals from the 
university of Tennessee skeletal collection. Analyses include the estimation
of isotopic values using Raman spetroscopy and the linear regression model
developed from individuals from the Patakfalva-Papdomb site. It then compares 
intraskeletal to interindividual isotopic variation 
This script outputs plots, txt files, and excel files with all resulting data.

All analyses in this script have been presented in doi: 
Ph.D. Dissertation by Armando Anzellini
https://trace.tennessee.edu/utk_graddiss/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from tqdm import tqdm
from scipy import stats
from scipy import signal
from scipy import sparse
from joblib import load
from numpy.linalg import norm
from scipy.sparse import linalg
from matplotlib.ticker import AutoMinorLocator
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# define directory paths
direct = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\"

samdir = "UT Collection\\"

donors = "Donors Raman\\"

dups   = donors + "Duplicates\\"

# load linear regression models
d13cmod = load(direct + 'Models\\' + 'd13c_model.joblib')
d18omod = load(direct + 'Models\\' + 'd18o_model.joblib')

# open non-raman data
donordata = pd.read_excel(direct + samdir + "DonorResidence18OReducedPrioritized_UTID.xlsx")

# clean up data making UTID the index and dropping unnecessary columns
donordata.dropna(inplace=True)

dropcols = ['KEY', 'Country', 'Lat', 'Long'] # columns not necessary

donordata.drop(dropcols, axis = 1, inplace=True)

donordata['UTID'] = [x.lstrip('UT') for x in donordata['UTID']]

donordata.set_index('UTID', inplace=True) # set UTID as index for future relates

# Opening up the donor raman spectra
# now list the folders in the donors directory to open each file
ramdir = os.listdir(direct + samdir + donors)

ramdir.remove('Duplicates') # remove duplicates directory

files = os.listdir(direct + samdir + donors + ramdir[0]) # create list of filenames in donor folders

# open the first folder and read the txt files
df = pd.concat((pd.read_csv(direct + samdir + donors + ramdir[0] +'\\' + f, 
                            skiprows=2,
                            index_col=0,
                            names = ['wavelength', f.strip('.txt')]) \
                for f in files), 
               axis=1)

# add multilevel index
df.columns = pd.MultiIndex.from_product([[ramdir[0]], df.columns])

# use a for loop to append new Raman spectra for other donors
for donor in ramdir[1:]:
    # open the first folder and read the txt files
    tempdf = pd.concat((pd.read_csv(direct + samdir + donors + donor +'\\' + f, 
                                skiprows=2,
                                index_col=0,
                                names = ['wavelength', f.rstrip('.txt')]) \
                    for f in files), 
                   axis=1)
    
    # add multilevel index
    tempdf.columns = pd.MultiIndex.from_product([[donor], tempdf.columns])
    
    # concat to original dupdf
    df = pd.concat([df, tempdf], axis=1)
    
# Opening up the duplicated spectra and placing in own df
# now list the folders in the dups directory to open each file
dupdir = os.listdir(direct + samdir + dups)

dupfiles = os.listdir(direct + samdir + dups + dupdir[0]) # create list of filenames in donor folders

# open the first folder and read the txt files
dupdf = pd.concat((pd.read_csv(direct + samdir + dups + dupdir[0] +'\\' + f, 
                            skiprows=2,
                            index_col=0,
                            names = ['wavelength', f.lstrip('d').rstrip('.txt')]) \
                for f in dupfiles), 
               axis=1)

# add multilevel index
dupdf.columns = pd.MultiIndex.from_product([[dupdir[0].lstrip('Dup')], 
                                            dupdf.columns])

# use a for loop to append new Raman spectra for other duplicates
for dup in dupdir[1:]:
    # open the first folder and read the txt files
    tempdf = pd.concat((pd.read_csv(direct + samdir + dups + dup +'\\' + f, 
                                skiprows=2,
                                index_col=0,
                                names = ['wavelength', f.lstrip('d').rstrip('.txt')]) \
                    for f in dupfiles), 
                   axis=1)
    
    # add multilevel index
    tempdf.columns = pd.MultiIndex.from_product([[dup.lstrip('Dup')], 
                                                 tempdf.columns])
    
    # concat to original dupdf
    dupdf = pd.concat([dupdf, tempdf], axis=1)
    


# FILES LOADED; Analysis Begins
# cut spectra to ROI and process data to remove baseline
df = df.loc[775:1900]

# now define arPLS function to baseline correct
def arPLS_baseline(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    # Define function for arPLS baseline correction
    # Use arPLS (https://doi.org/10.1039/C4AN01061B) to remove fluorescence
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
            break
        
    # now smooth using SG
    sd = pd.Series(signal.savgol_filter(d, 
                                        window_length=11, 
                                        polyorder=5, 
                                        deriv=0), 
                   index = d.index)

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, sd, info
    else:
        return z

# baseline correct all spectra in df and dupdf
cordf = pd.concat(map(lambda col: arPLS_baseline(df[col], 
                                              ratio = 1e-6, 
                                              lam=1e4, 
                                              niter=5500, 
                                              full_output=True)[1].rename(col), 
                       tqdm(df.columns)), 
                   axis=1)



# check duplications for precision
dupdons = list(set([i[0] for i in dupdf.columns])) # list ID for duplicated donors

duporgs = df[dupdons] # get orginal data for duplicated donors

rho, p = stats.spearmanr()