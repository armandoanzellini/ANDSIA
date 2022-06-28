# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:10:40 2022

@author: Armando Anzellini

Read CSV data from a single Raman spectrum of bone and will calculate isotope
ratios of 13C, 15N, and 18O (from both phosphate and carbonate) either OPLSR, 
Bayes B, or Bayesian Ridge Regression. It also calculates the crystallinity 
index (CI), AmidI/PO4 ratio, and CH/PO4 ratio to assess diagenesis.

Citation: available once published
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyopls import OPLS

isotope   = '13C' 

direct    = "C:\\Users\\aanzellini\\OneDrive\\Documents\\Academic\\Dissertation\\"

project   = "Patakfalva-Papdomb" + "\\"

specfile  = ".csv"

isotopedb = "IsoSpecDB.csv" # spectra and IRMS results used in training model

spectrum = pd.read_csv(direct + project + specfile)
irisdb   = pd.read_csv(direct + project + isotopedb)

# fit OPLS model
opls = OPLS(39)

Z = opls.fit(isotopedb[:-3], isotopedb[isotope])






































# Plot output
fig = plt.figure(figsize=(4,6))
gs  = gridspec.GridSpec(2,1, height_ratios=[1,1])
ax1 = fig.add_subplot(gs[0])

plt.show()
