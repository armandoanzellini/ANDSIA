# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:37:03 2019

@author: Armando

This file processes data for isotope ratio results from Raman (RA) and 
FTIR (IR) when compared to GasBench IRMS (MS)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

file = 'ANDSIA_IsotopeResults.xlsx'

data = pd.read_excel(file, index_col=0)

msc = data['MS d13C']
mso = data['MS d18O']
rac = data['RA d13C']
rao = data['RA d18O']
irc = data['IR d13C']
iro = data['IR d18O']
dup = data['Duplicate'][:-11].dropna()

# Test normality
msc_sw = stats.shapiro(msc[:-11])
mso_sw = stats.shapiro(mso[:-11])

# Not normally distributed
# Now need non-parametric tests for replicability within MS data

dup_array = []

for i in range(len(dup)):
    val = [dup.index[i], dup[i]]
    dup_array.append(val)
    
# subtract differences and create array to use Wilcoxon on it

# MSC Wilcoxon
cdiff = []

for i in dup_array:
    val = msc[i][0] - msc[i][1]
    cdiff.append(val)

cwil = stats.wilcoxon(cdiff)

abscdiff = []
for i in dup_array:
    val = abs(msc[i][0] - msc[i][1])
    abscdiff.append(val)

cerr = np.average(abscdiff)

# MSO Wilcoxon
odiff = []

for i in dup_array:
    val = mso[i][0] - mso[i][1]
    odiff.append(val)

owil = stats.wilcoxon(odiff)

absodiff = []
for i in dup_array:
    val = abs(mso[i][0] - mso[i][1])
    absodiff.append(val)

oerr = np.average(absodiff)

# W results: C p=0.533, O p=0.722
# Do not reject null, no difference between the groups

# Define a line of best fit function
def best_fit(xs,ys):
    xs = np.array(xs)
    ys = np.array(ys)
    
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b

def regression_line(m, b, x_array):
    line = [(m*x)+b for x in x_array]
    
    return line

# Line plot and regression to test that approach to statsitical comparison IRMS
xc = []
yc = []

for i in dup_array:
    xc.append(msc[i][0])
    yc.append(msc[i][1])
    
mc, bc = best_fit(xc, yc)
linec  = regression_line(mc, bc, xc)

plt.scatter(xc, yc)
plt.plot(xc, linec)
plt.title('IRMS Duplication d13C')
plt.show()

xo = []
yo = []

for i in dup_array:
    xo.append(mso[i][0])
    yo.append(mso[i][1])
    
mo, bo = best_fit(xo, yo)
lineo  = regression_line(mo, bo, xo)

plt.scatter(xo, yo)
plt.plot(xo, lineo)
plt.title('IRMS Duplication d18O')
plt.show()

# Calculate TEM
d2c = []

for i in cdiff:
    d2c.append(i*i)
    
tem_c = np.sqrt(sum(d2c)/22)
r_c   = 1- (tem_c**2)/(np.std(xc+yc)**2)
    
d2o = []
for i in cdiff:
    d2o.append(i*i)

tem_o = np.sqrt(sum(d2o)/22)
r_o   = 1- (tem_o**2)/(np.std(xo+yo)**2)

# Results no different than Wilcoxon or error calculations

# Begin Analysis of other methods
# Define Wilcoxon and error function
def comparison(x, y):
    '''Make sure the inputs are of equal length'''
    if len(x) != len(y):
        raise Exception('Inputs not of equal length')
    
    diff = []

    for i in range(len(x)):
        val = x[i] - y[i]
        diff.append(val)

    wil = stats.wilcoxon(diff)

    absdiff = []
    for i in range(len(x)):
       dval = abs(x[i] - y[i])
       absdiff.append(dval)

    err = np.average(absdiff)
    
    return diff, absdiff, wil, err

# Compare MS to RA
ramsdiffc, ramsabsdiffc, ramswilc, ramserrc = comparison(msc, rac)
ramsdiffo, ramsabsdiffo, ramswilo, ramserro = comparison(mso, rao)

# Compare MS to IR
irmsdiffc, irmsabsdiffc, irmswilc, irmserrc = comparison(msc, irc)
irmsdiffo, irmsabsdiffo, irmswilo, irmserro = comparison(mso, iro)

