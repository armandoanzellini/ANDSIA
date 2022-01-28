# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:32:02 2021

@author: Armando Anzellini
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def lin(x,m,b):
    return (m*x) + b

X    = np.linspace(-13, -3, 100)
line = [lin(x, 1, 0) for x in X]

noise = np.random.normal(0, 0.6, 100)

signal = line + noise

rmse = np.sqrt(metrics.mean_squared_error(signal, X))

print(rmse)

# Make the data and X a pd df
df = pd.DataFrame({'X' : X, 'S': signal})

# Get sample that is similar to my own data
samp = df[df['X'] > -5.8]

samp_rmse = np.sqrt(metrics.mean_squared_error(samp['X'], samp['S']))

print(samp_rmse)

# Plot Data with inset to show what's happening in my data
# Create subplot
fig, ax = plt.subplots(figsize=[5, 4])

# Scatter plot
ax.scatter(df['X'], df['S'], color = 'k', alpha = 0.7)
ax.plot(X, line, '--', color = 'k')

# inset axes....
axins = ax.inset_axes([0.55, 0.05, 0.4, 0.4])
axins.scatter(samp['X'], samp['S'], color='k', alpha = 0.7)
axins.plot(X, line, '--', color = 'k')

# sub region of the original image
x1, x2, y1, y2 = samp['X'].min(), samp['X'].max(), samp['S'].min(), samp['S'].max(),
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(axis='both', which='major', length = 1, labelsize= 'x-small')

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

axins.xaxis.set_minor_locator(AutoMinorLocator())
axins.yaxis.set_minor_locator(AutoMinorLocator())

ax.set_xlabel(r'Spectrometric Approximation $\delta^{13}$C')
ax.set_ylabel(r'Original $\delta^{13}$C')

ax.indicate_inset_zoom(axins)

plt.show()