# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:59:20 2020

@author: Armando Anzellini

This program reads the FTIR peak data and correlates the values to the
real delta and ratio values to get a correlation function
"""
import os
import pymc3 as pm
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

components = 4

# List of samples to remove since noisy (see AllData file with normalized plots)
noise_list = [1, 3, 5, 6, 8, 10, 15, 17, 21, 22, 26, 28, 29, 36, 37, 38, 39, 44, 47, 48]

# To be used only when testing without single peak examples
''' 
noise_list += [2, 9, 12, 14, 23, 26, 27, 31, 32, 33, 50]
''' 

direct = 'D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\Preliminary Study\\'

file = 'ANDSIA_FTIR_Results_2Prime no corr.xlsx'
fname = file.strip('.xlsx')

os.chdir(direct)

raw_df = pd.read_excel(file, sheet_name = 'Peaks')

# If there are values in noise_list, remove them, else skips this step
nl = []
for i in noise_list:
    nl += ["P" + str(i)]
    
if nl:
    raw_df = raw_df[~raw_df['Sample ID'].isin(nl)]

# Drop all rows that have NAN for any of the peak values
df = raw_df.dropna(subset = ['Apeak', 'Bpeak', 'Ashift', 'Bshift'])

# Create 3-D plot of expected predictive ratios
'''
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(df['A3A12'], df['A4A12'], df['DeltaC'])

plt.show()
'''
# Create function to label individual points in a scatter plot to use below
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

# Plot 2-D figure of A34A12 for reference
'''
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(df['A34A12'], df['DeltaC'])

label_point(df['A34A12'], df['DeltaC'], df['Sample ID'], ax)

plt.show()
'''
# MR Model using A3A12 and A4A12 (scikit-learn)
X = df[['A3A12', 'A4A12']]
Y = df['DeltaC']

reg = LinearRegression()

mr_model  = reg.fit(X, Y)

# Begin results df
res = pd.DataFrame(df[['Sample ID', 'DeltaC']])

# Predict results using the MR model and convert to delta values
predmr  = mr_model.predict(X)
predmr.shape = (len(predmr),)
predmr = pd.Series(predmr, index = res.index)

#preddel = predmr.apply(del13C)
preddel = predmr

preddel.name = 'MR_Delta'

# Add MR results to results df
res = res.join(preddel)

# t-test comparing the MR results
obs    = res['DeltaC']
mrpred = res['MR_Delta']

print('MR vs. IRMS t-test: ')
print(sp.stats.ttest_1samp(obs - mrpred, popmean=0))

# Bland-Altman plot for the MR results
f, ax = plt.subplots(1, figsize = (6,4))
sm.graphics.mean_diff_plot(obs, mrpred, ax = ax, 
                           scatter_kwds={'color' :'k', 'alpha': 0.6})
ax.set_title(r'Bland-Altman Plot: Multiple Regression vs. IRMS $\delta^{13}$C')
plt.show()

# Try PLS using all four peak values and reducing
Y = df['DeltaC'].to_numpy()
X = df[['Apeak', 'Bpeak', 'Ashift', 'Bshift']].to_numpy()

pls = PLSRegression(n_components = components)

pls_model = pls.fit(X, Y)

def pls(a, b, c, d):
    '''
    T: x_scores_
    U: y_scores_
    W: x_weights_
    C: y_weights_
    P: x_loadings_
    Q: y_loadings__
    
    X = T P.T + Err and Y = U Q.T + Err
    
    X = pls_model.x_scores_

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    i = pls_model.coef_[0][0] * a
    j = pls_model.coef_[1][0] * b
    k = pls_model.coef_[2][0] * c
    l = pls_model.coef_[3][0] * d
    
    return i + j + k + l

i = 0
pls(df['Apeak'].iloc[i], df['Bpeak'].iloc[i], df['Ashift'].iloc[i], df['Bshift'].iloc[i])

# Predict results using the PLS model
predpls = pls_model.predict(X)
predpls.shape = (len(predpls),)
predpls = pd.Series(predpls, index = res.index)

#preddel = predpls.apply(del13C)
preddel = predpls

preddel.name = 'PLS_Delta'

# Add PLS results to results df
res = res.join(preddel)

# t-test comparing the PLS results
plspred = res['PLS_Delta']

print('PLS vs. IRMS t-test: ')
print(sp.stats.ttest_1samp(obs - plspred, popmean=0))

# Bland-Altman plot for the PLS results
f, ax = plt.subplots(1, figsize = (6,4))
sm.graphics.mean_diff_plot(obs, plspred, ax = ax, 
                           scatter_kwds={'color' :'k', 'alpha': 0.6})
ax.set_title(r'Bland-Altman Plot: Partial Least Square vs. IRMS $\delta^{13}$C')
plt.show()

# Try Bayesian Linear Regression
# first create R-style Patsy formulas to pass into model
#formula = 'Y ~ ' + ' + '.join(['Apeak', 'Bpeak', 'Ashift', 'Bshift'])
#formula = 'Y ~ 0 + ' + ' + '.join(['A3A12', 'A4A12'])
formula = 'Y ~ ' + ' + '.join(['A3A12', 'A4A12'])

# Define data to be used in the model
X = df[['A3A12', 'A4A12']]
Y = df['DeltaC']

# Context for the Bayesian model
with pm.Model() as model:
    # set distribution for priors
   
    priors = {'A3A12':     pm.InverseGamma.dist(mu=0.01, sigma=1.0),
              'A4A12':     pm.Normal.dist(mu=1.0),
              'Intercept': pm.Normal.dist(mu=1.0)}
    
    
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data
    pm.GLM.from_formula(formula, data = X, family = family) #priors = priors)
    
    # Perform Markov Chain Monte Carlo sampling (with freeze_support to exec file)
    trace = pm.sample(draws=4000, cores = 1, tune = 1000)

# Plot the results of MCMC and GLM
pm.traceplot(trace)
plt.show()

for variable in trace.varnames:
    print('Variable: {:15} Mean weight in model: {:.8e}'.format(variable, 
                                                                np.mean(trace[variable])))

# Define a function that gives the estimated value using the mean weights
# Fix function to give a more consistent result for 95% CI!!!!!!!!!
def bayes_predict(trace, test_observation, plot = True):
    
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)
    
    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()

   
    # Add in intercept term
    test_observation['Intercept'] = 1
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # Estimates of grade
    estimates = np.random.normal(loc = mean_loc, scale = sd_value,
                                 size = 1000)

    # Plot all the estimates if plot is set to True
    if plot == True:
        fig, ax = plt.subplots(1, figsize = (8, 8))
        sns.histplot(estimates, bins = 19,kde = True, edgecolor = 'k',
                     color = 'darkblue', line_kws = {'linewidth' : 4},
                     label = 'Estimated Dist.')
    
        # Plot the mean estimate
        ax.axvline(x = mean_loc, ymin = 0,
                   linestyle = '--', 
                   color = 'orange',
                   label = 'Mean Estimate',
                   linewidth = 2.5)
    
        ax.set_title('Density Plot for Test Observation')
        ax.legend(loc = 1)
        ax.set_xlabel('Grade')
        ax.set_ylabel('Density')
    
    # Prediction information
    print('Average Estimate = %e' % mean_loc)
    print('5%% Estimate = %e    95%% Estimate = %e' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))
    
    return mean_loc
    
# get all predicted values using function defined above
predbys = []
for i in range(len(df[['A3A12', 'A4A12']])):
    predbys += [bayes_predict(trace, df[['A3A12', 'A4A12']].iloc[i], plot = False)]
    
predbys = pd.Series(predbys, index = res.index)

#preddel = predbys.apply(del13C)
preddel = predbys

preddel.name = 'BYS_Delta'

# Add Bayes results to results df
res = res.join(preddel)

# t-test comparing the PLS results
byspred = res['BYS_Delta']

print('BYS vs. IRMS t-test: ')
print(sp.stats.ttest_1samp(obs - byspred, popmean=0))

# Bland-Altman plot for the Bayes results
f, ax = plt.subplots(1, figsize = (6,4))
sm.graphics.mean_diff_plot(obs, byspred, ax = ax, 
                           scatter_kwds={'color' :'k', 'alpha': 0.6})
ax.set_title(r'Bland-Altman Plot: Bayesian Regression vs. IRMS $\delta^{13}$C')
plt.show()

# Check for duplication values using the duplicate DeltaC values from raw_df
dup_df = pd.read_excel(file, sheet_name = 'Duplicates')

dups   = dup_df[['Duplicate', 'DeltaC']].rename(columns = 
                                                {'Duplicate': 'Sample ID', 
                                                 'DeltaC'   : 'DupDelta'})
orig   = raw_df[['Sample ID', 'DeltaC']]

orig   = orig[orig['Sample ID'].isin(dups['Sample ID'])]

dupd   = orig.merge(dups, on = 'Sample ID')

# Test duplication of results
obs_dup = dupd['DeltaC']
dup     = dupd['DupDelta']

print('Duplicate DeltaC t-test: ')
print(sp.stats.ttest_1samp(obs_dup - dup, popmean=0))

# Bland-Altman plot for the duplication results
f, ax = plt.subplots(1, figsize = (6,4))
sm.graphics.mean_diff_plot(obs_dup, dup, ax = ax, 
                           scatter_kwds={'color' :'k', 'alpha': 0.6})
ax.set_title(r'Bland-Altman Plot: Duplication of IRMS $\delta^{13}$C')
plt.show()

# Plot the duplication results (make line of pearson correlation coefficient?)
x = np.linspace(dupd['DeltaC'].min(), dupd['DeltaC'].max(), 50)
y = x

fig, ax = plt.subplots(figsize = [4, 4])
ax.scatter(dupd['DeltaC'], dupd['DupDelta'], color = 'k', alpha = 0.7)
ax.plot(x, y , 'k--')
'''
for row in red.iterrows():
    label = df.loc[row[0]]['Sample ID']
    x     = red.loc[row[0]]['PLS_Delta']
    y     = red.loc[row[0]]['DeltaC']
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
'''
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'Duplicated $\delta^{13}$C')
ax.set_ylabel(r'Original $\delta^{13}$C')
plt.show()
plt.show()

# Print coefficient mean values for the MR and Bayesian GLM results
print(mr_model.coef_)
print(mr_model.intercept_)

for variable in trace.varnames:
    print('Variable: {:15} Mean weight in model: {:.8e}'.format(variable, 
                                                                np.mean(trace[variable])))

# Check for goodness of fit for each of the models through RMSE and real STD
delta_sd = res['DeltaC'].std()
dup_rmse = np.sqrt(metrics.mean_squared_error(dupd['DeltaC'], dupd['DupDelta']))
mr_rmse  = np.sqrt(metrics.mean_squared_error(res['DeltaC'], res['MR_Delta']))
pls_rmse = np.sqrt(metrics.mean_squared_error(res['DeltaC'], res['PLS_Delta']))
bys_rmse = np.sqrt(metrics.mean_squared_error(res['DeltaC'], res['BYS_Delta']))


print( '\n', 'STD: ', delta_sd, '\n',
             'DUP: ', dup_rmse, '\n',
             ' MR: ',  mr_rmse, '\n',
             'BYS: ', bys_rmse, '\n',
             'PLS: ', pls_rmse, '\n')

# Plot scatter without outliers after running analysis with them (reduced = red)
rem = [37, 38, 39]

red = res[~(res['DeltaC'] < -5.8)]

delta_sd = red['DeltaC'].std()
dup_rmse = np.sqrt(metrics.mean_squared_error(dupd['DeltaC'], dupd['DupDelta']))
mr_rmse  = np.sqrt(metrics.mean_squared_error(red['DeltaC'], red['MR_Delta']))
pls_rmse = np.sqrt(metrics.mean_squared_error(red['DeltaC'], red['PLS_Delta']))
bys_rmse = np.sqrt(metrics.mean_squared_error(red['DeltaC'], red['BYS_Delta']))

print( '\n', 'STD: ', delta_sd, '\n',
             'DUP: ', dup_rmse, '\n',
             ' MR: ',  mr_rmse, '\n',
             'BYS: ', bys_rmse, '\n',
             'PLS: ', pls_rmse, '\n')


# find R2 for the expected function to the actual values
SSR = sum((red['DeltaC'] - red['PLS_Delta'])**2)
SST = sum((red['DeltaC'] - np.mean(red['DeltaC']))**2)
R2  = 1 - (SSR/SST)

# Create a line and plot all together for PLS
x_corr = np.linspace(-6, -4, 100)
y_corr = x_corr

band1_y = y_corr + 0.6 # create bands that collect 0.6 per mil interpretive differences
band2_y = y_corr - 0.6

fig, ax = plt.subplots(figsize = [6, 4])
ax.set_title("IRViS $\delta^{13}$C Results", family="serif")
ax.scatter(red['PLS_Delta'], red['DeltaC'], color = 'k', alpha = 0.7)
ax.plot(x_corr, y_corr , 'k--')
ax.plot(x_corr, band1_y, 'k:')
ax.plot(x_corr, band2_y, 'k:')
'''
for row in red.iterrows():
    label = df.loc[row[0]]['Sample ID']
    x     = red.loc[row[0]]['PLS_Delta']
    y     = red.loc[row[0]]['DeltaC']
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
'''
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlim([-5.5, -4])
ax.set_ylim([-5.5, -4])
ax.set_xlabel(r'IRViS $\delta^{13}$C', family='serif')
ax.set_ylabel(r'IRMS $\delta^{13}$C', family = 'serif')

fig.savefig(direct + 'PLS_results_wBands.png', bbox_inches='tight', dpi = 300 )
plt.show()

# Create a line and plot all together for BYS
x_corr = np.linspace(red['BYS_Delta'].min(), red['BYS_Delta'].max(), 50)
y_corr = x_corr

fig, ax = plt.subplots(figsize = [4, 4])
ax.scatter(red['BYS_Delta'], red['DeltaC'], color = 'k', alpha = 0.7)
ax.plot(x_corr, y_corr , 'k--')
'''
for row in red.iterrows():
    label = df.loc[row[0]]['Sample ID']
    x     = red.loc[row[0]]['PLS_Delta']
    y     = red.loc[row[0]]['DeltaC']
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
'''
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'BYS Results $\delta^{13}$C')
ax.set_ylabel(r'Original $\delta^{13}$C')
plt.show()

# Create a line and plot all together for MR
x_corr = np.linspace(red['MR_Delta'].min(), red['MR_Delta'].max(), 50)
y_corr = x_corr

fig, ax = plt.subplots(figsize = [4, 4])
ax.scatter(red['MR_Delta'], red['DeltaC'], color = 'k', alpha = 0.7)
ax.plot(x_corr, y_corr , 'k--')
'''
for row in red.iterrows():
    label = df.loc[row[0]]['Sample ID']
    x     = red.loc[row[0]]['PLS_Delta']
    y     = red.loc[row[0]]['DeltaC']
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
'''
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlabel(r'MR Results $\delta^{13}$C')
ax.set_ylabel(r'Original $\delta^{13}$C')
plt.show()

# Now try the duplicates for each method
if nl:
    dup_df = dup_df[~dup_df['Sample ID'].isin(nl)]
    
if nl:
    dup_df = dup_df[~dup_df['Duplicate'].isin(nl)]

pkdup = dup_df[['Duplicate', 'Apeak', 'Bpeak', 'Ashift', 
                'Bshift', 'A3A12', 'A4A12']].dropna()

# Create results DataFrame
dupres = pkdup['Duplicate'].to_frame(name = 'Sample ID')

dupres = pd.merge(dupres, dupd, how = 'inner', on='Sample ID')

# Use the MR, PLS, and BYS predictors to predict DeltaC values of duplicates
# Attempt keeps the spectra with some peaks being zero like above
# Maybe try only with two peaks separate from one peak with more data
X = pkdup[['A3A12', 'A4A12']]

# Predict results using the MR model
dupmr  = mr_model.predict(X)
dupmr.shape = (len(dupmr),)
dupmr = pd.Series(dupmr, index = dupres.index)

dupmr.name = 'MR_Dup'

# Add MR results to results df
dupres = dupres.join(dupmr)

# Predict Bayes results
dupbys = []
for i in range(len(X)):
    dupbys += [bayes_predict(trace, X.iloc[i], plot = False)]
    
dupbys = pd.Series(dupbys, index = dupres.index)

dupbys.name = 'BYS_Dup'

# Add Bayes results to results df
dupres = dupres.join(dupbys)

# Predict results using the PLS model
X = pkdup[['Apeak', 'Bpeak', 'Ashift', 'Bshift']]

duppls = pls_model.predict(X)
duppls.shape = (len(duppls),)
duppls = pd.Series(duppls, index = dupres.index)

duppls.name = 'PLS_Dup'

# Add PLS results to results df
dupres = dupres.join(duppls)

allvals = pd.merge(dupres, res, how = 'inner', on = 'Sample ID')

allvals = allvals.drop(columns='DeltaC_x')

res.to_excel('DeltaVals.xlsx')
allvals.to_excel('Duplication.xlsx')