# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:26:15 2022

@author: Armando Anzellini

Analysis of the skeleltal remains representing individuals from the 
Patakfalva-Papdomb site. Analyses include comparisons to previous apatite 
isotopic data from the site as well as the building of linear regresion models 
to estimate isotopic values from apatite using Raman spectrometry. 
This script outputs plots, joblib files, txt files, and excel files with all 
resulting data.

All analyses in this script have been presented in doi: 
Ph.D. Dissertation by Armando Anzellini
https://trace.tennessee.edu/utk_graddiss/
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy import signal
from scipy import sparse
from joblib import dump
from numpy.linalg import norm
from scipy.sparse import linalg
from itertools import combinations
from matplotlib.ticker import AutoMinorLocator, NullLocator
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split

# Define directories for each of the datasets to be opened

direct  = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\"

dfdir   = "Patakfalva-Papdomb\\" 

sampdir = "Patakfalva-Papdomb\\Samples Raman\\Selected\\"

dupdir  = "Patakfalva-Papdomb\\Samples Raman\\Duplicates\\"

figout  = direct + "Figures\\" 

prefile = 'PatakfalvaUnassociatedIsotope_Anzellini.xlsx'

# opening non-raman data
sampdf  = pd.read_excel(direct + dfdir + prefile, sheet_name = 'All')

sampdp  = pd.read_excel(direct + dfdir + prefile, sheet_name = 'Duplicates')

# drop unnecessary columns from both sample df and duplicates df
dropcols    = ['Weight', 
               'Minitube', 
               'Carb Sample', 
               'Solution Volume',
               'Batch #', 
               'Final Tube+Sample', 
               'Final Weight']

dupdropcols = dropcols + ['Absolute Error d13C', 
                          'Absolute Error d18Ovpdb',
                          'Absolute Error d18Ovsmow', 
                          'Absolute Error %CaCO3',
                          'Square Error d13C', 
                          'Square Error d138Ovpdb',
                          'Square Error d138Ovsmow', 
                          'Square Error %CaCO3']

sampdf.drop(dropcols,    axis = 1, inplace=True)
sampdp.drop(dupdropcols, axis = 1, inplace=True)

# make sample column the new index for both df and dup
sampdf.set_index('Sample', inplace = True)
sampdp.set_index('Sample', inplace = True)

# Finished with preliminary analyses


# Opening up the PP raman spectra; first originals then duplicates

# READING ORIGINAL SCANS
files = os.listdir(direct + sampdir) # create list of filenames in folder

# read all files into a single dataframe, column names are file names
df = pd.concat(map(lambda file: pd.read_csv(direct + sampdir + file, 
                                            skiprows=1, 
                                            header=0, 
                                            names=['wavelength', 
                                                   file.strip('.txt')]), 
                                                                       files),
                                                                      axis = 1)

# combine all wavelength columns and make wavelength column the index
df = df.loc[:,~df.columns.duplicated()].copy()

df.set_index('wavelength', inplace=True)

# cut df to fingerprint region of the spectrum (removing the first bit that 
#has too much fluorescence)
df = df.loc[775:1900]

# strip column names of integration and scan numbers
pattern = r'\(\d-\d{2}\)(\(\d\))?'

df.columns = (re.sub(pattern, '', col) for col in df.columns)

# Subtract minimum from each column to remove negative values
for col in df.columns:
    df[col] = df[col] - df[col].min()





# READING DUPLICATE SCANS
# now list the folders in the duplicates directory to open each file
dupfiles = os.listdir(direct + dupdir + 'good\\') # create list of filenames

# read all duplicate files into a single dataframe, column names are file names
dupdf = pd.concat(map(lambda file: pd.read_csv(direct + dupdir +'good\\'+ file, 
                                            skiprows=1, 
                                            header=0, 
                                            names=['wavelength', 
                                                   file.strip('.txt')]), 
                                                                     dupfiles),
                                                                      axis = 1)

# combine all wavelength columns and make wavelength column the index
dupdf = dupdf.loc[:,~dupdf.columns.duplicated()].copy()

dupdf.set_index('wavelength', inplace=True)

# cut df to fingerprint region of the spectrum (removing the first bit that 
# has too much fluorescence)
dupdf = dupdf.loc[775:1900]

# strip column names of integration and scan numbers
pattern = r'\(\d-\d{2}\)(\(\d\))?'

dupdf.columns = (re.sub(pattern, '.2', col) for col in dupdf.columns)

# Subtract minimum from each column to remove negative values
for col in dupdf.columns:
    dupdf[col] = dupdf[col] - dupdf[col].min()

'''
It seems to be more likely that the spectrum will be slightly different due to
noise than for the isotopic values to differ significantly when done with the
same lab. for this reason, duplicated scans considered 'good' will be joined 
with the selected scans that have a QC of good or great before any additional 
analyses.
'''
# since this analysis will actually combine the duplicates and the originals
# to create a test/train split, that is what needs to happen now
df = df.join(dupdf)




# CALCULATE BASELINES AND INDICES

# now define arPLS function to calculate fluoratio and to baseline correct
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

# calculate fluorescence ratio by taking value below peak and dividing
# by PO4 peak total height giving percentage of peak is fluorescence
def fluorindex(sr, samp, granularity=11, plot = False):
    # fit polynomial at given values
    # coef = p.polyfit(sr.index, sr[samp], granularity) # create coefficients
    # ffit = p.Polynomial(coef) # create polynomial function with given coefs
    # Find the amount that the baseline must be shifted to avoid signal
    # shiftval = max(ffit(sr.index) - sr[samp])
    
    # get baseline using arPLS function
    baseline = pd.Series(arPLS_baseline(sr[samp], 
                                        ratio = 1e-6, 
                                        lam=1e4, 
                                        niter=5500, 
                                        full_output=False), 
                         index = sr.index)

    # find peak height of PO4 peak for index calculations
    peaks = signal.find_peaks(sr[samp])
    
    # get peak that fits expected values
    peak = sr[samp].iloc[peaks[0]].loc[940:990]
    
    peakval = peak.max()
    
    # get value of polynomial at the given peak wavelength
    baseval = baseline.loc[peak.idxmax()]
    
    if plot:
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [15, 5])
        
        # Spectral plot
        ax.plot(sr[samp], '-', color = 'k', linewidth=0.6)
        
        # baseline plot
        ax.plot(baseline, ':', color = 'k', linewidth=0.6)
        
        # check baseline value is correct
        ax.scatter(peak.idxmax(), peakval, color = 'k', marker = 'o')
        ax.scatter(peak.idxmax(), baseval, color = 'k', marker = '*')
        
        # Fix axis scale, set title, add labels
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_title('Polyfit Baseline: ' + samp)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
        ax.set_ylabel("Intensity",family="serif",  fontsize=12)
        
        # Set minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.show()    
        
    # calculate fluorescence ratio
    findex = baseval/peakval
        
    return findex

# calculate Fluorescence Ratio and add to index df
for col in tqdm(df.columns):
    sampdf.loc[col,'FluorRatio'] = fluorindex(df, col, plot = False)
    

# All following analyses will be conducted on corrected spectra
# get the corrected baseline for all spectra in new df
cordf = pd.concat(map(lambda col: arPLS_baseline(df[col], 
                                              ratio = 1e-6, 
                                              lam=1e4, 
                                              niter=5500, 
                                              full_output=True)[1].rename(col), 
                       tqdm(df.columns)), 
                   axis=1)

# aftter correction, must drop spectra whose fluorescence changes shape of
# spectrum by having CO3 peak higher than PO4 peak
po4peaks = cordf.idxmax()

cordf  = cordf.drop(po4peaks[po4peaks > 980].index, axis = 1)
sampdf = sampdf.drop(po4peaks[po4peaks > 980].index, axis = 0)

# to keep consistency, all values will be normalized to their highest peak
cordf = cordf/cordf.max()

# now define calculation of the CIraman index
def ci_raman(data, add_baseline=False, plot=False):
    """
    

    Parameters
    ----------
    data : Pandas Series
        Signal ready for Raman diegenesis calculations with the wavenumber as
        index.
    add_baseline : bool, optional
        If True, a baseline is created for the calculation. 
        The default is False.

    Returns
    -------
    ci
        Crystalinity index of the Raman spectrum.

    """
    
    # limit extent in case of baseline correction, creating a copy
    data = data.loc[900: 1050].copy()
    
    if add_baseline:
        # find minima to correct baseline
        left  = data.loc[900: 950]
        lmini = signal.argrelmin(left.values)[-1] # closest local minimum to peak
        lmin  = (left.iloc[lmini].idxmin(), 
                 left.loc[left.iloc[lmini].idxmin()])
        
        right = data.loc[980: 1040]
        rmini = signal.argrelmin(right.values)[0] # closest local minimum to peak
        rmin  = (right.iloc[rmini].idxmin(), 
                 right.loc[right.iloc[rmini].idxmin()])
                        
        # create linear baseline for correction
        m = (rmin[1]-lmin[1])/(rmin[0]-lmin[0])
        b = rmin[1] - m*rmin[0]
    
        x = np.linspace(min(data.index), max(data.index), len(data))
        y = [(m*x0 + b) for x0 in x]
    
        # correct signal with baseline
        new_df = data - y
        
        # redefine y_col for all the following calculations
        data = new_df
        
        
    # find peaks with relatively big heights to select peak around 960 cm-1    
    peaks_ix, _ = signal.find_peaks(data, prominence=0.5)
    
    maxima = data.iloc[peaks_ix]
    
    #find most likely candidate for PO4 peak at 960 cm-1
    ix = maxima.index.to_list() # get list of posible indices
    locp = sorted(ix, key = lambda x: abs(960 - x)) # sort list by proximity
    
    po4 = data[locp[0]]
    
    # calculate FWHM, staring with figuring out half max
    hm = po4/2
    
    # root function adapted from solution provided by:
    # https://stackoverflow.com/users/4124317/importanceofbeingernest
    def find_roots(x,y):
        # find where the sign changes if height value subtracted
        s    = np.abs(np.diff(np.sign(y))).astype(bool)
        # find the difference between the known values to interpolate
        numerator   = np.diff(x)[s]
        denominator = np.abs(y.iloc[1:][s].values[:]/y.iloc[:-1][s].values[:])
        diff = numerator/(denominator + 1)
        return x[:-1][s] + diff #return x values with interpolation adjustment

    z = find_roots(data.index.array, data-hm) # ix needs to be an array
    
    # get the FWHM from the calculated x root values above
    fwhm = z[1] - z[0]
    
    # calculate CI
    ci = 4.9/fwhm
    
    if plot:
        # create output plots for more transparent output and interpretation
        # Plot lines, peaks, and baselines to ensure accuracy
        fig = plt.figure()
        ax  = fig.add_subplot()
        plt.tight_layout()
        
        # Plot original data
        ax.plot(data, color = "k")
        
        # Add plot for baseline
        line_x    = data.index.to_list()
        corr_line = [0]*len(line_x) 
        ax.plot(line_x, corr_line, color = "k", linestyle = "--")
        
        # add plot for FWHM
        ax.hlines(hm,   xmin=z[0], xmax=z[1], colors="k", linestyle = "--")
        ax.hlines(po4,  xmin=z[0], xmax=z[1], colors="k", linestyle = ":" )
        ax.vlines(z[0], ymin=0,    ymax=po4,  colors="k", linestyle = ":" )
        ax.vlines(z[1], ymin=0,    ymax=po4,  colors="k", linestyle = ":" )
        
        # label axes and tight layout around signal
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=10.5)
        ax.set_ylabel("Arbitrary Intensity Units",family="serif",  fontsize=10.5)
        
        ax.set_title(str(data.name))
        
        # add annotation of CIraman in plot
        ax.annotate("$CI_{raman}$ = " + f"{ci: .2f}", 
                    xy=(990, hm),  xycoords='data', family="serif")
        
        # Set Minor tick marks
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        
        plt.show()
     
    return ci


# add ci_raman values to sampdf
for col in tqdm(cordf.columns):
    sampdf.loc[col,'CIraman'] = ci_raman(cordf[col], 
                                             add_baseline=True,
                                                plot = False)
    
# now time to calculate organic content index
def amIPO4raman(data, smooth=False, plot=False):
    '''
    This formula calculates the 960:1636 ratio for a given Raman spectrum of bone

    Parameters
    ----------
    data : Pandas Series
        Pandas DataFrame containing the wavelength and signal data. Data must 
        be in the form of the x_column being the index and selecting the y_col
        E.g., data=df['selection']
    smooth : bool
        if True, smooths the signal using a Savitzky-Golay algorithm with a
        window of 25 and a polyorder of 2. If False, no smoothing is applied.
        Default is False.

    Returns
    -------
    amipo4 : float
        the C/P ratio for the spectrum provided.

    '''
    rang  = [800, 1800]
    apeak = 1636
    ppeak = 960
    
    data = data.loc[rang[0]: rang[1]].copy()
    
    # make sure data is not below or above 0 by raising up or bringing down
    data = data - data.min()
    
    if smooth:
        data = pd.Series(signal.savgol_filter(data, window_length=11, 
                      polyorder=5, deriv=0),
                         index = data.index)
    
    # find all local minima and maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    maxima = data.iloc[signal.argrelextrema(data.values, 
                                               np.greater_equal, order=n)[0]]
        
    # get uncorrected peak heights for PO4
    ix = maxima.index.to_list() # get list of posible indices
    locp = sorted(ix, key = lambda x: abs(ppeak - x)) # sort list by proximity
    
    # since peak of AmideI at 1636 is usually a shoulder of the peak at 1671
    # peak height should be found by finding the highest value between 1610 and
    # 1650 to try and get the peak next to 1671 but not that peak    
    # find closest maximum to 1636
    loca = sorted(ix, key = lambda x: abs(apeak - x))[:2]
    
    if not loca:
        loca = [data.loc[1620:1650].max()]
    
    # get peak heights
    Aa = data[loca[0]]
    Ap = maxima[locp[0]]
    
    # Calculate AmIPO4 ratio (see France et al 2014)
    amipo4 = Ap/Aa
    
    if plot:
        # Plot lines, peaks, and baselines to ensure accuracy
        fig = plt.figure()
        ax  = fig.add_subplot()
        plt.tight_layout()
        
        # Plot original data
        ax.plot(data, color = "k", linewidth = 1)
        
        # Add plot for baseline
        line_x    = data.index.to_list()
        corr_line = [0]*len(line_x) 
        ax.plot(line_x, corr_line, color = "k", linestyle = "--" )
        
        # Plot lines for peak heights
        ax.vlines(x=loca[0], ymin=0, ymax=Aa, 
                           color = "k", linestyle = "-.")
        ax.vlines(x=locp[0], ymin=0, ymax=Ap,
                           color = "k", linestyle = "-.")
        
        # label axes and title
        ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=10.5)
        ax.set_ylabel("Arbitrary Intensity Units",family="serif",  fontsize=10.5)
        
        ax.set_title(str(data.name))
        
        # add annotation of ratio value in plot
        ax.annotate("$PO_4$:Amide I = " + f"{amipo4: .2f}", 
                    xy=(1200, Ap/2 ),  xycoords='data', family="serif")
            
        # Set Minor tick marks
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        
        plt.show()
    
    return amipo4

# add amIPO4 values to sampdf
for col in tqdm(cordf.columns):
    sampdf.loc[col,'AmIPO4'] = amIPO4raman(cordf[col], 
                                               smooth=True, 
                                                   plot=False)
    
# get peak locations for PO4 and CO3 to calculate CO3-PO4 ratio (concat later)
def peakfind(data, plot=False):
    # find all peaks in data
    peaks = signal.find_peaks(data)
    
    peaklocs = data.iloc[peaks[0]].index.to_list()
    
    # find the peaks closest to 960 for PO4
    locsp = sorted(peaklocs, key = lambda x: abs(x - 960))[:5]
    
    # get location of highest peak within possible range to get PO4 peak
    po4 = data.loc[locsp].idxmax()
    
    # find the peaks closest to 960 for PO4
    locsc = sorted(peaklocs, key = lambda x: abs(x - 1070))[:5]
    
    # make sure the locations for CO3 do not include the PO4 peak
    if po4 in locsc:
        locsc.remove(po4)
    
    # get location of highest peak within possible range to get PO4 peak
    co3 = data.loc[locsc].idxmax()
    
    if plot:
        plt.plot(data)
        plt.axvline(po4)
        plt.axvline(co3)
        plt.title(str(data.name))
        plt.show()
    
    return po4, co3

# run peak finder on the corrected df
corpeaks = {}
for col in cordf.columns:
    corpeaks[col] = peakfind(cordf[col], plot=False)
    
# get CO3 peak value which now is equivalent to CO3/PO4 ratio, add to sampdf
for key, val in corpeaks.items():
    ap, ac = cordf[key].loc[[val[0], val[1]]]
    # since the ratio of val at c by val at p has already been normalized
    sampdf.loc[key, 'CO3-PO4'] = ac 

'''
QC was visually determined on corrected spectra rather than statistically.
Corected spectra exist on a range from 0 to 1 (sometimes crossing belo 0)
called normalized intensity.
Stochastic QC:
Great: peaks between 800:850 and 875:925 below 0.075 threshold
Good:  peaks between 800:850 and 875:925 below 0.15 threshold    
Fair:  one of the above 2 peaks cross the 0.15 threshold
Bad:   either peak above 0.3 threshold or both peaks above 0.15 threshold

boxplots demonstratee that this approach actually has basis since all but CI
correlate well with QC category
'''

# Now that we have all indices, we have to check correlations to select
# between QC categories
# Since these need to beconfirmed prior to selecting duplicates for model 
# building, analyses should only be conducted on the original 100 spectra
indices = ['CIraman', 'FluorRatio','AmIPO4','CO3-PO4']

indexdf = sampdf[indices + ['QC']].iloc[:-32].copy()

# boxplot plot for 'CIraman', 'FluorRatio','AmIPO4','CO3-PO4', done each 
# individually to save on lines of code
ax = indexdf.boxplot(column = 'CO3-PO4', by = 'QC', grid = False, patch_artist=True,
                    boxprops=dict(   color = 'k',linestyle='-', facecolor = '0.5', alpha = 0.5),
                    flierprops=dict(marker = '*'),
                    medianprops=dict(color = 'k',linestyle='-'),
                    whiskerprops=dict(           linestyle='-'),
                    capprops=dict(               linestyle='-'))

# Fix axis scale, set title, add labels
plt.suptitle('')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_title('Observer Defined QC and ' + 'CO$_3$\PO$_4$', family = "serif")
ax.set_xlabel("QC Category", family="serif",  fontsize=12)
ax.set_ylabel('CO$_3$\PO$_4$' ,family="serif",  fontsize=12)

# Set minor ticks
ax.yaxis.set_minor_locator(AutoMinorLocator())

# add horizontal major axes
ax.yaxis.grid()

# plt.savefig(figout + 'QCCO3PO4.png', dpi = 300, bbox_inches = 'tight')
# show figure and clear for next
plt.show()

# Test the significance of the correlation in the boxplots of the indices
# since QC is ordinal and indeices are continuous, must do a Kendall's tau
ordinals = {'great' : 4, 'good': 3, 'fair': 2, 'bad' : 1} # define order

# make nominal ordinal variable into numerical
indexdf['QC'].replace(ordinals, inplace=True)

# run Kendall tau for each of the columns (c variant)
taures = {}
for col in indexdf.columns[:-1]:
    tau = stats.kendalltau(indexdf.iloc[:,-1], indexdf[col], variant = 'c')
    taures[col] = (tau[0], tau[1])
    
# convert results to dataframe for better output
taudf = pd.DataFrame.from_dict(taures, orient = 'index', columns=['tau', 'p-val']) 

# add effect size eta2
def eta2(tau):
    return np.square(np.sin(.5*np.pi*tau))

taudf = taudf.merge(eta2(taudf['tau']).rename('eta2'), 
                    copy=False, 
                    left_index=True, 
                    right_index=True)

# export tau and index results as excel sheets
#taudf.to_excel(direct + dfdir + 'Results_KendallTau.xlsx')

#sampdf[indices].to_excel(direct + dfdir + 'Results_Indices.xlsx')






# INDEX CALCULATION COMPLETE
# BEGIN CORRELATION ANALYSES


# transpose dataframes, add isotope data and add potentially important data
# remove the index name for ease in joining later and readability
cordf.index.name = None

# transpose df to join on sample number
tcordf = cordf.T


# concat the peak locations previously found to the transposed df
tcordf = tcordf.join(pd.DataFrame.from_dict(corpeaks, 
                                        orient='index', 
                                        columns=['PO4peak', 'CO3peak']))

#join isotope data to df
tcordf = tcordf.join(sampdf[['d13Cap', 'd18Ovpdb']])





# NOW LASSO ANALYSIS BEGINS 
# select only the spectra with QC of "good" or "great"
qcix = sampdf[sampdf['QC'].isin(('good', 'great'))].index

good_tcordf = tcordf.loc[qcix]

# define columns that will be Y and those that will be X
y_cols = ['d13Cap' , 'd18Ovpdb']
x_cols = [col for col in tcordf.columns if col not in y_cols]

# do alpha selection using a Lasso CV and then train and test the model
# using Lasso with the appropriate alpha
def alpha_select(data, x, y):
        # define predictor and response variables
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    
    
    # Apply StandardScaler to get unit variance and 0 mean
    scaler = StandardScaler(with_mean = False, with_std = True)
    
    X = scaler.fit_transform(X)
    
    
    # define cross validation method
    splits  = 8
    repeats = 1000
    
    cv = RepeatedKFold(n_splits=splits, n_repeats = repeats)
    
    # define model
    lasso = LassoCV(n_alphas = 1000, 
                    max_iter = 10000, 
                    cv = cv, 
                    n_jobs = -1)
    
    # run the model
    model = lasso.fit(X, Y)
    
    # now get r2 for training
    r2 = model.score(X, Y)
    
    # get # of coefficients from Lasso and alpha
    expvar = np.count_nonzero(model.coef_)
    
    alpha = model.alpha_
    
    # calculate RMSE of test data for the selected alpha
    mse  = model.mse_path_.mean(axis=-1).min()
    rmse_test = np.sqrt(mse)
    
    # calculate MAE for all data
    pred_vals = model.predict(X)
    abserror  = np.abs(pred_vals - Y)
    
    mae       = np.mean(abserror)
    
    print(f'LassoCV  for Optimal Alpha with {splits}-fold CV {repeats} repeats\n'     +\
          f'R2 model:              {r2}\n'     +\
          f'Alpha (lambda):        {alpha}\n'  +\
          f'# of Coefficients:     {expvar}\n' +\
          '\n'  +\
          f'MSE of testing data:   {mse}\n'   +\
          f'RMSE of testing data:  {rmse_test}\n'  +\
          '\n' +\
          f'MAE of all data:       {mae}')
    
    metrics = {'Title:' : f'LassoCV Model with {splits}-fold CV {repeats} repeats\n',
                'R2': r2,
               'Alpha(lambda)': alpha,
               'Ncoefs'  : expvar,
               'MSEtest' : mse,
               'RMSEtest': rmse_test,
               'MAE:'    : mae}
    
    return alpha, metrics


d13calpha, d13calphamets = alpha_select(good_tcordf, 
                                        x_cols, 
                                        y_cols[0])

d18oalpha, d18oalphamets = alpha_select(good_tcordf, 
                                        x_cols, 
                                        y_cols[1])

# creating linear model using Lasso with alpha from cross validation
def lasso(data, x, y, alpha, plot = True, name = '', split=True, rand = 1, print_out = False):    
    
    # define predictor and response variables
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    
    # train/test split if split == True
    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            random_state = rand,
                                                            test_size = 0.1)
    else:
        X_train = X
        Y_train = Y
    
    # Apply StandardScaler to get unit variance and 0 mean
    scaler = StandardScaler(with_mean = False, with_std = True).fit(X_train)
    
    X_train = scaler.transform(X_train)
    
    if split:
        X_test = scaler.transform(X_test)
    
    # define model
    lasso = Lasso(alpha = alpha, fit_intercept = True)
    
    # run the model
    model = lasso.fit(X_train, Y_train)
    
    # now get r2 for training
    r2_train = model.score(X_train, Y_train)
    
    # get # of coefficients from Lasso
    expvar = np.count_nonzero(model.coef_)
    
    # calculate RMSE of training data
    mse_train  = np.sum(np.square(model.predict(X_train).flatten()-Y_train))/len(Y)
    rmse_train = np.sqrt(mse_train)
    
    # calculate mean absolute error for training data
    mae_train = np.mean(np.abs(model.predict(X_train).flatten()-Y_train))
    
    if split:
        # calculate RMSE and MAE for test data
                
        # calculate RMSE of testing data
        mse_test  = np.sum(np.square(model.predict(X_test).flatten()-Y_test))/len(Y)
        rmse_test = np.sqrt(mse_test)
        
        # calculate mean absolute error for testing data
        mae_test = np.mean(np.abs(model.predict(X_test).flatten()-Y_test))
        
    if print_out:
        print('Model with Train/Test Split at 10%\n'     +\
              f'R2 model:              {r2_train}\n'     +\
              f'RMSE of training data: {rmse_train}\n'   +\
              f'MAE of trainig data:   {mae_train}\n'      +\
              '\n'  +\
              f'Alpha (lambda):        {alpha}\n'  +\
              f'# of Coefficients:     {expvar}\n' +\
              '\n'  +\
              f'MAE of testing data:   {mae_test}\n'   +\
              f'RMSE of testing data:  {rmse_test}\n')
    
    metrics = {'Title:' : 'Model with Train/Test Split at 10%\n',
               'R2 model':              r2_train,
               'RMSE of training data': rmse_train,
               'MAE of trainig data':   mae_train,
               'Alpha (lambda)':        alpha,
               '# of Coefficients':     expvar,
               'MAE of testing data':   mae_test,
               'RMSE of testing data':  rmse_test}
                       
    if plot:   
        # define the thresholds dependent on the isotope being explored
        if   y == 'd18Ovpdb':
            rid = 3.1
        elif y == 'd13Cap':
            rid = 1.2
        else:
            rid = (max(Y)-min(Y))/2
        
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [7, 6])
        
        # Scatter plot of training data
        ax.scatter(model.predict(X_train), Y_train, marker = 'o', 
                                                    color  = 'k', 
                                                    alpha  = 0.7)
    
        # add annotations of sample size, r2, adjr2, and rmse
        ax.annotate(f'N       =  {len(Y_train)}',        
                                    (.69, .244),  
                                    xycoords = 'subfigure fraction')                  
        ax.annotate(f'R$^2$      = {r2_train: .2f}',        
                                    (.69, .211),  
                                    xycoords = 'subfigure fraction')
        ax.annotate(f'RMSE = {rmse_train: .2f}',       
                                    (.69, .178),  
                                    xycoords = 'subfigure fraction')
        ax.annotate(f'Coefficients = {expvar}',        
                                    (.69, .145),  
                                    xycoords = 'subfigure fraction')
        
        # Lines depicting fit and 0.6 real interpretive difference
        corline_x = np.linspace(min(Y), max(Y), len(Y))
        corline_y = corline_x
        
        ax.plot(corline_x, corline_y,      '--k',  linewidth = 1)
        ax.plot(corline_x, corline_y + rid, ':k' , linewidth = 1)
        ax.plot(corline_x, corline_y - rid, ':k' , linewidth = 1)
        
        # Fix axis scale, set title, add labels
        ax.set_title( f'Correlation of IRIS to IRSM: {name}', 
                      family          = 'serif', 
                      math_fontfamily = 'dejavuserif')
        ax.set_xlabel(f'IRIS {name}', 
                      family='serif', fontsize = 12, 
                      math_fontfamily = 'dejavuserif')
        ax.set_ylabel(f'IRMS {name}', 
                      family = 'serif', fontsize = 12, 
                      math_fontfamily = 'dejavuserif')
        ax.set_xlim([min(Y) - .05, max(Y) + .05])
        ax.set_ylim([min(Y) - .05, max(Y) + .05])
        
        # Set minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        #plt.savefig(figout + y + '_Lasso_results_wBands.png', bbox_inches='tight', dpi = 300 )
        
        # show figure and clear for next
        plt.show()
        
        if split:
            # plot Absolute Error for each test value and marginal boxplot
            fig, (ax1, ax2) = plt.subplots(ncols = 2,
                                           sharey = 'row',
                                           figsize= [7, 6], 
                                           gridspec_kw={'width_ratios': [5, 1],
                                                        'hspace': 0.5, 
                                                        'wspace': 0.05})
            
            # plot absolute error per testing point
            pointnum = np.linspace(1, len(Y_test), len(Y_test))
            abserror = model.predict(X_test).flatten()-Y_test
            
            ax1.scatter(pointnum, abserror, marker = 'o', 
                                           color  = 'k', 
                                           alpha  = 0.7)
            
            # now make the boxplot
            ax2.boxplot(abserror, widths = 0.7,
                            patch_artist=True,
                    boxprops=dict(color = 'k',linestyle='-', 
                                  facecolor = '0.5', alpha = 0.5),
                    flierprops=dict(marker = '*'),
                    medianprops=dict(color = 'k',linestyle='-'),
                    whiskerprops=dict(           linestyle='-'),
                    capprops=dict(               linestyle='-'))
            
            
            # Line for 0.6 real interpretive difference in absolute error
            ax1.axhline(rid, linestyle = '--', color = 'k')
            
            # Fix axis scale, set title, add labels
            fig.suptitle(f'Error for Test Samples: {name}', y=0.92,
                          family          = 'serif', 
                          math_fontfamily = 'dejavuserif')
            ax1.set_xlabel('Test Sample Number', 
                          family='serif', fontsize = 12, 
                          math_fontfamily = 'dejavuserif')
            ax1.set_ylabel('Error (IRIS - IRMS)', 
                          family = 'serif', fontsize = 12, 
                          math_fontfamily = 'dejavuserif')
            
            # Set minor ticks
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
            ax1.xaxis.set_minor_locator(NullLocator())
            
            # add grid at major ticks
            ax1.grid(axis='y', alpha = 0.5)
            
            
            # Remove labels from boxplot
            ax2.axes.yaxis.set_visible(False)
            ax2.axes.xaxis.set_visible(False)
            
            #plt.savefig(figout + y + '_MAE_results_wBox.png', bbox_inches='tight', dpi = 300 )
            
            # show figure and clear for next
            plt.show()
        
  
    return model, metrics


# run lasso on d13c
d13cmod, d13cmetrics = lasso(good_tcordf, 
                             x_cols, 
                             y_cols[0],
                             alpha = d13calpha,
                             name = '$\delta^{13}C_{apatite}$',
                             plot = True,
                             print_out = True)

# run lasso on d18Ovpdb; BAD MODEL
d18omod, d18ometrics = lasso(good_tcordf, 
                             x_cols, 
                             y_cols[1],
                             alpha = d18oalpha,
                             name = '$\delta^{18}O_{vpdb}$',
                             plot = True,
                             print_out = True)


# now get which coefficients are being used for each model
posfreqc = [np.array(x_cols)[i] for i in (d13cmod.coef_ > 0).nonzero()]
negfreqc = [np.array(x_cols)[i] for i in (d13cmod.coef_ < 0).nonzero()]


posfreqo = [np.array(x_cols)[i] for i in (d18omod.coef_ > 0).nonzero()]
negfreqo = [np.array(x_cols)[i] for i in (d18omod.coef_ < 0).nonzero()]

# export model, data, (and metrics separately) to import into UT donors analysis
dump(d13cmod, direct + 'Models\\' + 'd13c_model.joblib')
dump(d18omod, direct + 'Models\\' + 'd18o_model.joblib')

# Exporting metrics as string
def export_metrics(metrics, posfreq, negfreq, name):
    e1 = str(metrics).strip('{}').replace(', ', '\n')
    e2 = '\n'.join(posfreqc[0])
    e3 = '\n'.join(negfreqc[0])
    
    string = f'Model for {name}'    +\
             'Metrics for Model:\n' +\
                e1      + '\n'      +\
             'Frequencies (cm-1) with Positive Coefficients in the Model:\n' +\
                e2      + '\n'      +\
             'Frequencies (cm-1) with Negative Coefficients in the Model:\n' +\
                e3
                
    with open(direct + 'Models\\' + f'{name}_ModelMetrics.txt', 'w') as f:
        f.write(string)
    
export_metrics(d13cmetrics, posfreqc, negfreqc, 'd13Capatite')
export_metrics(d18ometrics, posfreqo, negfreqo, 'd18Ovpdb')

# export training data for standard scaler in other implementations
good_tcordf.to_excel(direct + 'Models\\' + 'TrainingData.xlsx')

# Export band frequencies and coefficients (and intercept if necessary) as xlsx
d13cmodbands = pd.Series(d13cmod.coef_, index = x_cols, name = 'd13C Bands')
d18omodbands = pd.Series(d18omod.coef_, index = x_cols, name = 'd18O Bands')

# get only the non-zero coefficients
d13cmodbands = d13cmodbands.iloc[d13cmodbands.to_numpy().nonzero()]
d18omodbands = d18omodbands.iloc[d18omodbands.to_numpy().nonzero()]

# export to excel
d13cmodbands.to_excel(direct + 'Models\\' + 'd13c_BandCoeffs.xlsx')
d18omodbands.to_excel(direct + 'Models\\' + 'd18o_BandCoeffs.xlsx')


# checking reassociation of of skeletal fragements using Berg et al. 2022
reasdf = sampdf['d13Cap'].iloc[:-32]


indices = list(combinations(reasdf.index, 2))

vals    = list(combinations(reasdf, 2))

combs = list(zip(indices, vals))

comarray = np.array(combs)

comarray.shape = (len(combs), 4)


comdf = pd.DataFrame(comarray, columns = ['ind1', 'ind2', 'val1', 'val2'])

comdf[['val1', 'val2']] = comdf[['val1', 'val2']].astype(float)

comdf['diff'] = abs(comdf['val1'] - comdf['val2'])

# export combinatorial difference table
comdf[['ind1', 'ind2', 'diff']].to_excel(direct + 'Results\\' + 'PPCombDiff.xlsx')




