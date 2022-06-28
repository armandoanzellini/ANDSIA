# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:12:30 2022

@author: Armando Anzellini
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
from numpy.linalg import norm
from scipy.sparse import linalg
from numpy.polynomial import polynomial as p
from matplotlib.ticker import AutoMinorLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

direct  = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\Patakfalva-Papdomb\\"

sampdir = "Samples Raman\\Best\\" # select only the best spectra

dupdir  = "Samples Raman\\Duplicates\\" 

file = 'PatakfalvaUnassociatedIsotope_Anzellini.xlsx'

# open non-raman data
sampdf  = pd.read_excel(direct + file, sheet_name = 'All')

sampdp = pd.read_excel(direct + file, sheet_name = 'Duplicates')

# drop unnecessary columns from both sample df and duplicates
dropcols = ['Weight', 'Minitube', 'Carb Sample', 'Solution Volume',
            'Batch #', 'Final Tube+Sample', 'Final Weight']

sampdf.drop(dropcols, axis = 1, inplace=True)
sampdp.drop(dropcols, axis = 1, inplace=True)

# change the Sample # column to included leading zeros and an S suffix
sampdf['Sample #'] = sampdf['Sample #'].apply(str).str.zfill(3) # pad w/ zeros
sampdp['Sample #'] = sampdp['Sample #'].apply(str).str.zfill(3) # pad w/ zeros

sampdf['Sample #'] = 'S'+ sampdf['Sample #'] # add S in front of number
sampdp['Sample #'] = 'S'+ sampdp['Sample #'] # add S in front of number

# make sample # cloumn the new index for both df and dup
sampdf.set_index('Sample #', inplace = True)
sampdp.set_index('Sample #', inplace = True)

# Opening up the PP raman spectra
# now list the folders in the donors directory to open each file
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

# cut df to fingerprint region of the spectrum (removing the first bit that has too much fluorescence)
df = df.loc[775:1900]

# strip column names of integration and scan numbers
pattern = r'\(\d-\d{2}\)(\(\d\))?'

df.columns = (re.sub(pattern, '', col) for col in df.columns)

# Subtract minimum from each column to remove negative values
for col in df.columns:
    df[col] = df[col] - df[col].min()

# calculate fluorescence ratio by taking area under curve or value below peak
# and dividing by PO4 peak total height
def fluorindex(sr, samp, granularity=11, index = 'height', plot = False):
    # fit polynomial at given values
    coef = p.polyfit(sr.index, sr[samp], granularity) # create coefficients
    ffit = p.Polynomial(coef) # create polynomial function with given coefs
    
    # Find the amount that the baseline must be shifted to avoid signal
    shiftval = max(ffit(sr.index) - sr[samp])
    
    if plot:
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [15, 5])
        
        # Spectral plot
        ax.plot(sr[samp], '-', color = 'k', linewidth=0.6)
        
        # baseline plot
        ax.plot(sr.index, ffit(sr.index) - shiftval, ':', color = 'k', linewidth=0.6)
        
        # Fix axis scale, set title, add labels
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_title('Polyfit Baseline: ' + samp)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
        ax.set_ylabel("Intensity",family="serif",  fontsize=12)
        
        # Set minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.show()    
    
    # for area of fluorescence calculation
    integral = ffit.integ()

    integrand = integral(sr.index[-1]) - integral(sr.index[0])
    
    # find peak height of PO4 peak for index calculations
    peaks = signal.find_peaks(sr[samp])
    
    # get peak that fits expected values
    peak = sr[samp].iloc[peaks[0]].loc[940:990]
    
    peakval = peak.max()
    
    # get value of polynomial at the given peak wavelength
    polyval = ffit(peak.idxmax())
    
    # if using height value not area index
    if index == 'height':
        findex = polyval/peakval
        
    # if using areas not heights
    if index == 'area':
        findex = peakval/integrand
        
    return findex

for col in tqdm(df.columns):
    sampdf.loc[col,'FluorRatio'] = fluorindex(df, col, plot = True)
    
# first we have to define and apply the arPLS baseline correction
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

# get the corrected baseline for all spectra in new df
cordf = pd.concat(map(lambda col: arPLS_baseline(df[col], 
                                                  ratio = 1e-6, 
                                                  lam=1e4, 
                                                  niter=5500, 
                                                  full_output=True)[1].rename(col), 
                       tqdm(df.columns)), 
                   axis=1)

# plot the figures properly to compare to online raman cleaner
def plotspec(sr):
    # Create subplot
    fig, ax = plt.subplots(1, figsize= [15, 5])
    
    # Spectral plot
    ax.plot(sr, '-', color = 'k', linewidth=0.6)
    
    # Fix axis scale, set title, add labels
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_title('Smoothed Corrected Spectrum: ' + sr.name)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
    ax.set_ylabel("Relative Absorbance",family="serif",  fontsize=12)
    
    # Set minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # add a temporary horizontal at 0.15 to select which samples need rerun
    plt.axhline(0.15, color = 'k', linestyle = ':')
    plt.axhline(0.30, color = 'k', linestyle = '--')
    
    # show figure and clear for next
    plt.show()
    
for col in cordf.columns:
    plotspec(cordf[col])

# aftter correction, must drop spectra whose fluorescence changes shape of
# spectrum by having CO3 peak higher than PO4 peak
po4peaks = cordf.idxmax()

cordf  = cordf.drop(po4peaks[po4peaks > 980].index, axis = 1)
df     = df.drop(po4peaks[po4peaks > 980].index, axis = 1)
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

# run peak finder for each of the dfs
dfpeaks = {}
for col in df.columns:
    dfpeaks[col] = peakfind(df[col])

corpeaks = {}
for col in cordf.columns:
    corpeaks[col] = peakfind(cordf[col], plot=False)
    
# get CO3 peak value which now is equivalent to CO3/PO4 ratio, add to sampdf
for key, val in corpeaks.items():
    ap, ac = cordf[key].loc[[val[0], val[1]]]
    # since the ratio of val at c by val at p has already been normalized
    sampdf.loc[key, 'CO3-PO4'] = ac 
    
# test which models for IRIS work the best then export for separate streamlit module
# get second derivative data (See Ortis-Herrero), clarifies separate peaks better
# add isotopic values to columns and transpose table to run PLS

# 1) get second derivative df from corrected df
derdf = pd.concat(map(lambda col: pd.Series(signal.savgol_filter(cordf[col], 
                                                                 window_length=15, 
                                                                 polyorder=3, 
                                                                 deriv=2)[:-2],
                                            index = df[col].index[:-2], 
                                            name=col), 
                      tqdm(df.columns)), 
                  axis=1)

# and get peak locations for PO4 and CO3
derpeaks = {}
for col in derdf.columns:
    derpeaks[col] = peakfind(derdf[col])

# now we have original, baseline corrected and normalized, and 2nd derivative df
# remove the index name from all of them
df.index.name    = None
cordf.index.name = None
derdf.index.name = None

# 2) transpose all dfs to join on sample number
tdf    = df.T
tcordf = cordf.T
tderdf = derdf.T


# concat the peak locations previously found to the transposed dfs
tdf    = tdf.join(pd.DataFrame.from_dict(dfpeaks, 
                                        orient='index', 
                                        columns=['PO4peak', 'CO3peak']))

tcordf = tcordf.join(pd.DataFrame.from_dict(corpeaks, 
                                        orient='index', 
                                        columns=['PO4peak', 'CO3peak']))

tderdf = tderdf.join(pd.DataFrame.from_dict(derpeaks, 
                                        orient='index', 
                                        columns=['PO4peak', 'CO3peak']))

# 3) join isotope data to each of the dfs
tdf    = tdf.join(sampdf[['d13C', 'd18Ovsmow']])
tcordf = tcordf.join(sampdf[['d13C', 'd18Ovsmow']])
tderdf = tderdf.join(sampdf[['d13C', 'd18Ovsmow']])

# 4) begin results df and run PLS on each
res = pd.DataFrame(sampdf[['d13C', 'd18Ovsmow']])

# define function that takes X and Y and returns model to repeat
def pls(data, x, y, components, plot = True):    
    
    # define predictor and response variables
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    
    # split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size=0.1, 
                                                        random_state=0)
    
    # define cross-validation method
    cv = LeaveOneOut()
    
    # define model
    pls = PLSRegression(n_components = components)
    
    # run the model
    model = pls.fit(X_train, Y_train)
    
    # evaluate model on the training data through CV by RMSE
    scores = cross_val_score(pls, X_train, Y_train, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
    
    rmse = np.sqrt(np.mean(np.absolute(scores)))
    
    # now calculate r2 from test data
    tss = np.sum(np.square(Y_test - np.mean(Y_test)))
    rss = np.sum(np.square(Y_test - model.predict(X_test).flatten()))
    
    r2 = 1 - (rss/tss)
    
    
    # calculate adjusted r2
    dof = (len(X) - 1)/(len(X) - components)
    adj_r2 = 1-(1-r2) * dof
        
    if plot:
        plt.scatter(model.predict(X_train), Y_train)
        plt.scatter(model.predict(X_test), Y_test)
        plt.plot(Y, Y)
        plt.plot(Y+0.6, Y)
        plt.plot(Y-0.6, Y)
        plt.show()
    
    return model, rmse, r2, adj_r2
    
# run the model for each df using d13C first and compare RMSE
val = 'd13C' # which isotopic value is being analyzed

r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    orgmod, orgrmse, orgr2, orgr2adj = pls(tdf, tdf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = orgr2
    adjr2[i] = orgr2adj
    rmse[i]  = orgrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Original Data')
plt.show()

plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Original Data')
plt.show()

r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    cormod, corrmse, corr2, corr2adj = pls(tcordf, tcordf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = corr2
    adjr2[i] = corr2adj
    rmse[i]  = corrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Baseline Corrected Data')
plt.show()


plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Baseline Corrected Data')
plt.show()


r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    dermod, derrmse, derr2, derr2adj = pls(tderdf, tderdf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = derr2
    adjr2[i] = derr2adj
    rmse[i]  = derrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Second Derivative Data')
plt.show()

plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Second Derivative Data')
plt.show()


# try removing outliers in the data using diagenesis markers
outdf = sampdf[['CIraman','FluorRatio','AmIPO4','CO3-PO4','% Yield','%CaCO3']]    
    
# check for normality to find outliers
for col in outdf.columns:
    outdf[col].hist()
    plt.title(col)
    plt.show()
    skew = stats.skew(outdf[col], bias=False)
    kurt = stats.kurtosis(outdf[col], bias=False)
    print(f'{col}\n skew: {skew}\n kurt: {kurt}\n')

# make PCA and do elliptical outlier detection sklearn
pc = PCA(n_components = 2)

# transform outdf to unit variance
uvoutdf = pd.DataFrame(StandardScaler().fit_transform(outdf), columns=outdf.columns, index=outdf.index)

# PCA transform of uvoutdf 
model = pd.DataFrame(pc.fit_transform(uvoutdf), columns=['PC1', 'PC2'], index=uvoutdf.index)

pc1mean = np.mean(model['PC1'])
pc2mean = np.mean(model['PC2'])

pc1std = np.std(model['PC1'])
pc2std = np.std(model['PC2'])

from matplotlib.patches import Ellipse

plt.scatter(model['PC1'], model['PC2'])
ellipse = Ellipse(xy=(pc1mean, pc2mean), width=pc1std*6, height=pc2std*6, 
                        edgecolor='r', fc='None', lw=2)
plt.gca().add_patch(ellipse)
plt.show()

# now try to make the model only with the "good" and "great" spectra
qcix = sampdf[sampdf['QC'].isin(('good', 'great'))].index

# run the model for each df using d13C first and compare RMSE
val = 'd13C' # which isotopic value is being analyzed

r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    orgmod, orgrmse, orgr2, orgr2adj = pls(tdf.loc[qcix], tdf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = orgr2
    adjr2[i] = orgr2adj
    rmse[i]  = orgrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Original Data')
plt.show()

plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Original Data')
plt.show()

r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    cormod, corrmse, corr2, corr2adj = pls(tcordf.loc[qcix], tcordf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = corr2
    adjr2[i] = corr2adj
    rmse[i]  = corrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Baseline Corrected Data')
plt.show()


plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Baseline Corrected Data')
plt.show()


r2    = {}
adjr2 = {}
rmse  = {} 

for i in tqdm(range(2, 30)):
    dermod, derrmse, derr2, derr2adj = pls(tderdf.loc[qcix], tderdf.iloc[:, 0:-2].columns, val, i, plot=False)
    r2[i]    = derr2
    adjr2[i] = derr2adj
    rmse[i]  = derrmse
    
plt.plot(r2.keys(), r2.values(), label = 'R2')
plt.plot(adjr2.keys(), adjr2.values(), label = 'adjusted R2')
plt.legend()
plt.title('Second Derivative Data')
plt.show()

plt.plot(rmse.keys(), rmse.values(), label = 'RMSE')
plt.legend()
plt.title('Second Derivative Data')
plt.show()

#!!!!!!!!! LASSO !!!!!!!!!

# Now trying a lasso approach
def lasso(data, x, y, plot = True, name = ''):    
    
    # define predictor and response variables
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    
    # define cross-validation method
    cv = LeaveOneOut()
    
    # define model
    lasso = LassoCV(fit_intercept = True,
                    max_iter = 20000,
                    cv = cv,
                    n_jobs = -1)
    
    # run the model
    model = lasso.fit(X, Y)
    
    # now get r2
    r2 = model.score(X, Y)
    
    # get # of coefficients from Lasso
    expvar = np.count_nonzero(model.coef_)
    
    # calculate adjusted r2
    dof = (len(X) - 1)/(len(X) - expvar)
    adj_r2 = 1-(1-r2) * dof
    
    # calculate RMSE
    mse  = np.sum(np.square(model.predict(X).flatten()-Y))/len(Y)
    rmse = np.sqrt(mse)
        
    if plot:
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [7, 6])
        
        # Scatter plot
        ax.scatter(model.predict(X), Y, marker = 'o', color = 'k', alpha=0.7)
        
        # Lines depicting fit and 0.6 real interpretive difference
        ax.plot(Y,     Y, '--k', linewidth = 1)
        ax.plot(Y, Y+0.6, ':k' , linewidth = 1)
        ax.plot(Y, Y-0.6, ':k' , linewidth = 1)
        
        # Fix axis scale, set title, add labels
        ax.set_title('Correlation of IRIS to IRSM ' + name, family= 'serif')
        ax.set_xlabel(r'IRIS $\delta^{13}$C', family='serif', fontsize = 12)
        ax.set_ylabel(r'IRMS $\delta^{13}$C', family = 'serif', fontsize = 12)
        ax.set_xlim([min(Y) - .05, max(Y) + .05])
        ax.set_ylim([min(Y) - .05, max(Y) + .05])
        
        # Set minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # plt.savefig(direct + 'Lasso_results_wBands.png', bbox_inches='tight', dpi = 300 )
        # show figure and clear for next
        plt.show()
    
    return model, rmse, r2, adj_r2
    
# run the model for each df using d13C first and compare RMSE and R2
# now try to make the model only with the "good" and "great" spectra
qcix = sampdf[sampdf['QC'].isin(('good', 'great'))].index

# run the model for each df using d13C first and compare RMSE
val = 'd13C' # which isotopic value is being analyzed

'''
not working on original uncorrected data or on second derivative data
orgmod, orgrmse, orgr2, orgr2adj = lasso(tdf.loc[qcix], 
                                         tdf.iloc[:, 0:-2].columns, 
                                         val, 
                                         plot=True, 
                                         name = 'Original')

dermod, derrmse, derr2, derr2adj = lasso(tderdf.loc[qcix], 
                                         tderdf.iloc[:, 0:-2].columns, 
                                         val, plot=True, 
                                         name = 'Derivative')
'''

cormod, corrmse, corr2, corr2adj = lasso(tcordf.loc[qcix], 
                                         tcordf.iloc[:, 0:-2].columns, 
                                         val, 
                                         plot=True, 
                                         name = 'Corrected')


# get duplicated spectra but only select the spectra that are "good" from the
# duplicated directory and make those the test data
# now list the folders in the duplicates directory to open each file
dupfiles = os.listdir(direct + dupdir + 'good\\') # create list of filenames in folder

# read all duplicate files into a single dataframe, column names are file names
dupdf = pd.concat(map(lambda file: pd.read_csv(direct + dupdir + 'good\\' +  file, 
                                            skiprows=1, 
                                            header=0, 
                                            names=['wavelength', 
                                                   file.strip('.txt')]), 
                                                                     dupfiles),
                                                                      axis = 1)

# combine all wavelength columns and make wavelength column the index
dupdf = dupdf.loc[:,~dupdf.columns.duplicated()].copy()

dupdf.set_index('wavelength', inplace=True)

# cut df to fingerprint region of the spectrum (removing the first bit that has too much fluorescence)
dupdf = dupdf.loc[775:1900]

# strip column names of integration and scan numbers
pattern = r'\(\d-\d{2}\)(\(\d\))?'

dupdf.columns = (re.sub(pattern, '', col) for col in dupdf.columns)

# Subtract minimum from each column to remove negative values
for col in dupdf.columns:
    dupdf[col] = dupdf[col] - dupdf[col].min()
    
# get the corrected baseline for all duplicated spectra in new df
dupcordf = pd.concat(map(lambda col: arPLS_baseline(dupdf[col], 
                                                  ratio = 1e-6, 
                                                  lam=1e4, 
                                                  niter=5500, 
                                                  full_output=True)[1].rename(col), 
                       tqdm(dupdf.columns)), 
                   axis=1)

# to keep consistency, all values will be normalized to their highest peak
dupcordf = dupcordf/dupcordf.max()

# get locations for PO4 and CO3 peaks
dupcorpeaks = {}
for col in dupcordf.columns:
    dupcorpeaks[col] = peakfind(dupcordf[col], plot=False)
    
# remove the index name
dupcordf.index.name = None

# transpose to join on sample number
tdupcordf = dupcordf.T

# concat the peak locations previously found to the transposed dfs
tdupcordf = tdupcordf.join(pd.DataFrame.from_dict(dupcorpeaks, 
                                        orient='index', 
                                        columns=['PO4peak', 'CO3peak']))

# define training data (original used for model) and testing data (duplicates)
traindatx =  tcordf.loc[qcix].iloc[:, 0:-2].to_numpy()
traindaty =  tcordf.loc[qcix][val]

testdatx  = tdupcordf.to_numpy()
testdaty  = sampdf[val].loc[tdupcordf.index]

# Create subplot
fig, ax = plt.subplots(1, figsize= [7, 6])

# Scatter plots
ax.scatter(cormod.predict(testdatx), testdaty, marker = 'o', color = 'tab:orange', alpha=0.7)
ax.scatter(cormod.predict(traindatx), traindaty, marker = 'o', color = 'k', alpha=0.7)

# Lines depicting fit and 0.6 real interpretive difference
ax.plot(traindaty,     traindaty, '--k', linewidth = 1)
ax.plot(traindaty, traindaty+0.6, ':k' , linewidth = 1)
ax.plot(traindaty, traindaty-0.6, ':k' , linewidth = 1)

# Fix axis scale, set title, add labels
ax.set_title('Correlation of IRIS to IRSM ' + 'Corrected', family= 'serif')
ax.set_xlabel(r'IRIS $\delta^{13}$C', family='serif', fontsize = 12)
ax.set_ylabel(r'IRMS $\delta^{13}$C', family = 'serif', fontsize = 12)
ax.set_xlim([min(traindaty) - .05, max(traindaty) + .05])
ax.set_ylim([min(traindaty) - .05, max(traindaty) + .05])

# Set minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# show figure and clear for next
plt.show()

# figure out automated selection of scan, first look at rough boxplots
for col in ['CIraman','FluorRatio','AmIPO4','CO3-PO4','% Yield','%CaCO3']:
    sampdf.boxplot(column=col, by='QC')
    
# correlations appear to be on FluorRatio, CO3-PO4, AmIPO4

# LASSO CV
# now implement the Lasso without standard scaling
def lassocv(data, x, y, plot = True, name = '', split=True):    
    
    # define predictor and response variables
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    
    # train/test split if split == True
    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            random_state = 0,
                                                            test_size = 0.2)
    else:
        X_train = X
        Y_train = Y
    
    # Apply StandardScaler to get unit variance and 0 mean
    scaler = StandardScaler(with_mean = False)
    
    X_train = scaler.fit_transform(X_train)
    
    if split:
        X_test = scaler.fit_transform(X_test)
    
    # define cross-validation method
    # cv = LeaveOneOut()
    cv = RepeatedKFold(n_splits=10, n_repeats=100)
    
    # define model
    lasso = LassoCV(fit_intercept = True,
                    max_iter = 20000,
                    cv = cv,
                    n_jobs = -1)
    
    # run the model
    model = lasso.fit(X_train, Y_train)
    
    # now get r2 for training
    r2_train = model.score(X_train, Y_train)
    
    # get # of coefficients from Lasso
    expvar = np.count_nonzero(model.coef_)
    
    # calculate adjusted r2 for training data
    dof_train = (len(X_train) - 1)/(len(X_train) - expvar)
    adjr2_train = 1-((1-r2_train) * dof_train)
    
    # calculate RMSE of training data
    mse_train  = np.sum(np.square(model.predict(X_train).flatten()-Y_train))/len(Y)
    rmse_train = np.sqrt(mse_train)
    
    if split:
        # calculate R2 and rmse for test data
        # r2 for testing data
        r2_test = model.score(X_test, Y_test)
                
        # calculate RMSE of testing data
        mse_test   = np.sum(np.square(model.predict(X_test).flatten()-Y_test))/len(Y)
        rmse_test = np.sqrt(mse_test)
        
        # calculate mean absolute error for testing data
        mae = np.mean(np.abs(model.predict(X_test).flatten()-Y_test))
        
        print('Model with Train/Test Split at 10%\n'     +\
              f'R2 of training data:   {r2_train}\n'     +\
              f'Adjusted R2:           {adjr2_train}\n'  +\
              f'RMSE of training data: {rmse_train}\n'   +\
              '\n'  +\
              f'R2 of testing data:   {r2_test}\n'     +\
              f'RMSE of testing data: {rmse_test}\n'   +\
              f'MAE of testing data:  {mae}'           +\
              '\n' +\
              f'# of Coefficients:     {expvar}')
           
        
    if plot:        
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [7, 6])
        
        # Scatter plot
        ax.scatter(model.predict(X_train), Y_train, marker = 'o', 
                                                    color  = 'k', 
                                                    alpha  = 0.7)
        # ax.scatter(model.predict(X_test),  Y_test,  marker = 'o', 
                                                    #color  = 'tab:orange', 
                                                    #alpha  = 0.7)
                                                    
        # add annotations of sample size, r2, adjr2, and rmse
        ax.annotate(f'N       =  {len(Y_train)}',        
                                    (.8, .244),  
                                    xycoords = 'subfigure fraction')
        ax.annotate(f'R$^2$      = {r2_train: .2f}',        
                                    (.8, .211),  
                                    xycoords = 'subfigure fraction')
        ax.annotate('R$^2_{adj}$    =' + f' {adjr2_train: .2f}', 
                                    (.8, .178),  
                                    xycoords = 'subfigure fraction')
        ax.annotate(f'RMSE = {rmse_train: .2f}',       
                                    (.8, .145),  
                                    xycoords = 'subfigure fraction')
        
        
        # Lines depicting fit and 0.6 real interpretive difference
        corline_x = np.linspace(min(Y), max(Y), len(Y))
        corline_y = corline_x
        
        ax.plot(corline_x, corline_y,      '--k',  linewidth = 1)
        ax.plot(corline_x, corline_y + 0.6, ':k' , linewidth = 1)
        ax.plot(corline_x, corline_y - 0.6, ':k' , linewidth = 1)
        
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
        
        # plt.savefig(figout + 'd18Opdb_Lasso_results_wBands.png', bbox_inches='tight', dpi = 300 )
        # show figure and clear for next
        plt.show()
    
    return model, rmse_train, r2_train, adjr2_train


d13cmod, d13crmse, d13cr2, d13cadjr2 = lassocv(good_tcordf, x_cols, y_cols[0], 
                                             name = '$\delta^{13}C_{apatite}$',
                                             split = True)