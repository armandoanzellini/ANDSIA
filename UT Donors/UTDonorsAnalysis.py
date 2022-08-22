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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy import stats
from scipy import signal
from scipy import sparse
from joblib import load
from numpy.linalg import norm
from collections import Counter
from collections import namedtuple
from scipy.sparse import linalg
from sklearn.decomposition import PCA
from itertools import combinations
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# define directory paths
direct = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\"

samdir = "UT Collection\\"

donors = "Donors Raman\\"

dups   = donors + "Duplicates\\"

# load linear regression models and training data
d13cmod = load(direct + 'Models\\' + 'd13c_model.joblib')
# d18omod = load(direct + 'Models\\' + 'd18o_model.joblib') BAD MODEL

traindat = pd.read_excel(direct + 'Models\\' + 'TrainingData.xlsx', 
                         index_col = 0).iloc[:,:-2] # remove independent vars

# open non-raman data
donordata = pd.read_excel(direct + samdir + "DonorResidence18OReducedPrioritized_UTID.xlsx")

# clean up data making UTID the index and dropping unnecessary columns
donordata.dropna(inplace=True)

dropcols = ['UTID', 'Country', 'Lat', 'Long'] # columns not necessary

donordata.drop(dropcols, axis = 1, inplace=True)

# donordata['UTID'] = [x.lstrip('UT') for x in donordata['UTID']]

donordata.set_index('KEY', inplace=True) # set UTID as index for future relates

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
df    = df.loc[775:1900]
dupdf = dupdf.loc[775:1900] 

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

cordpdf = pd.concat(map(lambda col: arPLS_baseline(dupdf[col], 
                                              ratio = 1e-6, 
                                              lam=1e4, 
                                              niter=5500, 
                                              full_output=True)[1].rename(col), 
                       tqdm(dupdf.columns)), 
                   axis=1)

# normalize all data in cordf
cordf   = cordf/cordf.max()

cordpdf = cordpdf/cordpdf.max() 

# check that duplicates and originals closely correlate after correction
# check Raman duplications for precision
dupdons = list(set([i[0] for i in dupdf.columns])) # list ID for duplicated donors

duporgs = cordf[dupdons] # get orginal data for duplicated donors

spearmandict = {} 
for col in duporgs.columns: 
    rho, p = stats.spearmanr(duporgs[col], cordpdf[col])
    spearmandict[col] = (rho, p)

spearmandf = pd.DataFrame.from_dict(spearmandict, 
                                   columns = ['rho', 'p-value'], 
                                   orient = 'index')

spearmandf.index = pd.MultiIndex.from_tuples(spearmandf.index, 
                                             names = ('Individual',
                                                      'Element and Location'))

# spearmandf.to_excel(direct + 'Results\\' + 'DupSpearmanResults.xlsx')

# figure out QC criteria for each original spectrum
def qc(data, col):
    # find creterion peak values
    peak1val = data[col].loc[800:850].max()
    peak2val = data[col].loc[850:925].max()
    
    # now categorize
    if   peak1val <= 0.075 and peak2val <= 0.075:
        cat = 'great'
    elif peak1val > 0.3 or peak2val > 0.3:
        cat = 'bad'
    elif peak1val > 0.15 and peak2val > 0.15:
        cat = 'bad'
    elif peak1val <= 0.15 and peak2val <= 0.15:
        cat = 'good'
    else:
        cat = 'fair'
        
    return cat

qcdict = {}
for col in cordf.columns:
    qcdict[col] = qc(cordf, col)
    
dpqcdict = {}
for col in cordpdf.columns:
    dpqcdict[col] = qc(cordpdf, col)

# add blank rows for peak locations since model doesn't use them but will
# expect them to keep track of x-val location
cordf.loc['PO4 Peak'] = 0
cordf.loc['CO3 Peak'] = 0

cordpdf.loc['PO4 Peak'] = 0
cordpdf.loc['CO3 Peak'] = 0

# now select only the scans that were of at least good quality
scans   = [key for key,val in qcdict.items() if val == 'good' or val == 'great']

dpscans = [key for key,val in dpqcdict.items() if val == 'good' or val == 'great']

gooddf  = cordf[scans]

gooddpdf = cordpdf[dpscans]


# Now calculate isotope values for both originals and duplicates
# first implement standard scaler prior to analysis (unit variance)
scaler = StandardScaler(with_mean = False).fit(traindat.to_numpy())

scaledf = pd.DataFrame(scaler.transform(gooddf.T.to_numpy()), 
                       columns=gooddf.T.columns, 
                       index=gooddf.T.index)

d13cvals = {}
for ix in tqdm(scaledf.index):
    d13cval = d13cmod.predict(scaledf.loc[ix].to_numpy().reshape(1,-1))
    d13cvals[ix] = d13cval[0]

# now convert into pandas Series for ease of analysis
cisovals = pd.Series(d13cvals, name = 'd13C')

cisovals.index.names = ['Individual', 'Element'] # add index names

# combine in new df with diagenesis indices and export to excel
# first make cisovals a dataframe
cisodf = pd.DataFrame(cisovals)

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
for col in tqdm(gooddf.columns):
    cisodf.loc[col,'CIraman'] = ci_raman(gooddf[col], 
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
for col in tqdm(gooddf.columns):
    cisodf.loc[col,'AmIPO4'] = amIPO4raman(gooddf[col], 
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
for col in gooddf.columns:
    corpeaks[col] = peakfind(gooddf[col], plot=False)
    
# get CO3 peak value which now is equivalent to CO3/PO4 ratio, add to sampdf
for key, val in corpeaks.items():
    ap, ac = gooddf[key].loc[[val[0], val[1]]]
    # since the ratio of val at c by val at p has already been normalized
    cisodf.loc[key, 'CO3-PO4'] = ac 



# cisodf.to_excel(direct + 'Results\\' + 'UTDonorIsotopeValues.xlsx')

# GET VALUES GROUPED BY INDIVIDUALS
cisodat = {col_name:col for col_name, col in cisovals.groupby('Individual')}

# check for equality of variances
lev_ind = stats.levene(*cisodat.values())
# LeveneResult(statistic=2.1014615401392946, pvalue=0.0008404607125538064)
if lev_ind[1] < 0.05:
    print('\nData Fails Equal Variances Test\n')

def anova(data):
    ANOVA_Results = namedtuple('ANOVA_OneWay', ('statistic', 'pvalue', 'eta2'))
    
    groups = [key for key in data.keys()]
    
    k = len(groups)
    
    sum_tot  = sum([sum(data[i].values) for i in groups])
    n_t      = sum([len(data[i]) for i in groups])
    
    mean_tot = sum_tot/n_t
    
    ssw = 0
    ssb = 0
    for i in groups:
        groupmean = np.mean(data[i].values)
        err = data[i].values - groupmean
        ssk = np.sum(np.square(err))
        ssw += ssk
        
        n_k  = len(data[i])
        sumb = n_k* np.square((groupmean - mean_tot))
        ssb += sumb
        
        
    sst = ssw + ssb
    
    dofb = k   - 1
    dofw = n_t - k
    doft = n_t - 1
    
    msb = ssb/dofb
    msw = ssw/dofw
    
    f = msb/msw
    
    p = stats.f.sf(f, dofb, dofw)
    
    eta2 = ssb/sst
    
    return ANOVA_Results(f, p, eta2)
    
def welch_anova(data):
    
    Welch_AnovaResult = namedtuple('Welch_Anova', ('statistic', 'pvalue', 'omega2'))
    
    groups = [key for key in data.keys()]
    
    k = len(groups)
    
    nj    = [len(data[i]) for i in groups]
    sj    = [np.var(data[i]) for i in groups]
    xbarj = [np.mean(data[i]) for i in groups]
    wj    = np.array(nj)/np.array(sj)
    
    w = np.sum(wj)
    
    adj_xbar = np.sum(wj*xbarj)/w
    
    tmp = sum((1-wj/w)**2 / (np.array(nj)-1))
    tmp /= (k**2 -1)
    
    
    dfb = k - 1
    dfw = 1 / (3* tmp)
    
    num = np.sum(wj*(np.array(xbarj) - adj_xbar)**2)/dfb
    denom = 1 + (2*(k-2))*tmp
    
    f = num/denom
    
    p = stats.f.sf(f, dfb, dfw)
    
    omega2 = (dfb*(f-1))/(dfb*(f-1) + np.sum(nj))
    
    return Welch_AnovaResult(f, p, omega2)
    
# since data failed the homoscedasticity test, using Welch's
waf_ind = welch_anova(cisodat)

# NOW GROUP BY LOCATION COMPARE BETWEEN INDIVIDUALS
cisodatloc = {col_name:col for col_name, col in cisovals.groupby('Element')}

# check for equality of variances
lev_loc = stats.levene(*cisodatloc.values())
# LeveneResult(statistic=2.1014615401392946, pvalue=0.0008404607125538064)
if lev_loc[1] < 0.05:
    print('\nData Fails Equal Variances Test\n')

# passes homoscedasticity test, using anova
anf_loc = anova(cisodatloc)

# NOW GROUP BY BONE COMPARE B\W INDIVIDUALS, regardless of scan location
bonevals = cisovals.copy()

bonevals = bonevals.reset_index(level='Element')

bonevals = pd.concat([bonevals['d13C'], bonevals['Element'].str.split(r"(\d)", 
                                                              expand = True)],
                     axis = 1)

# rename the split columns
bonevals.rename(columns = {0: 'Element', 1 : 'location'}, inplace = True)

bonevals.drop(columns=2, inplace = True)

cisodatbone = {col_name:col for col_name, col in bonevals.groupby('Element')['d13C']}

# check for equality of variances
lev_bone = stats.levene(*cisodatbone.values())

# LeveneResult(statistic=2.1014615401392946, pvalue=0.0008404607125538064)
if lev_bone[1] < 0.05:
    print('\nData Fails Equal Variances Test\n')

# passes homoscedasticity test, using anova
anf_bone = anova(cisodatbone)

# NOW GROUP BY LOCATION REGARDLESS OF ELEMENT (checking sup-inf, post-ant)
cisoboneloc = {col_name:col for col_name, col in bonevals.groupby('location')['d13C']}

# check for equality of variances
lev_boneloc = stats.levene(*cisoboneloc.values())

# LeveneResult(statistic=2.1014615401392946, pvalue=0.0008404607125538064)
if lev_boneloc[1] < 0.05:
    print('\nData Fails Equal Variances Test\n')

# passes homoscedasticity test, using ANOVA
anf_boneloc = anova(cisoboneloc)


# now check the mean deviation and range within individuals
cisoranges  = {}
cisomeandev = {}
for i in cisodat.keys():
    # get ranges for each indivdiual
    cisoranges[i] = max(cisodat[i]) - min(cisodat[i])
    # calculate mean deviation for each indivdiual
    n    = len(cisodat[i])
    mean = np.mean(cisodat[i])
    meandev = np.sum(np.abs(cisodat[i].values - mean))/n
    cisomeandev[i] = meandev


# output mean deviation and ranges as excel
devdf = pd.concat(map(lambda d: pd.Series(d), 
                      [cisoranges, cisomeandev]), 
                  axis = 1)

devdf.rename(columns = {0 : 'Maximum Deviation', 1 : 'Mean Deviation'}, 
             inplace = True)

# devdf.to_excel(direct + 'Results\\' + 'individualDeviationResults.xlsx')

# values are not too dissimilar from mean (.4 per mil mean dev is max)
# but ranges are relatively large (up to 2.9 per mil)

# now regroup into skeletal element and comparewithin each elment regardless 
# of individual

# first get deviation from mean at each sampling location for each bone
# separate bones into their own dfs
femurix   = [j for j in cisovals.index if 'Femur'   in j[1]]
fibulaix  = [j for j in cisovals.index if 'Fibula'  in j[1]]
humerusix = [j for j in cisovals.index if 'Humerus' in j[1]]
radiusix  = [j for j in cisovals.index if 'Radius'  in j[1]]
ribix     = [j for j in cisovals.index if 'Rib'     in j[1]]
tibiaix   = [j for j in cisovals.index if 'Tibia'   in j[1]]
ulnaix    = [j for j in cisovals.index if 'Ulna'    in j[1]]

# make each series a dataframe and add a column of scan location
cisovals.rename('d13C', inplace=True) #name the series prior to DF creation

femdf = pd.DataFrame(cisovals.loc[femurix]).reset_index(level='Element')
fibdf = pd.DataFrame(cisovals.loc[fibulaix]).reset_index(level='Element')
humdf = pd.DataFrame(cisovals.loc[humerusix]).reset_index(level='Element')
raddf = pd.DataFrame(cisovals.loc[radiusix]).reset_index(level='Element')
ribdf = pd.DataFrame(cisovals.loc[ribix]).reset_index(level='Element')
tibdf = pd.DataFrame(cisovals.loc[tibiaix]).reset_index(level='Element')
ulndf = pd.DataFrame(cisovals.loc[ulnaix]).reset_index(level='Element')

# remove skeletal element name
femdf['Element'] = femdf['Element'].str[-1]
fibdf['Element'] = fibdf['Element'].str[-1]
humdf['Element'] = humdf['Element'].str[-1]
raddf['Element'] = raddf['Element'].str[-1]
ribdf['Element'] = ribdf['Element'].str[-1]
tibdf['Element'] = tibdf['Element'].str[-1]
ulndf['Element'] = ulndf['Element'].str[-1]

# rename column for clarity
femdf.rename(columns = {'Element':'location'}, inplace = True)
fibdf.rename(columns = {'Element':'location'}, inplace = True)
humdf.rename(columns = {'Element':'location'}, inplace = True)
raddf.rename(columns = {'Element':'location'}, inplace = True)
ribdf.rename(columns = {'Element':'location'}, inplace = True)
tibdf.rename(columns = {'Element':'location'}, inplace = True)
ulndf.rename(columns = {'Element':'location'}, inplace = True)

# get mean per individual
femdf['Ind13Cmean'] = femdf.groupby('Individual').mean()
fibdf['Ind13Cmean'] = fibdf.groupby('Individual').mean()
humdf['Ind13Cmean'] = humdf.groupby('Individual').mean()
raddf['Ind13Cmean'] = raddf.groupby('Individual').mean()
ribdf['Ind13Cmean'] = ribdf.groupby('Individual').mean()
tibdf['Ind13Cmean'] = tibdf.groupby('Individual').mean()
ulndf['Ind13Cmean'] = ulndf.groupby('Individual').mean()


# get standard deviation per individual
femdf['Ind13Cstd'] = femdf['d13C'].groupby('Individual').std()
fibdf['Ind13Cstd'] = fibdf['d13C'].groupby('Individual').std()
humdf['Ind13Cstd'] = humdf['d13C'].groupby('Individual').std()
raddf['Ind13Cstd'] = raddf['d13C'].groupby('Individual').std()
ribdf['Ind13Cstd'] = ribdf['d13C'].groupby('Individual').std()
tibdf['Ind13Cstd'] = tibdf['d13C'].groupby('Individual').std()
ulndf['Ind13Cstd'] = ulndf['d13C'].groupby('Individual').std()


# subtract across columns to get deviation value
femdf['deviation'] = femdf['d13C'] - femdf['Ind13Cmean']
fibdf['deviation'] = fibdf['d13C'] - fibdf['Ind13Cmean']
humdf['deviation'] = humdf['d13C'] - humdf['Ind13Cmean']
raddf['deviation'] = raddf['d13C'] - raddf['Ind13Cmean']
ribdf['deviation'] = ribdf['d13C'] - ribdf['Ind13Cmean']
tibdf['deviation'] = tibdf['d13C'] - tibdf['Ind13Cmean']
ulndf['deviation'] = ulndf['d13C'] - ulndf['Ind13Cmean']

# check if deviations are within 2 std or (95% CI)
femdf['95%CI'] = np.abs(femdf['deviation']) < (femdf['Ind13Cstd']*2)
fibdf['95%CI'] = np.abs(fibdf['deviation']) < (fibdf['Ind13Cstd']*2)
humdf['95%CI'] = np.abs(humdf['deviation']) < (humdf['Ind13Cstd']*2)
raddf['95%CI'] = np.abs(raddf['deviation']) < (raddf['Ind13Cstd']*2)
ribdf['95%CI'] = np.abs(ribdf['deviation']) < (ribdf['Ind13Cstd']*2)
tibdf['95%CI'] = np.abs(tibdf['deviation']) < (tibdf['Ind13Cstd']*2)
ulndf['95%CI'] = np.abs(ulndf['deviation']) < (ulndf['Ind13Cstd']*2)

# now find the mean grouped by location
femloc = femdf.groupby(['location'], dropna=True).mean()
fibloc = fibdf.groupby(['location'], dropna=True).mean()
humloc = humdf.groupby(['location'], dropna=True).mean()
radloc = raddf.groupby(['location'], dropna=True).mean()
ribloc = ribdf.groupby(['location'], dropna=True).mean()
tibloc = tibdf.groupby(['location'], dropna=True).mean()
ulnloc = ulndf.groupby(['location'], dropna=True).mean()


# find where 95% C.I. is false and why
femouts = femdf[femdf['95%CI'] == False]
fibouts = fibdf[fibdf['95%CI'] == False]
humouts = humdf[humdf['95%CI'] == False]
radouts = raddf[raddf['95%CI'] == False]
ribouts = ribdf[ribdf['95%CI'] == False]
tibouts = tibdf[tibdf['95%CI'] == False]
ulnouts = ulndf[ulndf['95%CI'] == False]

# Output tables to excel to create figures
'''
femloc.to_excel(direct + 'Results\\' + 'femloc.xlsx')
fibloc.to_excel(direct + 'Results\\' + 'fibloc.xlsx')
humloc.to_excel(direct + 'Results\\' + 'humloc.xlsx')
radloc.to_excel(direct + 'Results\\' + 'radloc.xlsx')
ribloc.to_excel(direct + 'Results\\' + 'ribloc.xlsx')
tibloc.to_excel(direct + 'Results\\' + 'tibloc.xlsx')
ulnloc.to_excel(direct + 'Results\\' + 'ulnloc.xlsx')
'''

# To test the threshold method (Berg et al., 2022) on the UT donors
# export the combination table of differences
reasdf = cisovals.reset_index().copy()


indices = list(combinations(reasdf['Individual'], 2))

vals    = list(combinations(reasdf['d13C'], 2))

combs = list(zip(indices, vals))

comarray = np.array(combs)

comarray.shape = (len(combs), 4)


comdf = pd.DataFrame(comarray, columns = ['ind1', 'ind2', 'val1', 'val2'])

comdf[['val1', 'val2']] = comdf[['val1', 'val2']].astype(float)

comdf['diff'] = abs(comdf['val1'] - comdf['val2'])

# export combinatorial difference table
comdf[['ind1', 'ind2', 'diff']].to_excel(direct + 'Results\\' + 'UTCombDiff.xlsx')

# due to bad models on d18O and to test the ANOVA result, have to take a different 
# approach using PCA (also good for re-association of individuals)
scaledf.reset_index(inplace=True)

scaledf.rename(columns = {'level_0' : 'Individual', 'level_1' : 'location'}, 
               inplace = True)

# drop peak location columns for scaledf and prepare for clustering
clustdf = scaledf.iloc[:,:-2]

# get values of X and y to do PCA and test the clustering
X = clustdf.iloc[:,2:].to_numpy()
y_names = clustdf['Individual'].to_numpy()
target_names = set(y_names)

group_map = dict(zip(target_names, range(len(target_names))))

y = np.array(list(map(group_map.get, y_names)))

comps = 6

pca = PCA(n_components=comps)
X_r = pca.fit(X).transform(X)

evr = pca.explained_variance_ratio_

print(f'PCA - {comps} components explain: {sum(evr)*100:.2f}% of the variance')

###!!! Make Better Plot! 3D to account for at least 50% of variation
# Create subplot
fig, ax = plt.subplots(1, figsize= [6, 5])

# get colors and color vector
colors = dict(zip(target_names, mcolors.CSS4_COLORS.keys()))

cvec = [colors[i] for i in y_names]

# plot PCA scatter
ax.scatter(X_r[:, 0], X_r[:,1], c = cvec, alpha = 0.8)

# Fix axis scale, set title, add labels
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_title('PCA on Raman Spectra Grouped By Individual', family='serif')
ax.set_xlabel("PC1", family="serif",  fontsize=12)
ax.set_ylabel("PC2",family="serif",  fontsize=12)

# Set ticks and grid
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.grid()

plt.show()

# create LDA to show that even in the best circumstances, the clustering won't
# work very well
lda = LinearDiscriminantAnalysis()

X_a = lda.fit(X, y).transform(X)

# Create subplot
fig, ax = plt.subplots(1, figsize= [6, 5])

# get colors and color vector
colors = dict(zip(target_names, mcolors.CSS4_COLORS.keys()))

cvec = [colors[i] for i in y_names]

# plot PCA scatter
ax.scatter(X_a[:, 0], X_a[:,1], c = cvec, alpha = 0.8)

# Fix axis scale, set title, add labels
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_title('LDA on Raman Spectra Grouped By Individual', family='serif')
ax.set_xlabel("Loading 1", family="serif",  fontsize=12)
ax.set_ylabel("Loading 2",family="serif",  fontsize=12)

# Set ticks and grid
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.grid()

plt.show()

# try BayesianGaussianMixture Mmodel for finding clusters
# calculate best guess for number of groups
clest = len(Counter(scaledf['location'])) +1



bay_gmm = BayesianGaussianMixture(n_components=clest, n_init=100)

bay_gmm.fit(X)

bay_gmm_weights = bay_gmm.weights_
np.round(bay_gmm_weights, 2)

n_clusters_ = (np.round(bay_gmm_weights, 2) > 0).sum()
print('Estimated number of clusters: ' + str(n_clusters_))


y_pred  = bay_gmm.predict(X)


# use Adjusted Mutual Information because it adjusts Normalized Mutual
# Information for chance; see citations
ami(y, y_pred)

