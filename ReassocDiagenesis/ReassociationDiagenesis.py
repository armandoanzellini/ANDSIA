# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:22:12 2022

@author: Armando Anzellini
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import spearmanr, pearsonr, describe
from collections import namedtuple
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import RepeatedKFold
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# define directory paths
direct = "D:\\Users\\Armando\\OneDrive\\Documents\\Academic\\Dissertation\\"

utdir = "UT Collection\\"

ppdir = "Patakfalva-Papdomb\\"

resdir = "Results\\"

# fixed the spreadsheets by hand and then re-uploaded
# upload combinatorial spreadsheets as they are output by UTDonorsAnalysis and PapdombAnalysis
utdf = pd.read_excel(direct + resdir + 'UTCombDiff.xlsx', index_col = 0)

ppdf = pd.read_excel(direct + resdir + 'PPCombDiff.xlsx', index_col = 0)

# add 1 for every time ind1 and ind2 match in new column 'same' for UT, 0 for not
utdf['same'] = (utdf['ind1'] == utdf['ind2']).astype(float)

# fill in 0 wherever they are not the same indivdiual for both pp and ut
ppdf['same'] = ppdf['same'].fillna(0.0)

# get Sensitivity, Specificity, PPV, and NPV for each sample separately
# first get true and false positives
def pred_metrics(comb_df, threshold):
    res = namedtuple('Pred_Metrics', ('sensitivity', 'specificity', 'ppv', 'npv'))
    
    countspos = comb_df[comb_df['diff'] > threshold].groupby('same')['diff'].count()
    
    if len(countspos) < 2:
        test1 = [0., countspos.index[0],  countspos.values[0]]
        test2 = [0., 1-countspos.index[0],         0         ]
    elif len(countspos) == 2:
        test1 = [0., countspos.index[0], countspos.values[0]]
        test2 = [0., countspos.index[1], countspos.values[1]]
    
    if sum(test1[:2]) == 0:
        truepos  = test1[-1]
        falsepos = test2[-1]
    elif sum(test1[:2]) == 1:
        truepos  = test2[-1]
        falsepos = test1[-1]
        
    # now get true and false negatives
    countsnegs = comb_df[comb_df['diff'] < threshold].groupby('same')['diff'].count()
    
    if len(countsnegs) < 2:
        test1 = [1., countsnegs.index[0],  countsnegs.values[0]]
        test2 = [1., 1-countsnegs.index[0],         0         ]
    elif len(countsnegs) == 2:
        test1 = [1., countsnegs.index[0], countsnegs.values[0]]
        test2 = [1., countsnegs.index[1], countsnegs.values[1]]
        
    if sum(test1[:2]) == 0:
        trueneg  = test1[-1]
        falseneg = test2[-1]
    elif sum(test1[:2]) == 1:
        falseneg = test1[-1]
        trueneg  = test2[-1]
        
    sensitivity = (truepos/(truepos + falseneg)) * 100
    specificity = (trueneg/(trueneg + falsepos)) * 100
    ppv = (truepos/(truepos + falsepos)) * 100
    npv = (trueneg/(trueneg + falseneg)) * 100

    return res(sensitivity, specificity, ppv, npv)


utmetrics_99 = pred_metrics(utdf, 1.9)
utmetrics_95 = pred_metrics(utdf, 1.55)

ppmetrics_99 = pred_metrics(ppdf, 1.9)
ppmetrics_95 = pred_metrics(ppdf, 1.55)

# create a logistic regression for probabilities of belonging to the same individual



# do confusion matrix to get prediction metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_pp = ppdf['diff'].to_numpy()
Y_pp = ppdf['same'].to_numpy()

X_ut = utdf['diff'].to_numpy()
Y_ut = utdf['same'].to_numpy()
 
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pp, Y_pp)

Xu_train, Xu_test, yu_train, yu_test = train_test_split(X_ut, Y_ut)

cv = RepeatedKFold(n_splits=10, n_repeats = 100)

logreg = LogisticRegressionCV(cv=cv, max_iter = 1000, n_jobs = -1)

ppmodel = logreg.fit(Xp_train.reshape(-1, 1), yp_train)
utmodel = logreg.fit(Xu_train.reshape(-1, 1), yu_train)

# check accuracy
accpp = ppmodel.score(Xp_test.reshape(-1, 1), yp_test)
accut = utmodel.score(Xu_test.reshape(-1, 1), yu_test)

# get confusion matrix
cmut = confusion_matrix(yu_test, utmodel.predict(Xu_test.reshape(-1, 1)))
cmpp = confusion_matrix(yp_test, utmodel.predict(Xp_test.reshape(-1, 1)))


def pred_mets_cm(cm):
    res = namedtuple('Pred_MetricsCM', ('sensitivity', 'specificity', 'ppv', 'npv'))
    
    truepos = cm[0,0]
    falsepos = cm[1,0]
    trueneg = cm[0,1]
    falseneg = cm[1,1]
    
    sensitivity = (truepos/(truepos + falseneg)) * 100
    specificity = (trueneg/(trueneg + falsepos)) * 100
    ppv = (truepos/(truepos + falsepos)) * 100
    npv = (trueneg/(trueneg + falseneg)) * 100
    
    return res(sensitivity, specificity, ppv, npv)


# upload the isotope and diagenesis values and test for correlations, and plot
ppisodf = pd.read_excel(direct + resdir + 'PatakfalvaUnassociatedIsotope_Anzellini.xlsx', 
                        sheet_name = 'All',
                        index_col = 0)

ppisodf = ppisodf.iloc[:-32] # select only the non-duplicated spectra

ppisodf = ppisodf.loc[ppisodf['QC'].isin(['good', 'great'])] # select good+ qc

utisodf = pd.read_excel(direct + resdir + 'UTDonorIsotopeValues.xlsx', 
                        sheet_name = 0,
                        index_col = [0, 1])

# Now only select the useful columns for both
ppisodf = ppisodf[['CIraman','AmIPO4', 'CO3-PO4', 'd13Cap', 'd18Ovpdb']]

utisodf = utisodf[['CIraman','AmIPO4', 'CO3-PO4', 'd13C']]

# check correlation on each alone (IRMS and IRIS separately)
# get descriptive stats for all non isotope data columns in each df
ppstats = []
for col in ppisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    ppstats += [col, describe(ppisodf[col])]
    

utstats = []
for col in utisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    utstats += [col, describe(utisodf[col])]
    
# UT stats shows some values above desired levels
ami_out = utisodf.loc[utisodf['AmIPO4']  > 19.4].count()[0]
cp_out  = utisodf.loc[(utisodf['CO3-PO4'] < 0.23) | (utisodf['CO3-PO4'] > 0.4)].count()[0]


# check outlier detection
def out_detect(data, col):
    std  = data[col].std()
    mean = data[col].mean()
    out  = data[col].loc[abs(data[col] - mean) > std*3.2]
    if out.empty:
        print('No outlier detected in the distribution')
        return []
    if not out.empty:
        print('Outliers were detected in the distribution')
        return out.index


for col in ppisodf:
    out_detect(ppisodf, col)
    
# no outliers detected in ppisodf, so checking pearsonsr
ppcorr13C = []
for col in ppisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    ppcorr13C += [col, pearsonr(ppisodf['d13Cap'], ppisodf[col])]
    
ppcorr18O = []
for col in ppisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    ppcorr18O += [col, pearsonr(ppisodf['d18Ovpdb'], ppisodf[col])]

outliers = []
    
for col in utisodf:
    outliers += [col, out_detect(utisodf, col)]

# due to the presence of outliers in the UT data, a test of monotnicity
# rather than linear correlation is used, Spearman's rho
utcorr = []
for col in utisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    utcorr += [col, spearmanr(utisodf['d13C'], utisodf[col])]
    

# since indices outside desired levels, testing them separately
utincorr_ami  = []
utoutcorr_ami = []
for col in utisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    utincorr_ami  += [col, spearmanr(utisodf.loc[utisodf['AmIPO4']  < 19.4]['d13C'],
                     utisodf.loc[utisodf['AmIPO4']  < 19.4][col])]
    utoutcorr_ami += [col, spearmanr(utisodf.loc[utisodf['AmIPO4']  > 19.4]['d13C'],
                     utisodf.loc[utisodf['AmIPO4']  > 19.4][col])]
    
cp_outdf = utisodf.loc[(utisodf['CO3-PO4'] < 0.23) | (utisodf['CO3-PO4'] > 0.4)]
cp_indf  = utisodf.loc[(utisodf['CO3-PO4'] >= 0.23) & (utisodf['CO3-PO4'] <= 0.4)]

utincorr_cp  = []
utoutcorr_cp = []
for col in utisodf[['CIraman','AmIPO4', 'CO3-PO4']]:
    utincorr_cp  += [col, spearmanr(cp_indf['d13C'],  cp_indf[col])]
    utoutcorr_cp += [col, spearmanr(cp_outdf['d13C'], cp_outdf[col])]

