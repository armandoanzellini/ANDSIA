# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:00:47 2021

@author: Armanco Anzellini

This code reads SPC data from a single RAMAN spectrum of bone and completes
post-processing on the spectrum to reveal underlying data by removing the
fluorescence.

Could also try with second derivative spectra and see if that provides better 
results and reduces the error possibly introduced by arPLS or polynomial 
baseline correction for IRViS
"""
import os
import re
import numpy as np
import scipy as sp
import pandas as pd
import scipy.signal as ss
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as nppoly
from scipy import sparse
from numpy.linalg import norm
from scipy.sparse import linalg
from pyspectra.readers.read_spc import read_spc_dir, read_spc
from matplotlib.ticker import AutoMinorLocator

direct = 'C:\\Users\\aanzellini\\OneDrive\\Documents\\Academic\\Dissertation\\Patakfalva-Papdomb\\'

filedir = 'Calibrations\\V1\\'

file    = 's1external-15-30-3pm-contactalcohol.spc'

# Read the spc files from the directory
class RamanRead():
    def averaged(self, df, plot = True):
        
        samp = re.match('[^-]*', df.columns[0])[0]
        
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
        
        # Average data together to see what comes out
        df[samp] = df.mean(axis=1)
            
        if plot:
            # Plot to see all scans together
            f, ax = plt.subplots(1, figsize=(15,5))
            ax.plot(df)
            ax.grid()
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_ylabel('Signal')
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            plt.show()
            
            # PLot either averaged spectrum
            # Create subplot
            fig, ax = plt.subplots(1, figsize= [15, 5])
            
            # Spectral plot
            ax.plot(df[samp], '-', color = 'k', linewidth=0.6)
            
            # Fix axis limits, set title, add labels
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_title('Average of all scans: ' + samp)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
            ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
            
            # Set minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            plt.show()
            
        return df
        
    def multifile_csv(self, directory, averaged = False, plot = False):
        
        files = os.listdir(directory)
        
        selfiles_txt = [f for f in files if f.endswith('.txt')]
        selfiles_csv = [f for f in files if f.endswith('.csv')]
        
        if selfiles_txt and selfiles_csv:
            raise Exception('Both TXT and CSV files present in directory')
            
        selfiles = selfiles_txt + selfiles_csv # add empty list to select not empty
        
        df = pd.concat(map(lambda file: pd.read_csv(file, 
                        skiprows = 2, 
                        names = ['index', re.sub('\.[a-z]{,3}', '', file)]), 
                list(map(lambda f: directory + f, [f for f in selfiles]))),
                axis=1)
        
        df = df.loc[:,~df.columns.duplicated()] #remove duplicated index columns
        
        # set index column as index and remove name
        df.set_index('index', inplace = True)
        
        # make sure index is read astype float64
        df.index = df.index.astype('float64')
        
        # remove name from index column for readability
        df.index.name = None
        
        # create dict of column names to remove directory path
        coldict = {col : col.replace(directory, '') for col in df.columns}
        
        # use coldict to rename columns for legibility
        df.rename(columns = coldict, inplace = True)
        
        # ensure minimum for each spectrum and subtract to bring baseline to 0
        df   = df.apply(lambda x: x-x.min())
        
        if averaged:
            df = self.averaged(df, plot = plot)
            
        return df    
    
    def multifile_spc(self, directory, averaged = False, plot = False):
        
        df_spc, dict_spc = read_spc_dir(directory)
        df = df_spc.transpose()
        
        # ensure minimum for each spectrum and subtract to bring baseline to zero
        df   = df.apply(lambda x: x-x.min())
        
        # change the name of the columns to remove the file extension for clarity
        # create dict of column names to remove directory path
        coldict = {col : re.sub('\.[a-z]{,3}', '', col) for col in df.columns}
        
        # use coldict to rename columns for legibility
        df.rename(columns = coldict, inplace = True)
        
        # ensure minimum for each spectrum and subtract to bring baseline to 0
        df   = df.apply(lambda x: x-x.min())
        
        if averaged:
            df = self.averaged(df, plot = plot)
        
        return df
    
    def singlefile(self, directory, file):
        fname = re.sub('\.[a-z]{,3}', '', file)
        
        if file.endswith(('.txt', '.csv')):
            # read .txt as a csv assuming header row
            df = pd.read_csv(directory + file, skiprows=1)
            # reset index to be wavenumbers
            df.set_index(df.columns[0], inplace = True)
            # change column name to name of file            
            df.rename(columns = {df.columns[0]:fname}, inplace = True)
            # remove name from index column for readability
            df.index.name = None
            
        elif file.endswith('.spc'):
            df_spc = read_spc(directory + file)
            df_spc.name = fname
            df = df_spc.to_frame()
        
        # ensure minimum for each spectrum and subtract to bring baseline to 0
        df   = df.apply(lambda x: x-x.min())
        
        return df
    
class RamanClean():
    def __init__(self, spec_sr):
        self.sr   = spec_sr[775:2000] # see notes
        self.samp = spec_sr.name
        
        
    def minima_baseline(self, sr, plot = False):
        samp = self.samp
        
        # try a non arPLS baseline, instead a baseline following minima
        minima = ss.find_peaks(-sr, distance=37)
        
        baseline = np.interp(sr.index, sr.iloc[minima[0]].index, sr.iloc[minima[0]])
        
        corr_bl = baseline - (max(baseline - sr))
        
        for i,val in enumerate(corr_bl):
            if val < 0:
                corr_bl[i] = 0.0
        
        # add new column to df with new baseline values and smooth
        corr_sr = sr - corr_bl
        
        if plot:
            # Create subplot
            fig, ax = plt.subplots(1, figsize= [15, 5])
            
            # Spectral plot
            ax.plot(corr_sr, '-', color = 'k', linewidth=0.6)
            
            # Fix axis scale, set title, add labels
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_title('Minima Baseline Correction: ' + samp)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
            ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
            
            # Set minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            plt.show()
        
        return corr_bl, corr_sr
    
    def polyfit_baseline(self, sr, plot = False):
        samp = self.samp
        
        # try a non arPLS baseline, instead a baseline following minima
        minima = ss.find_peaks(-sr, distance=37)
        
        polyfit = nppoly.Polynomial.fit(x = sr.iloc[minima[0]].index, 
                                         y = sr.iloc[minima[0]],
                                         deg = 3)
        
        # get coefficients of polyfit and create values at x
        coefs = polyfit.convert().coef
        
        baseline = np.polyval(coefs[::-1], sr.index)
        
        # make baseline a series and match it to index of sr
        base_sr = pd.Series(baseline)
        
        base_sr.index = sr.index
        
        # subtract baseline from data to get baseline correction
        corr_sr = sr - base_sr
        
        if plot:
            # Create subplot
            fig, ax = plt.subplots(1, figsize= [15, 5])
            
            # Spectral plot
            ax.plot(corr_sr, '-', color = 'k', linewidth=0.6)
            
            # Fix axis scale, set title, add labels
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_title('Polyfit Baseline Correction: ' + samp)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
            ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
            
            # Set minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            plt.show()
            
        return base_sr, corr_sr
     
    def arPLS_baseline(self, y, ratio=1e-6, lam=100, niter=10, full_output=False):
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
                print('Maximum number of iterations exceeded')
                break
    
        if full_output:
            info = {'num_iter': count, 'stop_criterion': crit}
            return z, d, info
        else:
            return z
    
    def oscillation_removal(self, algo, per = 'mean'):
        sr = self.sr

        if algo == 'arPLS':
            base, corrected, info = self.arPLS_baseline(sr, lam=1e4,
                                                         niter=5000,
                                                         full_output=True)
        elif algo == 'minima':
            base, corrected = self.minima_baseline(sr)
        elif algo == 'polyfit':
            base, corrected = self.polyfit_baseline(sr)
        else:
            Exception(f'Alogrithm does not have a {algo} type')
                
        # write a dampened sin wave to remove fluorescent oscillations
        # find the peak heights of oscillations
        peaklocs = ss.find_peaks(corrected, prominence=0.3)
        peaks = corrected.iloc[peaklocs[0]][:850]

        # get difference between peak locations to know oscillation period
        oscillation = [j-i for i, j in zip(peaks.index[:-1], peaks.index[1:])]

        # get mean of oscillation period
        muperiod = np.mean(oscillation)

        # get mode of oscillation period
        moperiod = sp.stats.mode(oscillation)[0][0]

        # create an x-array for sine wave
        if per == 'mean':
            period = muperiod
        elif per == 'mode':
            period = moperiod

        x = corrected.index.to_numpy()

        amplitude = peaks.iloc[0]
        period    = (2 * np.pi)/period
        intercept = peaks.index[0] - period

        # define decay constant
        lam = 0.002

        damp  = np.exp(-lam * (x-350))

        sine = amplitude* damp * np.sin(period * x + intercept)

        sine = pd.Series(sine, index = x, name = 'Dampened Sine')

        #sin = sine.clip(lower=0)

        return sine

    def apply_baseline(self, algo, normalize = True, plot = False):        
        samp = self.samp
        sr = self.sr

        if algo == 'arPLS':
            base, corrected, info = self.arPLS_baseline(sr, lam=1e4,
                                                         niter=5000,
                                                         full_output=True)
        elif algo == 'minima':
            base, corrected = self.minima_baseline(sr)
        elif algo == 'polyfit':
            base, corrected = self.polyfit_baseline(sr)
        else:
            Exception(f'Alogrithm does not have a {algo} type')
        
        sin = self.oscillation_removal(sr)
        plt.plot(sin + base)

        # add corrected spectrum to df
        corrected.name = samp + algo
        
        if normalize:
            # normalize to values between 0 and 1
            corrected = corrected/corrected.max()
        
        if plot:
            # Plot baseline corrected spectra
            # Create subplot
            fig, ax = plt.subplots(1, figsize= [15, 5])
            
            # Spectral plot
            ax.plot(corrected, '-', color = 'k', linewidth=0.6)
            
            # Fix axis scale, set title, add labels
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_title('arPLS Corrected Spectrum: ' + samp)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
            ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
            
            # Set minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            plt.show()
        
        self.corrected = corrected
        
        return corrected

    def apply_smoothing(self, algo):
        
        samp = self.samp
        
        corr_signal = self.apply_baseline(algo)
        
        self.corr_signal = corr_signal
        
        # try SG smoothing
        smooth = ss.savgol_filter(corr_signal, window_length=11, 
                                                  polyorder=5, 
                                                  deriv=0)
        
        # make it a pd.Series
        smoothsr = pd.Series(smooth)
        
        # ensure indices still match
        smoothsr.index = corr_signal.index
              
        if algo != 'arPLS':
            algo = algo.capitalize()
        
        # Plot smoothed baseline corrected spectra
        # Create subplot
        fig, ax = plt.subplots(1, figsize= [15, 5])
        
        # Spectral plot
        ax.plot(smoothsr, '-', color = 'k', linewidth=0.6)
        
        # Fix axis scale, set title, add labels
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_title(f'Smoothed {algo} Corrected Spectrum: ' + samp)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", family="serif",  fontsize=12)
        ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
        
        # Set minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.show()
        
        return smoothsr
    
    def run(self, baseline='arPLS', smooth = True):
        
        sr = self.sr
        
        smoothed = self.apply_smoothing(baseline)
        
        corrected = self.corrected
        
        return sr, corrected, smoothed
    

if __name__ == "__main__":
    df = RamanRead().singlefile(direct + filedir, file)
    for i in df.columns:
        sr, corrected, smoothed = RamanClean(df[i]).run()





# export new smoothed and corrected spectrum as a CSV (using arPLS not baseline)
# df['PLSsmooth'].to_csv(direct + f'{samp}_Raman.csv')

