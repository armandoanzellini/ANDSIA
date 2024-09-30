# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:57:36 2019

@author: Armando Anzellini

Module of functions necessary for IsoSpec to work properly and can also be used
for other purposes if suitable.
"""
import numpy as np
import scipy as sp
import pandas as pd
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib.ticker import AutoMinorLocator
from scipy.sparse import linalg
from numpy.linalg import norm

def oscillation(m1, m2, k):
    '''
    Provides the expected frequency of oscillation for a molecule composed of 
    two atoms of mass m1 and m2 using k as the bonding energy.

    Parameters
    ----------
    m1 : float
        mass of isotope 1 in molecule. In atomic mass units (amu).
    m2 : float
        mass of isotope 2 in molecule. In atomic mass units (amu).
    k : float
        estimated bonding energy for the atom in the analysis

    Returns
    -------
    nu_bar : float
        frequency of the oscillation of the atom.

    '''
    mu  = (m1*m2)/(m1+m2)
    var = np.sqrt(k/mu)
    nu_bar = int(round(1302*var))
    return nu_bar 

def k_constant(m1, m2, nu_bar):
    '''
    estimates the bonding energy k of a molecule composed of two atoms of mass
    m1 and m2.

    Parameters
    ----------
    m1 : float
        mass of isotope 1 in molecule. In atomic mass units (amu).
    m2 : float
        mass of isotope 2 in molecule. In atomic mass units (amu).
    nu_bar : float
        wavelength location of peak when observing original molecule. Serves as
        a proxy for the frequency of oscillation of the spring as per Hooke's 
        law

    Returns
    -------
    k : float
        estimated bonding energy of the molecule.

    '''
    mu  = (m1*m2)/(m1+m2)
    var = np.square(nu_bar/1302)
    k   = mu*var
    return k

def shift(m1, m2, m1_star, m2_star, nu):
    '''
    Provides the estimated isotopic shift for any molecule given the parameters
    through the use of Hooke's law. Greatly simplifies the physics of the 
    interaction but provides a good aproximation for small molecules like
    carbonate and phosphate.

    Parameters
    ----------
    m1 : float
        mass of isotope 1 in original molecule. In atomic mass units (amu).
    m2 : float
        mass of isotope 2 in original molecule. In atomic mass units (amu).
    m1_star : float
        mass of isotope 1 in isotopically shifted molecule. In amu.
    m2_star : float
        mass of isotope 2 in isotopically shifted molecule. In amu.
    nu : float
        wavelength location of peak when observing original molecule (m1-m2).

    Returns
    -------
    nu_star : float
        returns the wavelength corresponding to the estimated isotopic shift.

    '''
    k       = k_constant(m1, m2, nu)
    nu_star = oscillation(m1_star, m2_star, k)
    return nu_star

def wavenumber(wavelength, unit='micron'):
    '''
    Function that quickly converts wavelengths to wavenumbers

    Parameters
    ----------
    wavelength : float
        wavelength in microns or nanometers to convert to wavenumbers.
    unit : str, optional
        define the unit of the input wavelength value. The default is 'micron'.
        can also be 'nanometer'

    Returns
    -------
    nu_bar : float
        wavelength in wavenumbers after conversion.

    '''
    if unit != 'micron' and unit != 'nanometer':
        Exception('Please provide a valid wavelength')
    w = wavelength
    if unit == 'micron':
        w /= 10000
    elif unit == 'nanometer':
        w /= 10000000
    nu_bar = 1/w
    return nu_bar

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    '''
    Not yet working
    
    This function implements the varimax rotation on a component matrix
    
    Parameters
    ----------
    Phi : TYPE
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 1.
    q : TYPE, optional
        DESCRIPTION. The default is 20.
    tol : TYPE, optional
        DESCRIPTION. The default is 1e-6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)

def baseline_linear(df, x_col, y_col, intercept1, intercept2):
    '''
    This function takes a signal dataframe and creates a linear baseline running
    between the provided intercepts. It adjusts that baseline to ensure that no
    data from the spectrum is removed by the subtraction of the line or is 
    intercepted by the baseline, creating a true linear baseline of the spectrum.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe with the wavelength and signal values of the spectrum.
    x_col : str
        column name from DataFrame defining the wavelengths.
    y_col : str
        column name from DataFrame defining the signal values.
    intercept1 : int or float
        DESCRIPTION.
    intercept2 : int or float
        DESCRIPTION.

    Returns
    -------
    x_axis : list
        x_values for the line.
    y_axis : list
        y-values for the line.

    '''
    
    # Find intercept values from the given dataframe assuming no exact values
    cept2 = df.iloc[(df[x_col]-intercept2).abs().argsort()[:1]] # closest value
    cept1 = df.iloc[(df[x_col]-intercept1).abs().argsort()[:1]] # closest value
    y2 = cept2[y_col].to_list()[0]
    y1 = cept1[y_col].to_list()[0]
    x2 = cept2[x_col].to_list()[0]
    x1 = cept1[x_col].to_list()[0]
    
    # Create list that allows numpy functions to be used
    x = [x1, x2]
    y = [y1, y2]
    
    # Use numpy to get coefficients
    coefficients = np.polyfit(x, y, 1)
    
    # Use numpy to create line
    polynomial = np.poly1d(coefficients)
    x_axis = df[x_col].values
    y_axis = polynomial(x_axis)
    
    # Adjust line downwards to not lose data to baseline
    lval = []
    for i in range(len(df)):
        if df[y_col].iloc[i] < y_axis[i]:
            diff = abs(df[y_col].iloc[i] - y_axis[i])
            lval += [[df[x_col].iloc[i], df[y_col].iloc[i], diff]]
        else:
            pass
        
    lval.sort(key=lambda x: x[2],reverse=True) # sort the list of values
    y_axis = [k - lval[0][2] for k in y_axis] # Subtract largest value to adjust
    
    return x_axis, y_axis

def baseline_quadratic(df, x_col, y_col, intercept1, intercept2):
    '''
    This function takes a signal dataframe and creates a parabolic baseline running
    between the provided intercepts. It adjusts that baseline to ensure that no
    data from the spectrum is removed by the subtraction of the line or is 
    intercepted by the baseline, creating a true parabolic baseline of the spectrum.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe with the wavelength and signal values of the spectrum.
    x_col : str
        column defining the wavelengths.
    y_col : str
        column defining the signal.
    intercept1 : int or float
        where the correction line should begin (sorting not necessary).
    intercept2 : int or float
        where the correction line should end (sorting not necessary).

    Returns
    -------
    x_axis : list
        x_values for the polynomial.
    y_axis : list
        y-values for the polynomial.

    '''
    
    # Find intercept values from the given dataframe assuming no exact values
    cept2 = df.iloc[(df[x_col]-intercept2).abs().argsort()[:1]] # closest value
    cept1 = df.iloc[(df[x_col]-intercept1).abs().argsort()[:1]] # closest value
    y2 = cept2[y_col].to_list()[0]
    y1 = cept1[y_col].to_list()[0]
    x2 = cept2[x_col].to_list()[0]
    x1 = cept1[x_col].to_list()[0]
    
    # Create list that allows numpy functions to be used
    x = [x1, x2]
    y = [y1, y2]
    
    # Use numpy to get line coefficients
    lincoeff = np.polyfit(x, y, 1)
    
    # Use numpy to create line
    line = np.poly1d(lincoeff)
    line_x = df[x_col].values
    line_y = line(line_x)
    
    # Adjust line downwards to not lose data to baseline
    lval = []
    for i in range(len(df)):
        if df[y_col].iloc[i] < line_y[i]:
            diff = abs(df[y_col].iloc[i] - line_y[i])
            lval += [[df[x_col].iloc[i], df[y_col].iloc[i], diff]]
        else:
            pass
        
    x3, y3 = max(lval, key=lambda x: x[2])[:2] # Find last point for polyfit
    
    # Add last point for a polynomial fit to the data
    x += [x3]
    y += [y3]
    
    # Fit data to a numpy quadratic polynomial
    coefficients = np.polyfit(x, y, 2)
    
    # Use numpy to create quadratic polynomial fit function
    polynomial = np.poly1d(coefficients)
    
    x_axis = df[x_col].values
    y_axis = polynomial(x_axis)
    
    # Adjust new polynomial to reduce how much of the data are below 0
    for j in range(1000):
        lval = []
        for i in range(len(df)):
            if df[y_col].iloc[i] < y_axis[i]:
                diff = df[y_col].iloc[i] - y_axis[i]
                lval += [[df[x_col].iloc[i], df[y_col].iloc[i], diff]]
                lval[:] = (i for i in lval if i[0] != x[0])
                lval[:] = (i for i in lval if i[0] != x[1])
        if lval:
            x3, y3 = min(lval, key=lambda x: x[2])[:2] # Find new points
            x[2] = x3 # Add last point for a polynomial fit to the data
            y[2] = y3
            coefficients = np.polyfit(x, y, 2) # Fit data to new polynomial
            polynomial = np.poly1d(coefficients)
            x_axis = df[x_col].values
            y_axis = polynomial(x_axis)            
        if not lval:
            break
        j += 1
                
    
    #Adjust new polynomial downward again if lval is still not empty
    if lval:
        shft = max(lval, key=lambda x: x[2])[2]
        # sort the list of values
        y_axis = [k - shft for k in y_axis] # Subtract largest value to adjust
    
    return x_axis, y_axis
    
def del13C(sample):
    '''
    Given the sample ratio of 13C to 12C, this function returns the delta value
    to the VPDB reference.

    Parameters
    ----------
    sample : float
        sample ratio of 13C to 12C.

    Returns
    -------
    float
        delta13C value of sample on the VPDB reference.

    '''
    return ((sample/0.01123720)-1)*1000

def invdel13C(permil):
    '''
    Given the delta13C value of a sample, this function returns the ratio of 
    13C to 12C in the given sample when the isotopic ratio is referenced to the
    VPDB scale

    Parameters
    ----------
    permil : float
        delta13C VPDB value of a sample.

    Returns
    -------
    float
        ratio of 13C to 12C in the sample for which the delta13C is reported.

    '''
    return ((permil/1000) + 1) * 0.01123720

def irsf_prom(data, x_col, y_col, baseline=False):
    '''
    This formulae provides the infrared splitting factor (IRSF), also known as
    the crystallinity index, for a given FTIR spectrum of bone using the
    prominence of the nu4PO4 peaks instead of their heighs. To be used in case
    of cut-off spectra or uncertain baseline correction.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the wavelength and signal data.
    x_col : str
        column name from DataFrame defining the wavelengths.
    y_col : str
        column name from DataFrame defining the signal values.
    baseline : bool, optional
        If true, creates a standard baseline between 450 cm-1 and 750 cm-1.
        If FALSE, it assumes that the data have been baseline corrected.
        The default is False.

    Returns
    -------
    ciir : float
        returns the IRSF value for the given spectrum.

    '''
    data = data[data[x_col].between(450, 700)].copy()
    
    # find all local maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    minima = data.iloc[sp.signal.argrelextrema(data[y_col].values, 
                                               np.less_equal, order=n)[0]]
    
    maxima = data.iloc[sp.signal.argrelextrema(data[y_col].values, 
                                               np.greater_equal, order=n)[0]]
    
       
    # sort minima by proximity to expected baseline wavelength and get nearest
    min1 = minima.iloc[(minima[x_col]-450).abs().argsort()[:1]]
    min2 = minima.sort_values(y_col).iloc[:1]
    #min2 = minima.iloc[(minima[x_col]-750).abs().argsort()[:1]]   
      
    # get peak heights
    A605 = maxima.iloc[(maxima[x_col]-605).abs().argsort()[:1]]
    A565 = maxima.iloc[(maxima[x_col]-565).abs().argsort()[:1]]
    A590 = minima.iloc[(minima[x_col]-590).abs().argsort()[:1]]  
    
    #Create line at peak height of A590 to get prominences
    un_corr_ht = A590[y_col].iloc[0]
 
    if baseline == True:
        # create x values for baseline if needed
        line_x = np.linspace(min1[x_col].iloc[0],
                             min2[x_col].iloc[0])
        
        # create linear baseline
        m      = (min2[y_col].iloc[0] - min1[y_col].iloc[0]) /    \
                 (min2[x_col].iloc[0] - min1[x_col].iloc[0])
        b      =  min1[y_col].iloc[0] - (m*line_x[0])
        
        def line(x):
            return m*x+b
              
        # calculate peak height at 590
        ht590 =  A590[y_col].iloc[0] - line(A590[x_col].iloc[0])
        
        
    elif baseline == False:
        # get peak height without baseline
        ht590 =  A590[y_col].iloc[0]
        
    # get prominences for both peaks
    prom605 = A605[y_col].iloc[0] - un_corr_ht
    prom565 = A565[y_col].iloc[0] - un_corr_ht
    
    ciir = ((prom605 + prom565)/ht590) + 2  
    
    # get crossing of data and line to limit in figure plot
    idx = np.argwhere(np.diff(np.sign(
                      np.array([un_corr_ht]*len(data)) - data[y_col].to_numpy())))
    
    # Plot lines, peaks, and baselines to ensure accuracy
    fig = plt.figure()
    ax  = fig.add_subplot()
    plt.tight_layout()
    
    # Plot original data
    ax.plot(data[x_col], data[y_col], "k")
    ax.invert_xaxis()
    
    # Add plot for prominence baseline
    ax.hlines(un_corr_ht, data[x_col].iloc[idx[0]],
                    data[x_col].iloc[idx[-1]], 
                                          color = "k", linestyle = "--")
    # Add plot from baseline
    if baseline == True:
        ax.plot(line_x, line(line_x), color = "k", linestyle = "--")
        ax.vlines(x = A590[x_col].iloc[0], 
                  ymin = line(A590[x_col].iloc[0]), 
                  ymax = A590[y_col].iloc[0],
                   color = "k", linestyle = "-.")
    else:
        ax.vlines(x = A590[x_col].iloc[0],
                  ymin = 0,
                  ymax = A590[y_col].iloc[0],
                  color = "k", linestyle = "-.")
    
    # Plot lines for peak heights
    ax.vlines(x=A605[x_col], ymin=un_corr_ht, ymax=A605[y_col], 
                                          color = "k", linestyle = "-.")
    ax.vlines(x=A565[x_col], ymin=un_corr_ht, ymax=A565[y_col],
                                          color = "k", linestyle = "-.")
    
    # label axes
    ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=12)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
    
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    return ciir

def irsf(data, x_col, y_col, baseline=True):
    '''
    This formulae provides the infrared splitting factor (IRSF), also known as
    the crystallinity index, for a given FTIR spectrum of bone.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the wavelength and signal data.
    x_col : str
        column name from DataFrame defining the wavelengths.
    y_col : str
        column name from DataFrame defining the signal values.
    baseline : bool, optional
        If true, creates a standard baseline between 450 cm-1 and 750 cm-1.
        If FALSE, it assumes that the data have been baseline corrected.
        The default is TRUE.

    Returns
    -------
    ciir: float
        the IRSF value for the given spectrum.

    '''
    data = data[data[x_col].between(450, 700)].copy()
    
    # find all local minima and maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    minima = data.iloc[sp.signal.argrelextrema(data[y_col].values, 
                                               np.less_equal, order=n)[0]]
    
    maxima = data.iloc[sp.signal.argrelextrema(data[y_col].values, 
                                               np.greater_equal, order=n)[0]]
    
    # sort minima by proximity to expected baseline wavelength and get nearest
    min1 = minima.iloc[(minima[x_col]-450).abs().argsort()[:1]]
    min2 = minima.sort_values(y_col).iloc[:1]
    #min2 = minima.iloc[(minima[x_col]-750).abs().argsort()[:1]]    
    
    line_x = np.linspace(min1[x_col].iloc[0],
                             min2[x_col].iloc[0]) # x-values for line
    
    if baseline == True:
        # create linear baseline
        m      = (min2[y_col].iloc[0]     - min1[y_col].iloc[0]) /    \
                 (min2[x_col].iloc[0] - min1[x_col].iloc[0])
        b      =  min1[y_col].iloc[0] - (m*line_x[0])
        
        def line(x):
            return m*x+b
        
        corr_line = line(line_x)
        
        # get uncorrected peak heights for 605, 565, and 590
        A605 = maxima.iloc[(maxima[x_col]-605).abs().argsort()[:1]]
        A565 = maxima.iloc[(maxima[x_col]-565).abs().argsort()[:1]]
        A590 = minima.iloc[(minima[x_col]-590).abs().argsort()[:1]]
        
        # get Y values of baseline at wavelength for each peak
        ymin605 = line(A605[x_col]).iloc[0]
        ymin565 = line(A565[x_col]).iloc[0]
        ymin590 = line(A590[x_col]).iloc[0]
        
        # correct peak heights
        val605 = A605[y_col].iloc[0] - ymin605
        val565 = A565[y_col].iloc[0] - ymin565
        val590 = A590[y_col].iloc[0] - ymin590
        
        # Calculate IRSF (aka Crystallinity Index)
        ciir = (val605 + val565)/val590
        
    elif baseline == False:
        # get peak heights for 605, 565, and 590
        A605 = maxima.iloc[(maxima[x_col]-605).abs().argsort()[:1]]
        A565 = maxima.iloc[(maxima[x_col]-565).abs().argsort()[:1]]
        A590 = minima.iloc[(minima[x_col]-590).abs().argsort()[:1]]
        
        # define default minima and baseline at 0 to ensure proper plotting
        ymin605 = 0
        ymin565 = 0
        ymin590 = 0
        
        corr_line = [0] * len(line_x)
        
        # Calculate IRSF (aka Crystallinity Index)
        ciir = (A605[y_col].iloc[0] + A565[y_col].iloc[0])/      \
                        A590[y_col].iloc[0]
    
    # Plot lines, peaks, and baselines to ensure accuracy
    fig = plt.figure()
    ax  = fig.add_subplot()
    plt.tight_layout()
    
    # Plot original data
    ax.plot(data[x_col], data[y_col], "k")
    ax.invert_xaxis()
    
    # Add plot for baseline
    ax.plot(line_x, corr_line, color = "k", linestyle = "--")
    
    # Plot lines for peak heights
    ax.vlines(x=A605[x_col], ymin=ymin605, ymax=A605[y_col], 
                                          color = "k", linestyle = "-.")
    ax.vlines(x=A565[x_col], ymin=ymin565, ymax=A565[y_col],
                                          color = "k", linestyle = "-.")
    ax.vlines(x=A590[x_col], ymin=ymin590, ymax=A590[y_col], 
                                          color = "k", linestyle = "-.")
    
    # label axes
    ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=12)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
    
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    return ciir

def cp_ratio(data, vers= 'raman', smooth=False):
    '''
    This formula calculates the C/P ratio for a given vibrational spectrum of bone

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
    vers  :  string
        {'raman', 'ftir'} choice of peak heights to go for depends on method

    Returns
    -------
    cp : float
        the C/P ratio for the spectrum provided.

    '''
    if vers == 'ftir':
        rang  = [800, 1800]
        cpeak = 1415
        ppeak = 1035
    if vers == 'raman':
        rang  = [800, 3200]
        cpeak = 2941
        ppeak = 960
    
    data = data[rang[0]: rang[1]].copy()
    
    if smooth == True:
        data = ss.savgol_filter(data, window_length=25, 
                      polyorder=2, deriv=0)
    
    # find all local minima and maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    maxima = data.iloc[ss.argrelextrema(data.values, 
                                               np.greater_equal, order=n)[0]]
        
    # get uncorrected peak heights for CH and PO4
    ix = maxima.index.to_list() # get list of posible indices
    locc = sorted(ix, key = lambda x: abs(cpeak - x)) # sort list by proximity
    locp = sorted(ix, key = lambda x: abs(ppeak - x)) # sort list by proximity
    
    Ac = maxima[locc[0]]
    Ap = maxima[locp[0]]
    
    # Calculate C/P ratio
    cp = Ac/Ap
        
    # Plot lines, peaks, and baselines to ensure accuracy
    fig = plt.figure()
    ax  = fig.add_subplot()
    plt.tight_layout()
    
    # Plot original data
    ax.plot(data, color = "k", linewidth = 0.75)
    if vers == 'ftir':
        ax.invert_xaxis()
    
    # Add plot for baseline
    line_x    = data.index.to_list()
    corr_line = [0]*len(line_x) 
    ax.plot(line_x, corr_line, color = "k", linestyle = "--")
    
    # Plot lines for peak heights
    ax.vlines(x=locc[0], ymin=0, ymax=Ac, 
                                          color = "k", linestyle = "-.")
    ax.vlines(x=locp[0], ymin=0, ymax=Ap,
                                          color = "k", linestyle = "-.")
    
    # label axes
    ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=10.5)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=10.5)
    
    # add annotation of ratio in plot
    ax.annotate(f"$CP_{{{vers}}}$ = " + f"{cp: .2f}", 
                xy=(1100, Ap/2),  xycoords='data', family="serif")
    
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    return cp

def amIPO4raman(data, smooth=False):
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
    
    data = data[rang[0]: rang[1]].copy()
    
    if smooth == True:
        data = ss.savgol_filter(data, window_length=25, 
                      polyorder=2, deriv=0)
    
    # find all local minima and maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    maxima = data.iloc[ss.argrelextrema(data.values, 
                                               np.greater_equal, order=n)[0]]
        
    # get uncorrected peak heights for PO4
    ix = maxima.index.to_list() # get list of posible indices
    locp = sorted(ix, key = lambda x: abs(ppeak - x)) # sort list by proximity
    
    # since peak of AmideI at 1636 is usually a shoulder of the peak at 1671
    # peak height for 1636 should be calculated from full data not just maxima
    ixdat = data.index.to_list() # get list of posible indices
    loca  = sorted(ixdat, key = lambda x: abs(apeak - x)) # sort list by proximity
    
    Aa = data[loca[0]]
    Ap = maxima[locp[0]]
    
    # Calculate AmIPO4 ratio (see France et al 2014)
    amipo4 = Ap/Aa
        
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
    
    # label axes
    ax.set_xlabel("Wavenumber ($cm^{-1}$)",family="serif",  fontsize=10.5)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=10.5)
    
    # add annotation of ratio value in plot
    ax.annotate("$PO_4$:Amide I = " + f"{amipo4: .2f}", 
                xy=(1200, Ap/2 ),  xycoords='data', family="serif")
        
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    plt.show()
    
    return amipo4

def amIPO4area(data, x_col, y_col, smooth=False, savefigdir=None):
    '''
    This function returns the AmidI/PO4 ratio for a given FTIR spectrum. Ratio
    calculations are based on Lebon et al 2016 using areas instead of peak
    heights.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the wavelength and signal data.
    x_col : str
        column name from DataFrame defining the wavelengths.
    y_col : str
        column name from DataFrame defining the signal values.
    smooth : bool, optional
        If TRUE, spectrum is smoothed using a Savitzky-Golay algorithm with a 
        window-length of 25 and a polyorder of 2. The default is False, which
        implements no smoothing.
    savefigdir: str, optional
        default is None which does not save figure. If you wish to save the 
        figure, provide a directory.

    Returns
    -------
    ratio: float
        returns the ratio of integrated areas for the AmideI and PO4 bands.

    '''
    if len(data.columns) > 2:
        data = data[[x_col, y_col]]    
    
    data = data[data[x_col].between(800, 1850)].copy()
    
    if smooth == True:
        data.loc[:, y_col] = ss.savgol_filter(data[y_col], window_length=25, 
                      polyorder=2, deriv=0)
    
    # find all local minima and maxima in the signal
    n = 5  # number of points to be checked before and after suspected min
    
    minima = data.iloc[sp.signal.argrelextrema(data[y_col].values, 
                                               np.less_equal, order=n)[0]]
    
    def line(x, m, b):
        return m*x + b
    
    
    # calculate minima 
    ammin1 = minima.iloc[(minima[x_col]-1570).abs().argsort()]
    ammin2 = minima.iloc[(minima[x_col]-1750).abs().argsort()]
    
    # select only the minima that make sense for each
    ammin1 = ammin1[ammin1[x_col] < 1700]
    ammin2 = ammin2[ammin2[x_col] > 1700]
        
    def create_line(df, x_col, y_col, min1, min2):
        # create baseline for Amide area
        line_x = df[df[x_col].between( min1[x_col], 
                                       min2[x_col])][x_col].to_numpy() # line x-values
        
        m      = (min2[y_col] - min1[y_col]) /  \
                 (min2[x_col] - min1[x_col])
                 
        b      =  min1[y_col] - (m*min1[x_col])
        
        return line_x, line(line_x, m, b)
    
    am_line_x, am_baseline = create_line(data, x_col, y_col, ammin1.iloc[0], ammin2.iloc[0])
    
    # check that basline doesn't cross data before subtracting
    # select only points of interest
    amide = data[data[x_col].between(ammin1[x_col].iloc[0],
                                     ammin2[x_col].iloc[0])].copy()  # extent to integrate and create deep copy to manipulate saving original
    

    # select flattening curve if part of ammin2 is same as ammin1
    ammin2 = ammin2.iloc[:1]
    
    sgdf = data[x_col].to_frame()

    sg      = sp.signal.savgol_filter(data[y_col], window_length=9, polyorder=2, deriv=2)
    sgdf[y_col] = sg
    
    # find the second derivative extrema to select a flattening curve
    sgdf_extrema = []
    for i in range(len(sgdf)-1):
        if np.sign(sgdf[y_col].iloc[i]) < np.sign(sgdf[y_col].iloc[i+1]):
            sgdf_extrema += [sgdf['wavelength'].iloc[i], sgdf['wavelength'].iloc[i+1]]
    
    sgdf_extrema = [i for i in sgdf_extrema if i > 1700]
    
    flats = data[data[x_col].isin(sgdf_extrema)].sort_values(y_col)
    
    ammin2 = ammin2.append(flats)

    # check baseline near minimum at 1800
    l = int(len(amide)/3)
    for i in range(len(ammin2)-1):
        if not all(amide[y_col].to_numpy()[l:-2] - am_baseline[l:-2] > 0): # increasing order so begin with l to the end
            i += 1
            am_line_x, am_baseline = create_line(data, x_col, y_col, ammin1.iloc[0], ammin2.iloc[i])
            amide = data[data[x_col].between(ammin1[x_col].iloc[0],
                                         ammin2[x_col].iloc[i])].copy()
        else:
            break       # to lock in the i value for the next iteration
        
    # now we do the same but for the minimum at 1600
    for j in range(len(ammin1)-1):
        if not all(amide[y_col].to_numpy()[2:-l] - am_baseline[2:-l] > 0): # increasing order so begin with l to the end
            j += 1
            am_line_x, am_baseline = create_line(data, x_col, y_col, ammin1.iloc[j], ammin2.iloc[i])
            amide = data[data[x_col].between(ammin1[x_col].iloc[j],
                                         ammin2[x_col].iloc[i])].copy()        
         
    # Subtract baseline to correct signal for integration    
    amide['corr_signal'] = amide[y_col] - am_baseline
    
    # Calculate area of amide peak
    amide_area = sp.integrate.simpson(amide['corr_signal'], sorted(am_line_x))    
    
    # AMIDE CALCULATION COMPLETE
    
    # get minima for PO4 area
    pomin1 = minima.iloc[(minima[x_col]-890).abs().argsort()]
    pomin2 = minima.iloc[(minima[x_col]-1160).abs().argsort()]
    
    # create baseline for PO4 area    
    po_line_x, po_baseline = create_line(data, x_col, y_col, pomin1.iloc[0], pomin2.iloc[0])
       
    # Subtract baseline to correct signal for integration
    po4 = data[data[x_col].between(pomin1[x_col].iloc[0],
                                   pomin2[x_col].iloc[0])].copy()  # extent to integrate and create deep copy to manipulate saving original
    
    po4['corr_signal'] = po4[y_col] - po_baseline
    
    # Calculate area of PO4 peak
    po_area = sp.integrate.simpson(po4['corr_signal'], sorted(po_line_x))
    
    # Plot lines, peaks, and baselines to ensure accuracy
    fig = plt.figure()
    ax  = fig.add_subplot()
    plt.tight_layout()
    
    # Plot original data
    ax.plot(data[x_col], data[y_col], "k")
    ax.invert_xaxis()
    
    # Add plots for baselines
    ax.plot(am_line_x, am_baseline, color = "k", linestyle = "--")
    ax.plot(po_line_x, po_baseline, color = "k", linestyle = "--")
    
    # Fill areas under the curve to show the integration occurring
    plt.rcParams['hatch.color'] = '0.5' # set hatch color to gray
    ax.fill_between(am_line_x, am_baseline, amide[y_col], 
                                            facecolor = '0.75', hatch = '//')
    ax.fill_between(po_line_x, po_baseline, po4[y_col]  , 
                                            facecolor = '0.75', hatch = '//')
    
    # label axes
    ax.set_xlabel("Wavenumber (cm$^{-1}$)",family="serif",  fontsize=12)
    ax.set_ylabel("Absorbance",family="serif",  fontsize=12)
    
    # title of plot
    ax.set_title('AmideI/PO$^{4}$ Ratio: ' + y_col, family="serif")
    
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    if savefigdir:
        plt.savefig(savefigdir + f'AMIPO4_{y_col}.png', dpi=300)
    
    # show plot
    plt.show()
    
    # Calculate ratio to return
    ratio = amide_area/po_area
    
    return ratio

def ci_raman (data, add_baseline=False):
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
    data = data[900: 1050].copy()
    
    if add_baseline:
        # find minima to correct baseline
        left  = data[900: 940]
        lmini = ss.argrelmin(left.values)[-1] # closest local minimum to peak
        lmin  = (left.index[lmini][0], left.iloc[lmini].values[0])
        
        right = data[980: 1010]
        rmini = ss.argrelmin(right.values)[0] # closest local minimum to peak
        rmin  = (right.index[rmini][0], right.iloc[rmini].values[0])
                        
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
    peaks_ix, _ = ss.find_peaks(data, height=50)
    
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
    ax.set_ylabel("Absorbance",family="serif",  fontsize=10.5)
    
    # add annotation of CIraman in plot
    ax.annotate("$CI_{raman}$ = " + f"{ci: .2f}", 
                xy=(990, hm),  xycoords='data', family="serif")
    
    # Set Minor tick marks
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    plt.show()
     
    return ci
    
def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
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

    
    