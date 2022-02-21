# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:17:58 2022

@author: Stefano
"""

import numpy as np
from numpy.random import randint
from random import choices
import numba as nb
from numba import jit, njit

from scipy import stats
from scipy.optimize import curve_fit, NonlinearConstraint
from iminuit import Minuit
from uncertainties import ufloat

        

class Ricepile:
    """
    A pile evolving according to the Oslo Model. Uses numba for faster 
    relxation. \n
    
    Parameters:
    -----------
        L : int
            Length of the pile, i.e. number of sites it has.
        zthmax: int
            Maximum threshold on the slope. 
        p: float or list of floats
            If scalar, then use flat probability distribution to generate
            threshold zth; if list, use it as weight.
    
    Attributes:
    ----------
        L : int
            Length of the pile, i.e. number of sites it has. 
        t : int
            Time from creation of pile. 
        tc: int
            Cross-over time, at which pile enters in critical state.
        h : list of int
            List of heights of the L sites. 
        z : list of int 
            List of slopes of the L sites, defined as 
                    z_i = h_i - h_i+1. 
        slist : list of int
            List of avalanches' sizes, from t=1 (Not just critical).
        confs : list of lists of ints
            list of different stable configurations, defined by values z_i,
            in which the pile has been after tc. 
            It will only get updated if trckconf is set True in the drive 
            function, though it's highly recommended to turn it on ONLY if 
            strictly necessary. 
        zth : List of int
            List of thresholds on the slopes for the L sites.
        zth_choices: list of int 
            List of possible values of zth. 
        zth_weights: list of float
            Weights for possible values of zth. 
            
    """
    
    def __init__(self, L = 16, zthmax = 2, p = 1/2):
        self.L = int(L)
        self.t = 0
        self.tc = 0
        
        self.h = np.zeros(L)
        self.z = np.zeros(L)
        
        self.slist = []
        self.confs = []
        
        self.zth = []
        self.zth_choices = np.array([i+1 for i in range(int(zthmax))])
        
        if not hasattr(p, "__len__"):
            self.zth_weights = np.array([p] * zthmax)
        else: 
            self.zth_weights = p
        
        for i in range(L):
            self.zth.append(choices(self.zth_choices, 
                                           self.zth_weights)[0])
        self.zth = np.array(self.zth)
            
    
    def __str__(self):
        return "A list of sites with height hi and slope zi, evolving \
        according to the Oslo Model"
   
    def __len__(self):
        return self.L
    
    def hlist(self):
        return self.h
    
    def zlist(self):
        return self.z
    
    def height(self):
        return self.h[0]
    
    def time(self):
        return self.t
    
    def set_conf(self, newzs, update_tc = False):
        self.z = np.array(newzs)
        self.h_update()
        if update_tc:
            self.tc = self.t + 1
    
    def h_update(self):
        self.h[0] = sum(self.z)
        for i in range(1, len(self)):
            self.h[i] = self.h[i-1] - self.z[i-1]
    
    def get_tc(self):
        if not self.tc:
            print("NOTE: it hasn't reached the critical state, hence tc =0 .")
        return self.tc
    
    def get_avax(self):
        return self.slist
    
    def get_crit_avax(self):
        if not self.tc:
            print("NOTE: it hasn't reached the critical state, hence tc =0 .")
        return self.slist[self.tc:]
        
    def drive(self, N = 1e4, trackconf = False, addspot = 0):
        """
        Parameters
        ----------
        N : int
            The number of grains added.
            
        trackconf: Bool or float, optional.
            If True, return number of different stable configurations detected 
            in critical state, where 2 configurations differ if one or more 
            values of z_i differ between them.
        
        addspot: int or "random charge" or "random height"
            If it is an integer i between 0 and len(pile)-1, then add grain to
            ith place in pile. 
            If it is "random charge", then choose randomly where to add a slope
            charge (hence z[random]->z[random]+1).
            If it is "random height", then choose randomly where to add a block
            (hence z[random]->z[random]+1 AND z[random-1]->z[random-1]-1).
            Default is 0. 

        Returns
        -------
        heightlist : list of int
            List of hights at all times
            
        mean_height : float
            Mean of the heights in steady state. If critical state has not
            been reached, return "NaN" instead.
            
        std_height: float
            Standard Deviation of the height in steady state. If critical state
            has not been reached, return "NaN" instead.
            
        Nconfs: int, if trackconf is True.
            Number of different stable configurations detected in critical 
            state, where 2 configurations differ if one or more values of z_i 
            differ between them. 
            Then returns: 
                (heightlist, mean_height, std_height), trackconf
        """
        
        if trackconf:
            heightlist = []
            confs = self.confs
            
            # Drive
            for j in range(int(N)):
                self.add(addspot)
                s = self.relax()
                self.slist.append(int(s))
                heightlist.append(int(self.h[0]))
                #heightlist.append(sum(self.z))
                
                if self.tc:
                    z2 = (self.z).tolist()
                    if z2 not in confs:
                        confs.append(z2 * 1)

            #Update h at the end (redundant variable)
            #self.h[0] = sum(self.z)
            for i in range(1, len(self)):
                self.h[i] = self.h[i-1] - self.z[i-1]
            
            #Data for configurations' tracking
            self.confs = confs
            Nconfs = len(confs)
            
            if self.tc:
                mean_height = np.mean(heightlist[self.tc:])
                std_height = np.std(heightlist[self.tc:])
            else:
                print("NOTE: it hasn't reached the critical state.")
                mean_height = float("nan")
                std_height = float("nan")
                
            return (heightlist, mean_height, std_height), Nconfs
        
        else: #no trackonf
            heightlist = []
            
            #Drive
            for j in range(int(N)):
                self.add(addspot)
                s = self.relax()
                self.slist.append(int(s))
                heightlist.append(int(self.h[0]))
            
            self.h_update()
                
            if self.tc:
                mean_height = np.mean(heightlist[self.tc:])
                std_height = np.std(heightlist[self.tc:])
            else:
                print("NOTE: it hasn't reached the critical state.")
                mean_height = float("nan")
                std_height = float("nan")
                
            return heightlist, mean_height, std_height
        
           
    def add(self, i = 0): #implicitly "grain",  to ith position
        """
        Add one grain at spot i, with i given by the addspot argument of the
        class. 
        """
        L = len(self)
        if isinstance(i, int):
            self.t += 1
            if i == 0:
                self.h[0] += 1
                self.z[0] += 1
            elif 0 < i < L:
                self.z[i-1] -= 1
                self.z[i] += 1
            else:
                raise Exception("Chosen addspot is too big or negative")
        elif i == "random charge" or i == "rc":
            ind = randint(0, L)
            self.t += 1
            self.h[0] += 1
            if ind == 0:
                self.z[0] += 1
            else:
                self.z[ind] += 1
        elif i == "random height" or i == "rh":
            ind = randint(0, L)
            self.t += 1
            if ind == 0:
                self.h[0] += 1
                self.z[0] += 1
            else:
                self.z[ind-1] -= 1
                self.z[ind] += 1
        else:
            raise Exception("Chosen addspot is neither an int in [0,L),"+
                            "nor 'random height' nor 'random charge'.")
                
                 
    def relax(self):
        s,self.h[0], self.z, self.zth, self.tc = numba_relax(self.h[0],self.z,
                                                             self.zth, self.tc,
                                                             self.L,   self.t,
                                                             self.zth_choices)
        return s



@njit                                            # no bulk correct, a0 correct
def numba_relax(h, z, zth, tc, L, t, zth_choices): #bulk D = 1.3, tau = 1.2
    s = 0 #avalanche size
    zthhigh = max(zth_choices) +1
    torelax = z > zth #if True is to be relaxed
    while True in torelax:
        for i in np.arange(0, L, 1):
            if torelax[i]:
                s += 1 #avalanche size
                if i == 0: #1st site
                    z[0] -= 2
                    h -= 1
                    z[1] += 1
                    zth[0] = randint(1, zthhigh)
                    
                elif i == L - 1: #last site
                    z[-1] -= 1
                    z[-2] += 1
                    zth[-1] = randint(1, zthhigh)
                    #If last site relaxed, steady critical state started
                    if not tc:
                        tc = t - 1
                
                else: # all sites in between
                    z[i] -= 2
                    z[i+1] += 1
                    z[i-1] += 1
                    zth[i] = randint(1, zthhigh)
        torelax = z > zth #if True is to be relaxed
            
    return s, h, z, zth, tc

#%%Other useful functions
 

def hMean (L, a0, a1, w1):
    havg = a0 * L - a0 * a1 * L**(1 - w1)
    return havg

def hmean_scaled (L, a0, b1, w1):
    havg_scaled = a0 - b1 * L ** (-w1)
    return havg_scaled

def h_std(L, A, k):
    return A * L**(k)


def fit_h1(hmean_list, Llist, hstd_list, guess, flexible = True):
    """
    Fit a0, a1, w1 by considering mean height vs theoretical hMean.
    """
    if flexible:
        try:
            bounds = ( [1.5, -np.inf, 0], np.inf)
            [a0, a1, w1], cov = curve_fit(hMean, Llist, hmean_list, 
                                          p0 = guess, bounds = bounds,
                                          sigma = hstd_list,
                                          absolute_sigma = True)
            perr = np.sqrt(np.diag(cov))
            a0 = ufloat(a0, perr[0])
            a1 = ufloat(a1, perr[1])
            w1 = ufloat(w1, perr[2])
            
            a0_v = a0.nominal_value
            a1_v = a1.nominal_value
            w1_v = w1.nominal_value
        except:
            print("fit_h1 failed, hence returning nan. For more info, please\
  set flexible False.")
            a0 = float("nan")
            a1 = float("nan")           
            w1 = float("nan")
            a0_v = float("nan")
            a1_v = float("nan")
            w1_v = float("nan")
    else:
        bounds = ( [1.5, -np.inf, 0], np.inf)
        [a0, a1, w1], cov = curve_fit(hMean, Llist, hmean_list, 
                                      p0 = guess, bounds = bounds,
                                      sigma = hstd_list,
                                      absolute_sigma = True)
        perr = np.sqrt(np.diag(cov))
        a0 = ufloat(a0, perr[0])
        a1 = ufloat(a1, perr[1])
        w1 = ufloat(w1, perr[2])
        
        a0_v = a0.nominal_value
        a1_v = a1.nominal_value
        w1_v = w1.nominal_value
    return (a0_v, a1_v, w1_v), (a0, a1, w1)



def fit_h2(hmean_list, Llist, hstd_list, guess, step, flexible = True):
    """
    Fit a0 by considering straightness of log(1-hmean/(a0L)) vs log(L), then
    find w1 and a1 as gradient and intercept.
    """
    a0s = np.arange(max(np.array(hmean_list)/Llist), 1.75, step)[1:]
    grads, grads_std = [],[]
    for a in a0s:
        grads.append([])
        grads_std.append([])
        for i in range(len(Llist) - 1):
            v1 = np.log(1- np.array(hmean_list)/a/Llist)[i]
            v2 = np.log(1- np.array(hmean_list)/a/Llist)[i+1]
            grads[-1].append(v2 - v1) #all the discrete slopes
        grads_std[-1] = ( np.std(grads[-1]) )
    imin = np.argmin(grads_std)
    a0 = a0s[imin]        # the one leading to more straightnes
                          # hence smaller std in discrete slopes
    #Then
    stds = hstd_list / (a0*Llist - np.array(hmean_list))
    if flexible:
        try:
            (w1, loga1), cov = np.polyfit(np.log(Llist), 
                                    np.log(1- np.array(hmean_list)/a0/Llist),
                                    1, w = 1/stds**2, cov = "unscaled")
            w1_err, loga1_err = np.sqrt(np.diag(cov))
            loga1 = ufloat(loga1, loga1_err)
            a0 = ufloat(a0, step)
            a1 = np.exp(1) ** loga1
            w1 = ufloat(-w1, w1_err)
            
            a0_v = a0.nominal_value
            a1_v = a1.nominal_value
            w1_v = w1.nominal_value
        except:
            print("fit_h3 failed, hence returning nan. For more info, please\
  set flexible False.")
            a0 = float("nan")
            a1 = float("nan")           
            w1 = float("nan")
            a0_v = float("nan")
            a1_v = float("nan")
            w1_v = float("nan")
    else:
        (w1, loga1), cov = np.polyfit(np.log(Llist), 
                                      np.log(1- np.array(hmean_list)/a0/Llist),
                                      1, w = 1/stds, cov = "unscaled")
        w1_err, loga1_err = np.sqrt(np.diag(cov))#/np.sqrt(len(Llist))
        loga1 = ufloat(loga1, loga1_err)
        a0 = ufloat(a0, step)
        a1 = np.exp(1) ** loga1
        w1 = ufloat(-w1, w1_err)
        
        a0_v = a0.nominal_value
        a1_v = a1.nominal_value
        w1_v = w1.nominal_value
    return (a0_v, a1_v, w1_v), (a0, a1, w1)
    
#%% Logbin
################################################################################
# Max Falkenberg McGillivray
# mff113@ic.ac.uk
# 2019 Complexity & Networks course
#
# logbin230119.py v2.0
# 23/01/2019
# Email me if you find any bugs!
#
# For details on data binning see Appendix E from
# K. Christensen and N.R. Moloney, Complexity and Criticality,
# Imperial College Press (2005).
###############################################################################

def logbin(data, scale = 1., zeros = False):
    """
    Max Falkenberg McGillivray. Complexity & Network course, 2019. 
    mff113@ic.ac.uk
    
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(
                count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y

# class Ricepile:
#     """
#     A pile evolving according to the Oslo Model.\n
    
#     Parameters:
#     -----------
#         L : int
#             Length of the pile, i.e. number of sites it has.
#         zthmax: int
#             Maximum threshold on the slope. 
#         p: float or list of floats
#             If scalar, then use flat probability distribution to generate
#             threshold zth; if list, use it as weight.
    
#     Attributes:
#     ----------
#         L : int
#             Length of the pile, i.e. number of sites it has. 
#         t : int
#             Time from creation of pile. 
#         tc: int
#             Cross-over time, at which pile enters in critical state.
#         h : list of int
#             List of heights of the L sites. 
#         z : list of int 
#             List of slopes of the L sites, defined as 
#                     z_i = h_i - h_i+1. 
#         slist : list of int
#             List of avalanches' sizes, from t=1 (Not just critical).
#         confs : list of lists of ints
#             list of different stable configurations, defined by values h_i,
#             in which the pile has been after tc. 
#             It will only get updated if trckconf is set True in the drive 
#             function, though it's highly recommended to turn it on ONLY if 
#             strictly necessary. 
#         zth : List of int
#             List of thresholds on the slopes for the L sites.
#         zth_choices: list of int 
#             List of possible values of zth. 
#         zth_weights: list of float
#             Weights for possible values of zth. 
            
#     """
    
#     def __init__(self, L = 16, zthmax = 2, p = 1/2):
#         self.L = int(L)
#         self.t = 0
#         self.tc = 0
        
#         self.h = []
#         self.z = []
        
#         self.slist = []
#         self.confs = []
        
#         self.zth = []
#         self.zth_choices = [i+1 for i in range(int(zthmax))]
        
#         if not hasattr(p, "__len__"):
#             self.zth_weights = [p] * zthmax
#         else: 
#             self.zth_weights = p
        
#         for i in range(L):
#             self.h.append(0)
#             self.z.append(0)
#             self.zth.append(choices(self.zth_choices, 
#                                             self.zth_weights)[0])
            
    
#     def __str__(self):
#         return "A list of sites with height hi and slope zi, evolving \
#         according to the Oslo Model"
   
#     def __len__(self):
#         return self.L
    
#     def hlist(self):
#         return self.h
    
#     def zlist(self):
#         return self.z
    
#     def height(self):
#         return self.h[0]
    
#     def time(self):
#         return self.t
    
#     def get_tc(self):
#         if not self.tc:
#             print("NOTE: it hasn't reached the critical state, hence tc =0 .")
#         return self.tc
    
#     def get_avax(self):
#         return self.slist
    
#     def get_crit_avax(self):
#         if not self.tc:
#             print("NOTE: it hasn't reached the critical state, hence tc =0 .")
#         return self.slist[self.tc:]
        
#     def drive(self, N, trackconf = False):
#         """
#         Parameters
#         ----------
#         N : int
#             The number of grains added.
            
#         trackconf: Bool or float, optional.
#             If True, return number of different stable configurations detected 
#             in critical state, where 2 configurations differ if one or more 
#             values of z_i differ between them.

#         Returns
#         -------
#         heightlist : list of int
#             List of hights at all times
            
#         mean_height : float
#             Mean of the heights in steady state. If critical state has not
#             been reached, return "NaN" instead.
            
#         std_height: float
#             Standard Deviation of the height in steady state. If critical state
#             has not been reached, return "NaN" instead.
            
#         Nconfs: int, if trackconf is True.
#             Number of different stable configurations detected in critical 
#             state, where 2 configurations differ if one or more values of h_i 
#             differ between them. 
#             Then returns: 
#                 (heightlist, mean_height, std_height), trackconf
#         """

#         if trackconf:
#             heightlist = []
#             confs = []
#             Nconfs = 0
#             for j in range(int(N)):
#                 self.add()
#                 self.relax()
#                 heightlist.append(self.height())
#                 if self.tc:
#                     if self.h not in confs:
#                         confs.append(self.h * 1)
#             self.confs = confs
#             Nconfs = len(confs)
            
#             if self.tc:
#                 mean_height = np.mean(heightlist[self.tc:])
#                 std_height = np.std(heightlist[self.tc:])
#             else:
#                 print("NOTE: it hasn't reached the critical state.")
#                 mean_height = float("nan")
#                 std_height = float("nan")
                
#             return (heightlist, mean_height, std_height), Nconfs
        
#         else:
#             heightlist = []
#             for j in range(int(N)):
#                 self.add()
#                 self.relax()
#                 heightlist.append(self.height())
                
#             if self.tc:
#                 mean_height = np.mean(heightlist[self.tc:])
#                 std_height = np.std(heightlist[self.tc:])
#             else:
#                 print("NOTE: it hasn't reached the critical state.")
#                 mean_height = float("nan")
#                 std_height = float("nan")
                
#             return heightlist, mean_height, std_height
        
#     def add(self, i = 0): #implicitly "grain",  to ith position
#         self.t += 1
#         self.h[i] += 1
#         self.z[i] += 1
    
    
#     def relax(self):
#         counter = 0 # count how many sites are allright
#         s = 0 #avalanche size
#         while counter < self.L:
            
#             counter = 0 #if repeating, reset counter
            
#             for i in range(self.L):
                
#                 if self.z[i] <= self.zth[i]: 
#                     counter += 1   
                    
#                 else:
#                     s +=1 #avalanche size
#                     if i == 0: #1st site
#                         self.h[0] -= 1
#                         self.z[0] -= 2
#                         self.h[1] += 1
#                         self.z[1] += 1
#                         self.zth[0] = choices(self.zth_choices, 
#                                                       self.zth_weights)[0]
                        
#                     elif i == self.L - 1: #last site
#                         self.h[-1] -= 1
#                         self.z[-1] -= 1
#                         self.z[-2] += 1
#                         self.zth[-1] = choices(self.zth_choices, 
#                                                       self.zth_weights)[0]
                        
#                         #If last site relaxed, steady critical state started
#                         if not self.tc:
#                             self.tc = self.t
                    
#                     else: # all sites in between
#                         self.h[i] -= 1
#                         self.z[i] -= 2
#                         self.h[i+1] += 1
#                         self.z[i+1] += 1
#                         self.z[i-1] += 1
#                         self.zth[i] = choices(self.zth_choices, 
#                                                       self.zth_weights)[0]
#         self.slist.append(s)

# @njit
# def numba_relax2(z, zth, tc, L, t, zth_choices):
#     counter = 0 # count how many sites are allright
#     s = 0 #avalanche size
#     zthhigh = max(zth_choices) +1
#     while counter < L:
        
#         counter = 0 #if repeating, reset counter
        
#         for i in np.arange(0, L, 1):
            
#             if z[i] <= zth[i]: 
#                 counter += 1   
                
#             else:
#                 s +=1 #avalanche size
#                 if i == 0: #1st site
#                     z[0] -= 2
#                     z[1] += 1
#                     zth[0] = randint(1, zthhigh)
                    
#                 elif i == L - 1: #last site
#                     z[-1] -= 1
#                     z[-2] += 1
#                     zth[-1] = randint(1, zthhigh)
                    
#                     #If last site relaxed, steady critical state started
#                     if not tc:
#                         tc = t
                
#                 else: # all sites in between
#                     z[i] -= 2
#                     z[i+1] += 1
#                     z[i-1] += 1
#                     zth[i] = randint(1, zthhigh)
                    
#     return s, z, zth, tc
