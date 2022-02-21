# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:17:54 2022

@author: Stefano
"""

from complexity_functions import *
from complexity_input import *

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from iminuit import Minuit
from uncertainties import ufloat
import matplotlib.pyplot as pl

from tqdm import tqdm
from timeit import default_timer as timer 

#start = timer()
ps = {"text.usetex": True,
        "font.size" : 16,
        "font.family" : "Times New Roman",
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.figsize": [7.5, 6],
        "mathtext.default": "regular"
        }
pl.rcParams.update(ps)
del ps

Llist = Llist_bulkh  #np.array([32, 64, 128, 256, 512]) # Ls
l = np.linspace(1, int(1.1*max(Llist)), 1000)

markers = ["o", "x", "s", "^", "h", "1", "d", "*", "v", "2", "8", "P", "D"]
if len(markers) < len(Llist):
    raise Exception("The number of markers must be same as number of Ls")
if len(markers) > len(Llist):
    markers = markers[:len(Llist)]

T = T_bulkh; t = [i+1 for i in range(T)] # Max time T, time steps
Tlow = Tlow_bulkh
Tlist = sorted([int(np.ceil(T/2**i)) for i in range(len(Llist))])
for i in range(len(Tlist)):
    if Tlist[i] < Tlow: Tlist[i] = int(Tlow)
protect = False

#%% TASK (bulkh) 2a: h vs L
print("\nTASK (bulkh) 2a: Height vs Time for different L\n")

if protect:
    raise Exception("Stop")
#All these lists will be useful later
hlist_list = []
hmean_list = []
hstd_list = []
tc_list = []
slist_list = []

for i in tqdm(range(len(Llist)), "Length"):
    L = Llist[i]
    #prepare initial "average" state
    zs = np.zeros(L, dtype = int) + 1
    inds = np.random.randint(0, L, size = int(0.9 * L))
    zs[inds] = 2
    pile = Ricepile(L)
    pile.set_conf(zs, update_tc = True)
    hlist, hmean, hstd = pile.drive(Tlist[i], addspot = "random height")
    tc = pile.get_tc()
    slist = pile.get_crit_avax()
    hlist_list.append(hlist)
    hmean_list.append(hmean)
    hstd_list.append(hstd)
    tc_list.append(tc)
    slist_list.append(slist)

# np.save("bulkh_hlist_list", hlist_list, allow_pickle = True)
# np.save("bulkh_hmean_list", hmean_list, allow_pickle = True)
# np.save("bulkh_hstd_list",  hstd_list,  allow_pickle = True)
# np.save("bulkh_slist_list", slist_list, allow_pickle = True)

del pile, hlist, hmean, hstd, tc, slist, zs
protect = True

#%% TASK (bulkh) 3a: Avalanche Probability
print("\nTASK 3a: Avalanche Size P(s;L) \n")

Limin = 0
pl.figure("Avalanche LogLog P(s;L)")
sprob = []
for i,L in enumerate(Llist):
    
    slist = slist_list[i]
    ss, P_s = logbin(slist, scale = 1.2, zeros = False)
    sprob.append([ss, P_s])
    if i >= Limin:
        pl.plot(ss, P_s, ".", ls = "--", label = f"L = {L}", zorder = -i)
pl.legend(fancybox = True)
pl.xlabel("s")
pl.ylabel("P(s;L)")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("bulkh_plot_3a_P(s;L)")
pl.show()

# Literature Values
D = 1.243
tau = 2 - 1 /D 

pl.figure("P(s;L) Collapse")
for i in range(len(Llist)):
    L = Llist[i]
    ss = sprob[i][0]; P_s = sprob[i][1]
    
    #Scale
    ss_scaled = ss / L ** D
    P_scaled = P_s * ss ** tau
    
    if i >= Limin:
        pl.plot(ss_scaled, P_scaled, ".", ls = "--", label = f"L = {L}", 
                zorder=-i, alpha = 0.3 + 0.7 *np.log2(L)/np.log2(max(Llist)))
props = dict(boxstyle='round', facecolor='white', alpha=0.5) 
pl.annotate (f"D = {round(D,3)}, "+r"$\tau_s$ = "+f"{round(tau, 3)}", 
             (0.03, 0.93), xycoords = "axes fraction", fontsize = 14,
             bbox = props) 
pl.legend(fancybox = True)
pl.xlabel(r"s/$L^D$")
pl.ylabel(r"$s^{\tau_s}$P(s;L)")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("bulkh_plot_3a_P(s;L)_collapse")
pl.show()

del slist, L
del ss, P_s, ss_scaled, P_scaled


#%% TASK (bulkh) 3b: Avalanche Momenta
print("\nTASK (bulkh) 3b: Avalanche Momenta \n")
#% Non including s = 0
kmax = 10
ks = np.arange(1, kmax + 1)
smoments = []
smoments_std = []
for i in range (len(Llist)):
    smoments.append([]) 
    smoments_std.append([])
    slist_loop = []
    slli = slist_list[i]
    
    L = Llist[i]        
    for k in tqdm(range(1, kmax + 1), f"L = {L}"):
        smoment_list = [s ** k for s in slli]
        smoment_i_k = np.mean(smoment_list)
        smoment_i_k_std = np.std(smoment_list)
        smoments[i].append(smoment_i_k)
        smoments_std[i].append(smoment_i_k_std)
 
smoments = np.array(smoments)
smoments_std = np.array(smoments_std)
smoments_log_std = smoments_std / smoments / np.log(10)
smoments_log = np.log10(smoments)
Llist_log = np.log10(Llist)
# smoments is now a list of len(Llist) sublists, one for each L,
# each sublist having kmax avg moments, one for each k


#%%% 
# step 1, find D(1+k+tau)
pl.figure("log<s^k> = D(1+k-tau) log L ")
Aks = []; Aks_std = []
cks = []
fits = []
Limin = 0
for k in tqdm(range(0, kmax)):
    # Linear Fit, with Ak = D(1+k+tau)
    (Ak, ck), cov = np.polyfit(Llist_log[Limin:], smoments_log[:,k][Limin:], 1, 
                               cov = True)
    perr = np.sqrt(np.diag(cov))
    Ak_err = perr[0]
    
    Aks.append(Ak)
    Aks_std.append(Ak_err)
    cks.append(ck)
    fit_s_k = Llist ** Ak * 10**ck
    fits.append(fit_s_k)
    
    #Plot
    p = pl.plot(Llist[Limin:], smoments[:,k][Limin:], ".", 
                label = f"k = {k+1}")
    pl.plot(Llist, fits[k], "--", color = p[0].get_color())
pl.legend(fancybox = True)
pl.xlabel("Length L")
pl.ylabel(r"$\langle s^k \rangle$")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("bulkh_plot_3b_scale")
del p

# Step 2, find D and tau
#First Perform a Linear Fit
Aks_std = np.array(Aks_std)
(D, c), cov = np.polyfit(ks, Aks, 1, cov = True)
perr = np.sqrt(np.diag(cov))
D_err = perr[0]
c_err = perr[1]
D_u_1 = ufloat(D, D_err)
c_u = ufloat(c, c_err)
tau_u_1 = 1 - c_u / D_u_1

#Then curvefit
def myAk (k, D, t):
    return D*(1+k-t)
bounds = [(1, 1), (3, 5/3)]
p0 = [D_u_1.n, tau_u_1.n]
(D, tau), cov = curve_fit(myAk, ks, Aks, sigma = Aks_std, p0 = p0,
                        bounds = bounds, absolute_sigma = True)
perr = np.sqrt(np.diag(cov))
D_err = perr[0]
tau_err = perr[1]
D_u_2 = ufloat(D, D_err)
tau_u_2 = ufloat(tau, tau_err)

#Then use Minuit
def myChi2 (D, tau):
    expected = D * (1 + ks - tau)
    return stats.chisquare(Aks, expected, ddof = 2)[0]
myChi2.errordef = Minuit.LEAST_SQUARES
m = Minuit(myChi2, D = D, tau = 1 - c/D)
m.limits = ((1, 3), (1, 5/3))
m.strategy = 2
m.migrad()
m.hesse()
D = m.values[0]
tau = m.values[1] 
D_err = m.errors[0] / np.sqrt(kmax)
tau_err = m.errors[1] / np.sqrt(kmax)
D_u_3 = ufloat(D, D_err)
tau_u_3 = ufloat(tau, tau_err)

D_u = (D_u_1 + D_u_2 + D_u_3) / 3
tau_u = (tau_u_1 + tau_u_2 + tau_u_3) / 3
D = D_u.n
tau = tau_u.n

ks2 = np.arange(0, kmax +1)
fit = D * (1 + np.array(ks2) - tau)

pl.figure("D(1+k-tau) vs k")
p = pl.errorbar(ks, Aks, yerr = Aks_std, fmt = ".", capsize = 4)
pl.plot(ks2, fit, "--", 
        label = f"D = {round(D,3)}, "+r"$\tau_s$ = "+f"{round(tau, 3)}")
pl.legend(fancybox = True)
pl.xlabel("Order of Moment k")
pl.ylabel(r"$D(k+1-\tau_s$)")
pl.grid(alpha = 0.4)
pl.savefig("bulkh_plot_3b_scale3")

print(f"My estimates are:\n\
      -  D    = {D_u} \n\
      - tau_s = {tau_u}")

pl.figure("P(s;L) Collapse 2")
for i in range(Limin, len(Llist)):
    L = Llist[i]
    ss = sprob[i][0]; P_s = sprob[i][1]
    
    #Scale
    ss_scaled = ss / L ** D
    P_scaled = P_s * ss ** tau
    if L < 20:
        alpha = 0.5
    else:
        alpha = 1

    pl.plot(ss_scaled, P_scaled, ".", ls = "--", label = f"L = {L}", zorder=-i,
            alpha = alpha)
props = dict(boxstyle='round', facecolor='white', alpha=0.5) 
pl.annotate (f"D = {round(D,3)}, "+r"$\tau_s$ = "+f"{round(tau, 3)}", 
             (0.03, 0.93), xycoords = "axes fraction", fontsize = 14,
             bbox = props)  
pl.legend(fancybox = True)
pl.xlabel(r"$s/L^D$")
pl.ylabel(r"$s^{\tau_s}$P(s;L)")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("bulkh_plot_3b_P(s;L)_collapse_check")
pl.show()

del p, perr, cov, fits, fit_s_k, ks2
del D, tau, D_err, c, c_err, fit
del L, ss, P_s
del props, alpha
#%%
del i, smoments, smoments_log, smoments_std
del Aks, Aks_std, Ak, ks




