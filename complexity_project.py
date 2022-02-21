# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:17:54 2022

@author: Stefano
"""

from complexity_functions import *
from complexity_input import *

import numpy as np
import matplotlib.pyplot as pl

from scipy import stats
from scipy.optimize import curve_fit, minimize, NonlinearConstraint
from iminuit import Minuit

from uncertainties import ufloat

from tqdm import tqdm
from timeit import default_timer as timer 

ps = {"text.usetex": True,
        "font.size" : 16,
        "font.family" : "Times New Roman",
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.figsize": [7.5, 6],
        #"mathtext.default": "default"
        }
pl.rcParams.update(ps)
del ps

Llist = Llist # np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
l = np.linspace(1, int(1.1*max(Llist)), 1000)

markers = ["o", "x", "s", "^", "h", "1", "d", "*", "v", "2", "8", "P", "D"]
if len(markers) < len(Llist):
    raise Exception("The number of markers must be same as number of Ls")
if len(markers) > len(Llist):
    markers = markers[:len(Llist)]

yn = input("Do you want to do the looong realisation tasks now? (y/n)")
M = 0
if yn == "y":
    #For Realisations 
    M = M_in #number of realisations
    n = n
    Tlow = Tlow_M
    Tlist = [int(n * L ** 2) for L in Llist]
    for i in range(len(Tlist)):
        if Tlist[i] < Tlow: Tlist[i] = int(Tlow)
    t = [i+1 for i in range(max(Tlist))]
else:
    #For non-Realisations bit
    T = T
    Tlow = Tlow
    Tlist = sorted([int(np.ceil(T/2**i)) for i in range(len(Llist))])
    for i in range(len(Tlist)):
        if Tlist[i] < Tlow: Tlist[i] = int(Tlow)
    t = [i+1 for i in range(max(Tlist))]

del yn, Tlow
protect = False
protect2 = False

#%% TASK 2a: h vs L
print("\nTASK 2a: Height vs Time for different L\n")
if protect:
    raise Exception ("Stooooop or you will lose data")
#All these lists will be useful later
hlist_list = []
hmean_list = []
hstd_list = []
tc_list = []
slist_list = []


for i in tqdm(range(len(Llist)), "Length"):
    L = Llist[i]
    pile = Ricepile(L)
    hlist, hmean, hstd = pile.drive(Tlist[i])
    tc = pile.get_tc()
    slist = pile.get_crit_avax()
    hlist_list.append(hlist)
    hmean_list.append(hmean)
    hstd_list.append(hstd)
    tc_list.append(tc)
    slist_list.append(slist)    


del pile, hlist, hmean, hstd, tc, slist

if not M:
    np.save("hlist_list", hlist_list, allow_pickle = True)
    np.save("hmean_list", hmean_list, allow_pickle = True)
    np.save("hstd_list",  hstd_list,  allow_pickle = True)
    np.save("slist_list", slist_list, allow_pickle = True)
    np.save("tc_list",    tc_list,    allow_pickle = True)
# #%%
# hlist_list = np.load("hlist_list.npy", allow_pickle = True)
# hmean_list = np.load("hmean_list.npy", allow_pickle = True)
# hstd_list = np.load("hstd_list.npy",  allow_pickle = True)
# slist_list = np.load("slist_list.npy", allow_pickle = True)
# tc_list = np.load("tc_list.npy", allow_pickle = True)
protect = True
#%% TASK 2b-2d: <t_c> vs L, havg vs t & Collapse
print("\nTASK 2b: Average Cross-over Time vs L")
print("         &")
print("TASK 2d: Data Collapse \n")

if protect2:
    raise Exception ("Stooooop or you will lose data")

if M:
    tclist = []
    hlist_Mlist = []
    havglist_list = []
for i in range(len(Llist)):
    L = Llist[i]
    hlist_Mlist.append([])
    tclist.append([])
    hstd_sum2 = (hstd_list[i]) ** 2
    for j in tqdm(range(M-1), f"Realisations for L = {Llist[i]}"):
        pile = Ricepile(L)
        hlist, hmean, hstd = pile.drive(Tlist[i])
        tc = pile.get_tc()
        slist = pile.get_crit_avax()
        tclist[i].append(tc)
        hlist_Mlist[i].append(hlist)
        
        #Add info to lists from task 2a, used also in other sections:
        hlist_list[i] += hlist[tc:] #add new samples
        hmean_list[i] += hmean #as equally sampled, add and at end divide
        hstd_sum2 += hstd ** 2 #sum squares of std, 
        slist_list[i] += slist
    
    hmean_list[i] /= M
    hstd_list[i] = np.sqrt(hstd_sum2) / np.sqrt(M)
        
    #append list havg vs t for certain L
    havglist_list.append( np.mean(hlist_Mlist[i], axis = 0 ))
    
    pl.figure("Avg Height vs time")
    pl.plot(t[:len(havglist_list[i])], havglist_list[i], label = f"L = {L}",
             zorder = 2 + 1/(i+1))
    
np.save(f"havglist_list_{M,n}", havglist_list, allow_pickle = True)
protect2 = True

pl.figure("Avg Height vs time")
pl.xlabel("t")
pl.ylabel(r"Average height $\tilde{h}$")
pl.grid(alpha = 0.4 )
pl.legend(loc = "lower right", fancybox = True)
pl.savefig("plot_2d_avgheight")
#%%% plot collapse
for i, L in enumerate(Llist):
    ind = n*L**2
    
    pl.figure("Data Collapse")
    pl.plot(np.array(t[:ind])/ L**2, 
            np.array(havglist_list[i][:ind]) / L, 
            label = f"L = {L}")
    
    pl.figure("Data Collapse log")
    pl.plot(np.array(t[:ind])/ L**2, 
            np.array(havglist_list[i][:ind]) / L, 
            label = f"L = {L}")
    
    pl.figure("Data Collapse 2")
    pl.plot(np.array(t[:ind])/ L**2, 
            np.array(havglist_list[i][:ind]) / L, 
            label = f"L = {L}")
    
    pl.figure("Data Collapse log2")
    pl.plot(np.array(t[:ind])/ L**2, 
            np.array(havglist_list[i][:ind]) / L, 
            label = f"L = {L}")

pl.figure("Data Collapse")
pl.xlabel(r"t/$L^2$")
pl.ylabel(r"$\tilde{h}$/L")
pl.legend(fancybox = True)
pl.grid(alpha = 0.4 )
pl.savefig("plot_2d_collapse")

pl.figure("Data Collapse log")
pl.xscale("log")
pl.yscale("log")
pl.xlabel(r"t/$L^2$")
pl.ylabel(r"$\tilde{h}$/L")
pl.legend(fancybox = True)
pl.grid(alpha = 0.4 )
pl.savefig("plot_2d_collapse_log")

def mysqrt (x, a):
    return np.sqrt(a * x)
a1, aerr1 = curve_fit(mysqrt, np.array(t[ :int(0.9*tc_list[-1]) ])/ L**2, 
                 np.array(havglist_list[-1][ :int(0.9*tc_list[-1]) ])/L)
a2, aerr2 = curve_fit(mysqrt, np.array(t[ :int(0.9*tc_list[-2]) ])/ L**2, 
                 np.array(havglist_list[-2][ :int(0.9*tc_list[-2]) ])/L)
a = np.mean([a1, a2])
aerr = np.std([a1, a2])
print("h/L appr sqrt(A t/L^2) with A = ", a, " +/- ", aerr)

pl.figure("Data Collapse 2")
g, = pl.plot(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2, 
             np.sqrt(a * np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2), 
             ls = "--", color = "r", label = r"$ \sim \sqrt{x}$") 
pl.legend(loc = "lower right", fancybox = True)
pl.xlabel(r"t/$L^2$")
pl.ylabel(r"$\tilde{h}$/L")
pl.legend(fancybox = True)
pl.grid(alpha = 0.4 )
pl.savefig("plot_2d_collapse")

pl.figure("Data Collapse log2")
pl.plot(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2, 
             np.sqrt(a * np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2), 
             ls = "--", color = "r", label = r"$ \sim \sqrt{x}$") 
pl.legend(loc = "lower right", fancybox = True)
pl.xscale("log")
pl.yscale("log")
pl.xlabel(r"t/$L^2$")
pl.ylabel(r"$\tilde{h}$/L")
pl.legend(fancybox = True)
pl.grid(alpha = 0.4 )
pl.savefig("plot_2d_collapse_log2")

#%%% tc vs L
# mean with axis 1 computes mean of each sublist
tcavg = np.mean(tclist, axis = 1)
tcstd = np.std(tclist, axis = 1)
np.save(f"tcavg_{M,n}", tcavg, allow_pickle = True)
np.save(f"tcstd_{M,n}", tcavg, allow_pickle = True)

def mytc (x, a):
    return a * x**2

a, aerr = curve_fit(mytc, Llist, tcavg, sigma = tcstd, 
                    absolute_sigma = True)
A = a[0]
print("tc appr a L^2 with a =", a, " +/- ", aerr)
tc_est = a * l ** 2
pl.figure("Average crossover Time vs Length") 
pl.errorbar(Llist, tcavg, 
            yerr =  tcstd,
            fmt = ".", capsize = 4)
p = pl.plot(l, tc_est, "--", label = r"$ A_t L^2$")
color = p[0].get_color()
pl.xlabel("L")
pl.ylabel(r"$\tilde {t_c} $")
pl.legend(fancybox = True)
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
note = f"$A_t$ = {round(A,6)}"
pl.annotate (note, (0.95, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "right",
             multialignment = "left", fontsize = 12) 
pl.xscale("log")
pl.yscale("log")
pl.xlabel("L")
pl.grid(alpha = 0.4 ) 
pl.savefig("plot_2b_tc")      
pl.show()

del pile, tc, hlist, hmean, hstd, hstd_sum2, hlist_Mlist, i, j, L
#del slist
del tcavg, tcstd, tc_est, a, A
del p, color

#%% TASK 2e: <h> expansion fit
print("\nTASK 2e: Mean height vs L with fit\n")  
# The mean and std of the height in critical state are computed in task 2a, 
# as hmean_list and hstd_list

# 1st Method
p0 = [1.75, 0.2, 0.5]
(a0, a1, w1), (a0_1, a1_1, w1_1) = fit_h1(hmean_list, Llist, hstd_list, p0)
#print(a0_1, a1_1, w1_1)

#2nd Method
step = 0.005; start = max(max(np.array(hmean_list)/Llist), 1.72)
p0 = [a0, a1, w1]
values, (a0_2, a1_2, w1_2) = fit_h2(hmean_list, Llist, hstd_list, p0, step)
#print(a0_2, a1_2, w1_2)

a0_u = (a0_1 + a0_2)/2
a1_u = (a1_1 + a1_2)/2
w1_u = (w1_1 + w1_2)/2
print(r"Estimates for havg vs L with havg= a0 L - a1 L^(-w1):")
print(f"    a0 = {a0_u}\n",f"    a1 = {a1_u}\n",f"    w1 = {w1_u}")
a0, a1, w1 = a0_u.nominal_value, a1_u.nominal_value, w1_u.nominal_value
expected = hMean(Llist, a0, a1, w1)
Chi2, pvalue = stats.chisquare(hmean_list, expected, ddof = 3)
print(f"Goodness of Fit: Chi2 = {Chi2},   pvalue = {pvalue}")
del p0, values, a0_1, a1_1, w1_1
del a0_2, a1_2, w1_2
del expected, Chi2, pvalue 


pl.figure("Mean height vs L with fit")
pl.errorbar(Llist, hmean_list, yerr = hstd_list, 
            fmt = ".", capsize = 4, label = "data")
p = pl.plot(l, hMean(l, a0, a1, w1), "--", 
        label = r"$a_0 L \left[1 - a_1 L^{-\omega_1} \right]$")
color = p[0].get_color()
pl.legend(fancybox = True)
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
magn_a0 = int(abs(np.log10(a0_u.s))) + 1
magn_a1 = int(abs(np.log10(a1_u.s))) + 2
magn_w1 = int(abs(np.log10(w1_u.s))) + 2
note = f"$a_0$ = {round(a0_u.n, magn_a0)} $\pm$ {round(a0_u.s, magn_a0)}"
note += "\n"
note +=f"$a_1$ = {round(a1_u.n, magn_a1)} $\pm$ {round(a1_u.s, magn_a1)}"
note += "\n"
note +=f"$\omega_1$ = {round(w1_u.n, magn_w1)} $\pm$ {round(w1_u.s, magn_w1)}"
pl.annotate (note, (0.95, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "right",
             multialignment = "left", fontsize = 12)  
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \langle h \rangle$")
pl.savefig("plot_2e_avgh_vs_L")
pl.show()


fig = pl.figure("a0's Best fit")
pl.plot(Llist, (1- np.array(hmean_list)/a0/Llist), ".", 
        label = r"Estimated with $a_0$"+f" = {round(a0, 3)}")
p = pl.plot(l, a1 * l**(-w1), "--", label = r"$a_1 \times L^{-\omega_1}$")
color = p[0].get_color()
pl.legend(fancybox = True)
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
note =f"$a_1$ = {round(a1_u.n, magn_a1)} $\pm$ {round(a1_u.s, magn_a1)}"
note += "\n"
note +=f"$\omega_1$ = {round(w1_u.n, magn_w1)} $\pm$ {round(w1_u.s, magn_w1)}"
pl.annotate (note, (0.10, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "left",
             multialignment = "left", fontsize = 12) 
pl.xlabel("L")
pl.ylabel(r"$1 - \frac{\langle h \rangle}{a_0 L}$")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
fig.tight_layout()
pl.savefig("plot_2e_fora02")

#Plot h two fits together, cause, why not
fig, axs = pl.subplots(1, 2)
pl.subplot(1, 2, 1)
pl.annotate("a)", (-0.2, 1.02), xycoords = "axes fraction", fontsize = 12)
pl.errorbar(Llist, hmean_list, yerr = hstd_list, 
            fmt = ".", capsize = 4, label = "data")
p = pl.plot(l, hMean(l, a0, a1, w1), "--", 
        label = r"$a_0 L \left[1 - a_1 L^{-\omega_1} \right]$")
color = p[0].get_color()
pl.legend(fancybox = True)
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
note = f"$a_0$ = {round(a0_u.n, magn_a0)} $\pm$ {round(a0_u.s, magn_a0)}"
note += "\n"
note +=f"$a_1$ = {round(a1_u.n, magn_a1)} $\pm$ {round(a1_u.s, magn_a1)}"
note += "\n"
note +=f"$\omega_1$ = {round(w1_u.n, magn_w1)} $\pm$ {round(w1_u.s, magn_w1)}"
pl.annotate (note, (0.95, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "right",
             multialignment = "left", fontsize = 12)  
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.xscale("log")
pl.yscale("log")
pl.ylabel(r"$ \langle h \rangle$")

pl.subplot(1,2,2)
pl.annotate("b)", (-0.2, 1.02), xycoords = "axes fraction",fontsize = 12)
pl.plot(Llist, (1- np.array(hmean_list)/a0/Llist), ".", 
        label = r"Estimated with $a_0$"+f" = {round(a0, 3)}")
p = pl.plot(l, a1 * l**(-w1), "--", label = r"$a_1 \times L^{-\omega_1}$")
color = p[0].get_color()
pl.legend(fancybox = True)
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
note = f"$a_1$ = {round(a1_u.n, magn_a1)} $\pm$ {round(a1_u.s, magn_a1)}"
note += "\n"
note +=f"$\omega_1$ = {round(w1_u.n, magn_w1)} $\pm$ {round(w1_u.s, magn_w1)}"
pl.annotate (note, (0.10, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "left",
             multialignment = "left", fontsize = 12) 
pl.xlabel("L")
pl.ylabel(r"$1 - \frac{\langle h \rangle}{a_0 L}$")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)

fig.set_tight_layout(True)
pl.savefig("plot_2e_BEST_h_fit_")

del fig, axs, p, color, props, note
del magn_a0, magn_a1, magn_w1
del a0_u, a1_u, w1_u
#%% TASK Additional: new heights collapse
if M:
    for i, L in enumerate(Llist):
        ind = n*L**2
        
        pl.figure("Data Collapse New")
        pl.plot(np.array(t[:ind])/ L**2, 
                np.array(havglist_list[i][:ind]) / hMean(L, a0, a1, w1), 
                label = f"L = {L}")
        
        pl.figure("Data Collapse New log")
        pl.plot(np.array(t[:ind])/ L**2, 
                np.array(havglist_list[i][:ind]) / hMean(L, a0, a1, w1), 
                label = f"L = {L}")
    
    def mysqrt (x, a):
        return a*np.sqrt(x)
    a, aerr = curve_fit(mysqrt, np.array(t[ :tc_list[-1] ])/ L**2, 
                     np.array(havglist_list[-1][ :tc_list[-1] ])/L)
    print("h/L appr a t/L^2 with a = ", a, " +/- ", aerr)
    b = a * L / hMean(L, a0, a1, w1)
    pl.figure("Data Collapse New")
    g, = pl.plot(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2, 
                 b * np.sqrt(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2),
                 ls = "--", color = "r", label = r"$ \sim \sqrt{x}$") 
    pl.legend(loc = "lower right", fancybox = True)
    pl.xlabel(r"t/$L^2$")
    pl.ylabel(r"$\tilde{h}/ \langle h \rangle$")
    pl.legend(fancybox = True)
    pl.grid(alpha = 0.4 )
    pl.savefig("plot_2d_extra_collapse")
    
    pl.figure("Data Collapse New log")
    pl.plot(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2, 
            b * np.sqrt(np.array(t[ :int(1.1*tc_list[-1]) ])/ L**2), 
                 ls = "--", color = "r", label = r"$ \sim \sqrt{x}$") 
    pl.legend(loc = "lower right", fancybox = True)
    pl.xscale("log")
    pl.yscale("log")
    pl.xlabel(r"t/$L^2$")
    pl.ylabel(r"$\tilde{h}/ \langle h \rangle$")
    pl.legend(fancybox = True)
    pl.grid(alpha = 0.4 )
    pl.savefig("plot_2d_extra_collapse_log2")
    
    del a, b, aerr, i, L, ind
    
#%% TASK 2f: Stds and mean slope
print("\nTASK 2f: Mean Height Std vs L\n")  

#Fit std_h
bounds = ([0, 0],[np.inf, 1]) #A between 0 and inf, k between 0 and 1
[A, chi], cov = curve_fit(h_std, Llist, hstd_list)
perr = np.sqrt(np.diag(cov))
#0.2412 and 0.2398 were obtained in first two repetitions, and here hardcoded
chi_u = ufloat(np.mean([0.2412, 0.2398, chi]), 
               max(perr[1], np.std([0.2412, 0.2398, chi])))
A_u = ufloat(A, perr[0])
chi = chi_u.n

magn_k = int(abs(np.log10(perr[1])))# + 1

fig = pl.figure("Mean height STD vs L")
pl.plot(Llist, hstd_list, ".")
p = pl.plot(l, h_std(l, A, chi), "--", 
        label = r"$\sim L^{\chi}$")
color = p[0].get_color()
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
note = f"$\chi$ = {round(chi_u.n, magn_k)} $\pm$ {round(chi_u.s, magn_k)}"
pl.annotate (note, (0.90, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "right",
             multialignment = "left", fontsize = 12) 
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \sigma_{h} $")
pl.legend(fancybox = True)
pl.xscale("log")
pl.yscale("log")
fig.tight_layout()
pl.savefig("plot_2f_std_h")
pl.show()
del fig, p, note, color, props

print(f"The std sigma_h appears to scale as {A_u}L^({chi_u})  \n")

print(f"As the average slope zavg = havg/L, we should have:\n\
      - zavg -> constant (a0) \n\
      - sigma_z = sigma_h/L \u007E L^({chi_u-1})-> 0")

zmean_list = np.array(hmean_list) / Llist
zstd_list = np.array(hstd_list) / Llist

fig = pl.figure("Mean slope <z> vs logL", figsize = [7.5, 4])
pl.errorbar(Llist, zmean_list, yerr = zstd_list, 
            fmt = ".", capsize = 4, label = "data")
pl.xscale("log")
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \langle z \rangle $ ")
pl.hlines(a0, 0, 1.1 * max(Llist), "r", "dashed", label = "$a_0$")
pl.legend()
pl.ylim(ymin = 1.0)
fig.tight_layout()
pl.savefig("plot_2f_avgz_logL")
pl.show()

pl.figure("Mean slope STD vs L")
pl.plot(Llist, zstd_list, ".")
pl.plot(l, h_std(l, A, chi)/l, "--",
        label = r"$\sigma_h / L$")
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \sigma_Z $ ")
pl.xscale("log")
pl.yscale("log")
pl.xlim(0.5 * min(Llist), 1.1 * max(Llist))
pl.ylim(pl.ylim()[0],  1.5 * max(zstd_list))
pl.legend(fancybox = True)
pl.savefig("plot_2f_std_z")
pl.show()


#Plot stds together
fig, axs = pl.subplots(1, 2, figsize = [10, 5])
pl.subplot(1, 2, 1)
pl.annotate("a)", (-0.2, 1.02), xycoords = "axes fraction", 
            fontsize = 12)
pl.plot(Llist, hstd_list, ".")
p = pl.plot(l, h_std(l, A, chi), "--", 
        label = r"$\sim L^{\chi}$")
color = p[0].get_color()
props = dict(boxstyle='round', facecolor='white', edgecolor = color, ls = "--",
             alpha = 0.8) 
#note = "A = {:.1u}".format(A_u) + "\n"
note = r"$\chi$ = "+f"{round(chi_u.n, magn_k)} $\pm$ {round(chi_u.s, magn_k)}"
pl.annotate (note, (0.90, 0.05), xycoords = "axes fraction", bbox = props,
             verticalalignment = "bottom", horizontalalignment = "right",
             multialignment = "left", fontsize = 12) 
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \sigma_{h} $")
pl.legend(fancybox = True)
pl.xscale("log")
pl.yscale("log")

pl.subplot(1,2,2)
pl.annotate("b)", (-0.2, 1.02), xycoords = "axes fraction",
            fontsize = 12)
pl.plot(Llist, zstd_list, ".")
pl.plot(l, h_std(l, A, chi)/l, "--",
        label = r"$\sigma_h / L$")
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \sigma_Z $ ")
pl.xscale("log")
pl.yscale("log")
pl.xlim(0.5 * min(Llist), 1.1 * max(Llist))
pl.ylim(pl.ylim()[0],  1.5 * max(zstd_list))
pl.legend(fancybox = True)

fig.set_tight_layout(True)
pl.savefig("plot_2f_BEST_stds_")


#Plot z stuff together, cause stil don't know
fig, axs = pl.subplots(1, 2)
pl.subplot(1, 2, 1)
pl.annotate("a)", (-0.2, 1.02), xycoords = "axes fraction", 
            fontsize = 12)
pl.errorbar(Llist, zmean_list, yerr = zstd_list, 
            fmt = ".", capsize = 4, label = "data")
pl.xscale("log")
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \langle z \rangle $ ")
pl.hlines(a0, 0, 1.1 * max(Llist), "r", "dashed", label = "$a_0$")
pl.legend()
pl.ylim(ymin = 0)

pl.subplot(1,2,2)
pl.annotate("b)", (-0.2, 1.02), xycoords = "axes fraction",
            fontsize = 12)
pl.plot(Llist, zstd_list, ".")
pl.plot(l, h_std(l, A, chi)/l, "--",
        label = r"$\sigma_h / L$")
pl.grid(alpha = 0.4)
pl.xlabel("L")
pl.ylabel(r"$ \sigma_Z $ ")
pl.xscale("log")
pl.yscale("log")
pl.xlim(0.5 * min(Llist), 1.1 * max(Llist))
pl.ylim(pl.ylim()[0],  1.5 * max(zstd_list))
pl.legend(fancybox = True)

fig.set_tight_layout(True)
pl.savefig("plot_2f_BEST_zstuff")
                                      
del bounds, cov, perr, A_u, chi_u
del zmean_list, zstd_list
del fig, axs, p, color, props, note

#%% TASK 2g: Height Probability
print("\nTASK 2g: Height P(h;L) \n")  

# pl.figure("P(h;L)")
# hprob = []
# for i in range(len(Llist)):
#     L = Llist[i]
#     tc = tc_list[i]
#     hmin_i = min(hlist_list[i][tc:]); edgemin = hmin_i - 0.5
#     hmax_i = max(hlist_list[i][tc:]); edgemax = hmax_i + 0.5
#     Nbins = int(hmax_i - hmin_i + 1)
#     range_i = (edgemin, edgemax)
#     P, hs, patch = pl.hist(hlist_list[i][tc:], Nbins, range_i, 
#                            density = True, label = f"L = {L}")

#     hs = np.array(hs[:-1]) +0.5
#     hprob.append([hs, P])

# pl.plot(a0 * l, 1 / (np.sqrt(2*np.pi) * h_std(l, A, chi)), "--", c = "Red",
#         label = r"$[\sqrt{2\pi} \sigma_h ]^{-1}$", alpha = 0.3)    
# pl.legend(fancybox = True)
# pl.xlabel("h")
# pl.ylabel(r"$P(h;L)$")
# pl.grid(alpha = 0.4)
# pl.savefig("plot_2g_P(h;L).pdf")
# pl.show()

pl.figure("P(h;L)_logL")
hprob = []
for i in range(len(Llist)):
    L = Llist[i]
    tc = tc_list[i]
    hmin_i = min(hlist_list[i][tc:]); edgemin = hmin_i - 0.5
    hmax_i = max(hlist_list[i][tc:]); edgemax = hmax_i + 0.5
    Nbins = int(hmax_i - hmin_i + 1)
    range_i = (edgemin, edgemax)
    P, hs, patch = pl.hist(hlist_list[i][tc:], Nbins, range_i, 
                            density = True, label = f"L = {L}")

    hs = np.array(hs[:-1]) +0.5
    hprob.append([hs, P])
legend1 = pl.legend(fancybox = True)
pl.gca().add_artist(legend1)
g, = pl.plot(hMean(l,a0,a1, w1), 1 / (np.sqrt(2*np.pi) * h_std(l, A, chi)), 
             "--", c = "Red",label = r"$[\sqrt{2\pi} \sigma_h ]^{-1}$", 
             alpha = 0.5)    
pl.legend(handles = [g], loc = "upper left", fancybox = True)
pl.xlabel("h")
pl.ylabel(r"$P(h;L)$")
pl.grid(alpha = 0.4)
pl.xscale("log")
pl.savefig("plot_2g_P(h;L)_logL.pdf")
pl.show()


pl.figure("Probability Collapse", tight_layout = True)
for i in range(len(Llist)):
    L = Llist[i]
    hs = hprob[i][0]; P = hprob[i][1]
    hs_scaled = ( hs - hMean(L, a0, a1, w1) ) / h_std(L, A, chi)
    P_scaled = P * h_std(L, A, chi)
    pl.plot(hs_scaled, P_scaled, markers[i], label = f"L = {L}", alpha = 0.7)
x = np.linspace(-7.5, 7.5, 100)
legend1 = pl.legend(fancybox = True)
pl.gca().add_artist(legend1)
g, = pl.plot(x, stats.norm.pdf(x, 0, 1), ls = "--", color = "red", alpha = 0.5, 
        label = r"$\frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$") 
pl.legend(handles = [g], loc = "upper left")
pl.xlabel(r"$(h-\langle h \rangle)/\sigma_h$")
pl.ylabel(r"$ \sigma_h P(h;L) $")
pl.grid(alpha = 0.4)
pl.savefig("plot_2g_P(h;L)_collapse")
pl.show()
del x, g, legend1

# #Plot Probability stuff together, cause stil don't know
# fig, axs = pl.subplots(1, 2)
# pl.subplot(1, 2, 1)
# pl.annotate("a)", (-0.1, 1.02), xycoords = "axes fraction", fontsize = 12)
# hprob = []
# for i in range(len(Llist)):
#     L = Llist[i]
#     tc = tc_list[i]
#     hmin_i = min(hlist_list[i][tc:]); edgemin = hmin_i - 0.5
#     hmax_i = max(hlist_list[i][tc:]); edgemax = hmax_i + 0.5
#     Nbins = int(hmax_i - hmin_i + 1)
#     range_i = (edgemin, edgemax)
#     P, hs, patch = pl.hist(hlist_list[i][tc:], Nbins, range_i, 
#                            density = True, label = f"L = {L}")

#     hs = np.array(hs[:-1]) +0.5
#     hprob.append([hs, P])
# legend1 = pl.legend(fancybox = True)
# pl.gca().add_artist(legend1)
# g, =pl.plot(a0 * l, 1 / (np.sqrt(2*np.pi) * h_std(l, A, chi)), "--", c = "Red",
#         label = r"$[\sqrt{2\pi} \sigma_h ]^{-1}$", alpha = 0.5)    
# pl.legend(handles = [g], loc = "upper left", fancybox = True)
# pl.xlabel("h")
# pl.ylabel(r"$P(h;L)$")
# pl.grid(alpha = 0.4)
# pl.xscale("log")

# pl.subplot(1,2,2)
# pl.annotate("b)", (-0.1, 1.02), xycoords = "axes fraction", fontsize = 12)
# for i in range(len(Llist)):
#     L = Llist[i]
#     hs = hprob[i][0]; P = hprob[i][1]
#     hs_scaled = ( hs - hMean(L, a0, a1, w1) ) / h_std(L, A, chi)
#     P_scaled = P * h_std(L, A, chi)#L**(chi)
#     pl.plot(hs_scaled, P_scaled, markers[i], label = f"L = {L}", alpha = 0.7)
# x = np.linspace(-7.5, 7.5, 100)
# legend1 = pl.legend(fancybox = True)
# pl.gca().add_artist(legend1)
# g, = pl.plot(x, stats.norm.pdf(x, 0, 1), ls = "--", color = "red", alpha = 0.5, 
#         label = r"$\frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$") 
# pl.legend(handles = [g], loc = "upper left", fancybox = True)
# pl.xlabel(r"$\frac{h-\langle h \rangle}{\sigma_h}$")
# pl.ylabel(r"$ \sigma_h P(h;L) $")
# pl.grid(alpha = 0.4)
# pl.savefig("plot_2g_P(h;L)_collapse1")
# pl.show()

# fig.set_tight_layout(True)
# pl.savefig("plot_2g_BEST_hprob")
# pl.show()

# del i, tc, L, hs, hs_scaled, P_scaled, P
# del fig, axs, legend1, x, g

#Test Skewness with scipy
hscaled_all = []
for i, (L,tc) in enumerate(zip(Llist, tc_list)):
    hscaled_i = (np.array(hlist_list[i][tc:]) - hMean(L, a0, a1, w1))
    hscaled_i /= h_std(L, A, chi)
    hscaled_all.extend(hscaled_i)

skewness = stats.skew(hscaled_all, bias = False)
skewtest = stats.skewtest(hscaled_all)
print(f"Skewness \gamma: {skewness} \n Null Hypotheisis of Gaussian", skewtest)
del i, L, hscaled_i, hscaled_all, skewness, skewtest

#%% TASK 3a: Avalanche Probability
print("\nTASK 3a: Avalanche Size P(s;L) \n")
  
pl.figure("Avalanche LogLog P(s;L)")
sprob = []
for i in range(len(Llist)):
    L = Llist[i]
    
    slist = slist_list[i]
    ss, P_s = logbin(slist, scale = 1.2, zeros = False)
    sprob.append([ss, P_s])

    #pl.step(ss, P_s, label = f"L = {L}", where = "post")
    pl.plot(ss, P_s, ".", ls = "--", label = f"L = {L}", zorder = -i)
pl.legend(fancybox = True)
pl.xlabel("s")
pl.ylabel("P(s;L)")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("plot_3a_P(s;L)")
pl.show()

# Literature Values
D = 2.25
tau = 1.555

pl.figure("P(s;L) Collapse")
for i in range(len(Llist)):
    L = Llist[i]
    ss = sprob[i][0]; P_s = sprob[i][1]
    
    #Scale
    ss_scaled = ss / L ** D
    P_scaled = P_s * ss ** tau
    
    #pl.step(ss, P_s, label = f"L = {L}", where = "post")
    pl.plot(ss_scaled, P_scaled, ".", ls = "--", label = f"L = {L}", zorder=-i,
            alpha = 0.3 + 0.7 *np.log2(L)/np.log2(max(Llist)))
pl.legend(fancybox = True)
pl.xlabel(r"s/$L^D$")
pl.ylabel(r"$s^{\tau_s}$P(s;L)")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("plot_3a_P(s;L)_collapse")
pl.show()

del slist, L
del ss, P_s, ss_scaled, P_scaled


#%% TASK 3b: Avalanche Momenta
print("\nTASK 3b: Avalanche Momenta \n")
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
    # for slli in slist_list[i]:
    #      if slli:
    #          slist_loop.append(slli)
    
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
#%%% WITH ALL L

# step 1, find D(1+k+tau)
pl.figure("log<s^k> = D(1+k-tau) log L ")
Aks = []
Aks_std = []
fits = []
for k in tqdm(range(0, kmax)):
    # Linear Fit, with Ak = D(1+k+tau)
    (Ak, c), cov = np.polyfit(Llist_log, smoments_log[:,k], 1,
                              w = np.log(np.log2(Llist)), cov = True)
    perr = np.sqrt(np.diag(cov))
    Ak_err = perr[0]
    
    Aks.append(Ak)
    Aks_std.append(Ak_err)
    fit_s_k = Llist ** Ak * 10**c
    fits.append(fit_s_k)
    
    #Plot
    p = pl.plot(Llist, smoments[:,k], ".", label = f"k = {k+1}")
    pl.plot(Llist, fits[k], "--", color = p[0].get_color())
pl.legend(fancybox = True)
pl.xlabel("L")
pl.ylabel(r"$\langle s^k \rangle$")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("plot_3b_scale")
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
print(D_u_1)
print(tau_u_1)

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
print(D_u_2)
print(tau_u_2)

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
pl.savefig("plot_3b_scale2")

print(f"My estimates are:\n\
      -  D    = {D_u} \n\
      - tau_s = {tau_u}")
      
pl.figure("P(s;L) Collapse 2")
for i in range(len(Llist)):
    L = Llist[i]
    ss = sprob[i][0]; P_s = sprob[i][1]
    
    #Scale
    ss_scaled = ss / L ** D
    P_scaled = P_s * ss ** tau
    if L < 10:
        alpha = 0.5
    else:
        alpha = 1
    #pl.step(ss, P_s, label = f"L = {L}", where = "post")
    pl.plot(ss_scaled, P_scaled, ".", ls = "--", label = f"L = {L}", zorder=-i,
            alpha = 0.3 + 0.7 *np.log2(L)/np.log2(max(Llist)))
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
pl.savefig("plot_3b_P(s;L)_collapse_check")
pl.show()

del p, perr, fit, fit_s_k, ks2
del D, tau, D_err, tau_err, c, c_err
del L, ss, P_s
del props, alpha

#%%% ONLY WITH BIG L
# step 1, find D(1+k+tau)
Limin = Limin_in
pl.figure("log<s^k> = D(1+k-tau) log L ")
Aks = []; Aks_std = []
cks = []
fits = []
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
pl.xlabel("L")
pl.ylabel(r"$\langle s^k \rangle$")
pl.xscale("log")
pl.yscale("log")
pl.grid(alpha = 0.4)
pl.savefig("plot_3b_scale_big")
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
pl.savefig("plot_3b_scale2_big")

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
    #pl.step(ss, P_s, label = f"L = {L}", where = "post")
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
pl.savefig("plot_3b_P(s;L)_collapse_mom_big")
pl.show()

del p, perr, cov, fits, fit_s_k, ks2
del D, tau, D_err, tau_err, c, c_err, fit
del L, ss, P_s
del props, alpha
#%%
del i, smoments, smoments_log, smoments_std
del Aks, Aks_std, Ak, ck, cks
del D_u, tau_u

















