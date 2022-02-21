# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:18:00 2022

@author: Stefano
"""
from complexity_functions import *
from complexity_input import *
import numpy as np
import pandas as pd

from uncertainties import ufloat

from timeit import default_timer as timer 
from tqdm import tqdm
import matplotlib.pyplot as pl

#%% TEST1: HEIGHTS
print("\nTASK 1 - TESTING 1: Height\n")
mean_mean_height_16 = []
mean_mean_height_32 = []

for i in range(1):
    pile = Ricepile(16)
    hlist, mean_height_16, std_16 = pile.drive(T_A)
    mean_mean_height_16.append(mean_height_16)
    
    pile = Ricepile(32)
    hlist, mean_height_32, std_32 = pile.drive(T_A)
    mean_mean_height_32.append(mean_height_32)
    
    print(f"The Ricepiles, after {T_A} grains added, present:")
    print(f"-for L = 16 -> <h> = {round(mean_height_16,4)}, sigmah = {round(std_16,4)}")
    print(f"-for L = 32 -> <h> = {round(mean_height_32,4)}, sigmah = {round(std_32,4)}")
    print("Exact values should be approximately 26.5 and 53.9 respectively, so \
the differences are:")
    print(f"{abs(26.5-round(mean_height_16,4))}")
    print(f"{abs(53.9-round(mean_height_32,4))}")

print(f"After 10 iterations, each with {T_A} grains added, we have:")
print(f"-for L = 16 -> <<h>>   = {round(np.mean(mean_mean_height_16),6)}, \
                      sigma<h> = {np.std(mean_mean_height_16)}")
print(f"-for L = 32 -> <<h>>   = {round(np.mean(mean_mean_height_32),6)}, \
                      sigma<h> = {np.std(mean_mean_height_32)}")
del pile, hlist, mean_height_16, mean_height_32, std_16, std_32
del i, mean_mean_height_16, mean_mean_height_32 

#%% TEST2: BTW HEIGHTS & AVALANCHES
print("\n\n\nTASK 1 - TESTING 2: BTW Height & Avalanches\n")
T = int(10e4); t = [i+1 for i in range(T)] # Max time T, time steps

print(f"The Ricepiles, after {T} grains added, present:")
for L in tests_B:
    pile = Ricepile(L, zthmax = 1)
    hlist, meanh, std = pile.drive(T)
    tc = pile.get_tc()
    print(f"-for L = {L} -> <h>_c = {round(meanh,3)}")
    print(f"              -> std is {std}.")
    print(f"              -> 2tc/[L(L+1)] is {2 * (tc) / L / (L+1)}.")

del L, pile, T 
del hlist, meanh, std, tc

#%% TEST3.2 RECURRENT CONFIGURATIONS
print("\n\n\nTASK 1 - TESTING 3: Recurrent Confs\n")

def recurN (L):
    phi = (1 + np.sqrt(5)) / 2
    A = phi * (1 + phi) ** L
    B = A + 1/A
    NR = B / np.sqrt(5)
    return int(NR)

x = np.linspace(1, tests[-1], 5 * tests[-1])
ns = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1]) * Ns_C

Nconfs = []
Nconfs_summary = []
print("Number of Recurrent Configurations Found")
for i,L in tqdm(enumerate(tests_C)):
    Nconfs.append([f"{L}"])
    Nconfs_summary.append([f"{L}"])
    pile = Ricepile(L)
    for j in range(Ns_C):
        zs = np.zeros(L, dtype = int) + 1
        pile.set_conf(zs)
        hstuff,Nconf=pile.drive(T_C, trackconf = True)
        Nconfs[i].append(Nconf)
        zs = np.zeros(L, dtype = int) + 2
        pile.set_conf(zs)
        hstuff, Nconf = pile.drive(T_C, trackconf = True)
        Nconfs[i].append(Nconf)
        if (j+1) in ns:
            Nconfs_summary[i].append(Nconf)
    Nconfs[i].append(recurN(L))
    Nconfs_summary[i].append(recurN(L))
col = ["L/N"]
for nj in ns:
    col.append(f"{int(nj)}")
col.append("TOT")
table = pd.DataFrame(Nconfs_summary, columns = col)
print("\n")
print(table)

del x, ns
del pile, hstuff, Nconf, Nconfs
del zs, col, table 



#%% TEST2: BULK DRIVEN BTW HEIGHTS & AVALANCHES
print("\n\n\nTASK 1 BULK - TESTING 2: BTW Height & Avalanches\n")

T = int(5e5); t = [i+1 for i in range(T)] # Max time T, time steps

pile = Ricepile(16, zthmax = 1)
zs = np.zeros(16, dtype = int) + 1
pile.set_conf(zs, update_tc = True)
hlist, mean_height_16, std_16 = pile.drive(T, addspot = "random height")
pl.figure("BTW Bulk-Driven Height")
pl.plot(t, hlist, label = f"L = {len(pile)}")
pl.figure("BTW Bulk-Driven Avalanches")
bins = np.linspace(1, 17, 17)
pl.hist(pile.get_crit_avax(), bins = bins, label = f"L = {len(pile)}", 
        density = True, zorder = 1)
pl.hlines(1/16, 1, 33, ls = "--", color = "red", zorder = 5)
pl.yticks(list(pl.yticks()[0]) + [1/16])
label = pl.gca().get_yticklabels()[-1]
label.set_text("1/32")
label.set_color("red")


pile = Ricepile(32, zthmax = 1)
zs = np.zeros(32, dtype = int) + 1
pile.set_conf(zs, update_tc = True)
hlist, mean_height_32, std_32 = pile.drive(T, addspot = "random height")
pl.figure("BTW Bulk-Driven Height")
pl.plot(t, hlist, label = f"L = {len(pile)}")
pl.figure("BTW Bulk-Driven Avalanches")
bins = np.linspace(1, 33, 33)
pl.hist(pile.get_crit_avax(), bins = bins, label = f"L = {len(pile)}", 
        density = True, zorder = 2)
pl.hlines(1/32, 1, 33, ls = "--", color = "red", zorder = 5, label = "1/L")
pl.yticks(list(pl.yticks()[0]) + [1/32])

pl.figure("BTW Bulk-Driven Height")
pl.xlabel("time t")
pl.ylabel("height")
pl.legend()
pl.figure("BTW Bulk-Driven Avalanches")
pl.xlabel("s")
pl.ylabel("P(s;L)")
pl.legend()

print(f"The Ricepiles, after {T} grains added, present:")
print(f"-for L = 16 -> mean of height in steady state {round(mean_height_16,3)}\
, std is {std_16}")
print(f"-for L = 32 -> mean of height in steady state {round(mean_height_32,3)}\
, std is {std_32}")

del pile, hlist, mean_height_32, std_32, mean_height_16, std_16
        




