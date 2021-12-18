#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:14:03 2021

@author: RileyBallachay
"""
import pandas as pd
import re
import multiprocessing as mp
import numpy as np
#from src.preprocess import motif_generator

test = np.load("data/BinFeatures.npy")

test = pd.read_csv("Testing_data.csv")[:10000]


"""
data = list(pd.read_csv("data/DataSet.csv")["seq"])

MOTIFS = motif_generator()
motifs = [re.compile(m) for m in MOTIFS]

def fun(motifs,aa_seq):
    matches = np.zeros((len(aa_seq),len(MOTIFS)),dtype=bool)
    
    for (i,m) in enumerate(motifs):
         ret = list(filter(m.match, aa_seq))
         idxs = [aa_seq.index(s) for s in ret]
         matches[idxs,:] = 1
         print(i/len(motifs))
    
    return matches
   
def fast(aa_seq,motif=MOTIFS):
    return bool(re.match(motif,aa_seq))


def vectorize(data, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return p.map(fast, data)

        
binSet = fun(motifs,data)
    
np.save("BinFeatures.npy",binSet)


np.load("BinFeatures.npy")
#result = list(imap(fast, motifs,data))
"""
