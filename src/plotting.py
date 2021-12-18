#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:50:43 2021

@author: RileyBallachay
"""
import seaborn as sns
from scipy import stats
import os 

def plot_predictions(y,y_hat,time):

    def r2(x, y): return stats.pearsonr(x, y)[0] ** 2
    
    y_hat = y_hat.flatten()
    plot = sns.jointplot(y, y_hat, kind="reg")
    
    path = filesystem(time)
    
    plot.savefig(path) 
    
    return

def filesystem(time):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = dir_path + '/data/plots/'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory + time + '.png'
    
    