#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 19:48:40 2021

@author: RileyBallachay
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from data.AADesc import Z3, Z5, VHSE, BLOSUM62
from src.preprocess import generate_encoded_smiles, create_motif_dataset, get_motif_titles
from src.model import VAE
from src.config import MODEL_CONFIG

def __get_feature_titles__(e_len):
   
    datasets = ['Z3', 'Z5', 'VHSE', 'BLOSUM62']
    
    f_titles = []
    
    for d_set,n in zip(datasets,e_len):
        
        for m in range(4):
            
            for i in range(n):
               
                title_temp = d_set + '.' + str(m+1) + '.' + str(i+1)
                
                f_titles.append(title_temp)
    
    return f_titles
    
    
def read_sequences(data,csv_name='data/sequence_labels.csv'):
    sequences = list(data['seq'])
    
    datasets = [Z3, Z5, VHSE, BLOSUM62]
    
    total_values =  [list(item.values())[0] for item in datasets]
    
    e_len = [len(i) for i in  total_values]
    
    t_len = len([i for L in  total_values for i in L]) * 4
    
    feature_array = np.zeros((len(sequences),t_len))
    
    for (j,seq) in enumerate(sequences):
        
        index_stop = 0
        
        for (i,(dset,stride)) in enumerate(zip(datasets,e_len)):
        
            for (k,pep) in enumerate(seq):

                index_start = index_stop
                
                index_stop = index_start + stride
                
                feature_array[j, index_start:index_stop] = datasets[i][pep]
       
    f_titles =  __get_feature_titles__(e_len)
    
    features_pandas = pd.DataFrame(feature_array,columns=f_titles)

    features_pandas.to_csv(os.path.join('', csv_name))
    return features_pandas


def __preprocess_features__(df_to_scale):
    # They appear to already be standard scaled, just to be safe
    
    cols = df_to_scale.columns.tolist()
   
    np_df = df_to_scale[cols]
    
    scaler = StandardScaler()
    
    scaled_np = scaler.fit_transform(np_df)
    
    df_scaled = pd.DataFrame(scaled_np,columns=cols)
    
    return df_scaled
    
