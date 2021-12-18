#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 11:03:02 2021

@author: RileyBallachay
"""
import os
import numpy as np
import json
import pandas as pd
from src.preprocess import AMINO_ACIDS
from src.data import DataCreation
from rdkit import Chem
from src.model import HyperModel

def make_all_tetramers(AMINO_ACIDS=AMINO_ACIDS):
    AA_list = []
    for AA1 in AMINO_ACIDS:
        for AA2 in AMINO_ACIDS:
            for AA3 in AMINO_ACIDS:
                for AA4 in AMINO_ACIDS:
                    AA_list.append(AA1+AA2+AA3+AA4)
    
    return AA_list  

def make_prediction_data():
    file_name = "data/prediction_data.csv"
    if os.path.isfile(file_name):
        aa_data = pd.read_csv(file_name,index_col=0)
    else:
        AA_list = make_all_tetramers()
        AA_smiles = [Chem.MolToSmiles(Chem.MolFromFASTA(a)) for a in AA_list]
        aa_data = pd.DataFrame()
        aa_data['seq'] = AA_list
        aa_data['SMILES'] = AA_smiles
        aa_data.to_csv(file_name)
        
    return aa_data

def main():
    make_prediction_data()
    
    csv = 'data/prediction_features.csv'
    dc = DataCreation()
    dc.import_training_data(haslabels=False,csv_features=csv)   

    x_tuple = dc.get_prediction_tuples()  
    
    print([x.shape for x in x_tuple])
    hypermodel = HyperModel()
    
    predictions = hypermodel.test(x_tuple).flatten()
    
    peptides= dc.peptides
    
    with open('data/160k_predictions.json', 'w') as f:
        json.dump(dict(zip(peptides,predictions.tolist())), f)

if __name__=='__main__':
    main()    