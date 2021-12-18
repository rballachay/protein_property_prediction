#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:04:44 2021

@author: RileyBallachay
"""
import multiprocessing as mp
import re
import glob
import pandas as pd
import os
import numpy as np
from openbabel import pybel
from os.path import dirname, abspath
from data.AADesc import Z3, Z5, VHSE, BLOSUM62


AMINO_ACIDS = ['A','R', 'N', 'D', 'C', 'E', 'Q', 
               'G', 'H','I', 'L', 'K', 'M', 'F',
               'P', 'S', 'T', 'W', 'Y', 'V']


PATTERNS = {
    "TRIMERS":
    ["{0}{0}{0}{1}",
    "{0}{0}{1}{0}",
    "{0}{1}{0}{0}",
    "{1}{0}{0}{0}"]
    ,
    "DIMERS":
    ["{0}{0}{1}{2}",
    "{0}{1}{2}{0}",
    "{1}{2}{0}{0}",
    "{1}{2}{0}{0}",
    "{1}{2}{0}{0}",
    "{0}{1}{0}{2}"]
    ,
    "MONOMERS":
    ["{1}{2}{3}{0}",
    "{1}{2}{0}{3}",
    "{1}{0}{2}{3}",
    "{0}{1}{2}{3}"],
    }

def generate_cartesian_products(AMINO_ACIDS):
    trimers = [] 
    dimers = []
    monomers = []  
    
    for AA1 in AMINO_ACIDS:
        monomers.append(AA1)
        for AA2 in AMINO_ACIDS:
            dimers.append(AA1+AA2)
            for AA3 in AMINO_ACIDS:
                trimers.append(AA1+AA2+AA3)
                
    return monomers,dimers,trimers
        

def motif_generator(amino_acids=AMINO_ACIDS,patterns=PATTERNS,X="[A-Z]"):
    
    monomers, dimers, trimers = generate_cartesian_products(AMINO_ACIDS)
    
    all_sequences = []
    
    for P in patterns["TRIMERS"]:
        for M in monomers:
            A = M
            seq = P.format(X,A)
            all_sequences.append(seq)
    
    for P in patterns["DIMERS"]:
        for D in dimers:
            A,B = D
            seq = P.format(X,A,B)
            all_sequences.append(seq)            
            
    for P in patterns["MONOMERS"]:
        for D in trimers:
            A,B,C = D
            seq = P.format(X,A,B,C)
            all_sequences.append(seq)                   
                
    return all_sequences

def get_motif_titles():
    motifs = motif_generator(X="X")    
    return motifs
            
def create_motif_dataset(haslabels,file_path):
    
    if haslabels:
        data = list(pd.read_csv("data/DataSet.csv")["seq"])
    else:
        data = list(pd.read_csv("data/prediction_data.csv")["seq"])

    MOTIFS = motif_generator()
    motifs = [re.compile(m) for m in MOTIFS]
    
    def fun(motifs,aa_seq):
        matches = np.zeros((len(aa_seq),len(MOTIFS)),dtype=bool)
        
        for (i,m) in enumerate(motifs):
             ret = list(filter(m.match, aa_seq))
             idxs = [aa_seq.index(s) for s in ret]
             matches[idxs,i] = 1
             print(i/len(motifs))
        
        return matches
            
    binSet = fun(motifs,data)
        
    np.save("data/BinFeatures.npy",binSet)


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


"""
All functions in section taken AND ADADPTED from :
    https://github.com/pnnl/darkchem 
"""

# Globals
SMI = ['PAD',
       'N','C','H','O','S','c','n','@','[',']','(',')','1','2','=']

def _encode(string, charset):
    '''
    Encodes string with a given charset.
    Returns None if s contains illegal characters
    If s is empty, returns an empty array
    '''

    if pd.isna(string):
        return np.array([])

    vec = np.zeros(len(string))
    for i in range(len(string)):
        s = string[i]
        if s in charset:
            vec[i] = charset.index(s)
        else:  # Illegal character in s
            print("ILLEGAL CHARACTER:")
            print(s)
            return None

    return vec

def _add_padding(l, length):
    '''
    Adds padding to l to make it size length.
    '''

    ltemp = list(l)
    ltemp.extend([0] * (length - len(ltemp)))
    return ltemp


def _smi2vec(smi, charset, max_length=117):
    # Encode SMILES
    vec = _encode(smi, charset)

    # Check for errors
    if vec is None:
        # print('%s skipped, contains illegal characters' % smi)
        return None
    if len(vec) > max_length:
        print('%s skipped, too long' % smi)
        print('length = %i' % len(smi))
        return None

    # Add padding
    vec = _add_padding(vec, max_length)

    # Return encoded InChI
    return vec


def struct2vec(struct, charset=SMI, max_length=117):
    '''
    Takes in structure and returns the encoded version using the default
    or passed in charset.

    Parameters
    ----------
    struct : str
        Structure of compound, represented as an InChI or SMILES string.

    charset : list, optional
        Character set used for encoding.

    max_length : int, optional
        Maximum length of encoding.

    Returns
    -------
    vec : unit8 array
        Encoded structure
    '''

    output = _smi2vec(struct, charset, max_length)

    if output is None:
        return np.zeros(max_length)
    else:
        return np.array(output, dtype=np.uint8)

def vectorize(smiles, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return p.map(struct2vec, smiles)


def _canonicalize(smi):
    '''Canonicalizes SMILES string.'''
    return pybel.readstring('smi', smi).write('can').strip()


def canonicalize(smiles, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return p.map(_canonicalize, smiles)


def _parse_formula(formula, targets='CHNOPS'):
    atoms = re.findall(r'([A-Z][a-z]?)(\d+)?', formula)

    d = {k: v for k, v in atoms if k in targets}

    for k in targets:
        if k in d.keys():
            if d[k] == '':
                d[k] = 1
            else:
                d[k] = int(d[k])
        else:
            d[k] = 0
    return d


def parse_formulas(formulas, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return pd.DataFrame(data=p.map(_parse_formula, formulas))

def check_for_encoded_smiles(filename):
    filename += '.npy'
    parent_dir = dirname(dirname(abspath(__file__)))
    file_path = parent_dir+'/data/'+filename
    file_path = file_path
    if os.path.isfile(file_path):
        embedding = np.load(file_path)
        return (True,embedding)
    else:
        return (False,None)

def generate_encoded_smiles(df, name='smiles_strings',output='', canonical=False, shuffle=True,haslabels=True):
    '''
    Assumes dataframe with InChI or SMILES columns and
    optionally a Formula column.  Any additional columns will
    be propagated as labels for prediction.
    '''
    if haslabels:
        check = check_for_encoded_smiles(name)
    else:
        name+="_prediction"
        print(name)
        check = check_for_encoded_smiles(name)
    if check[0]:
        arr = check[1]
    else:
        # already converted
        if 'SMILES' in df.columns and canonical is True:
            pass
        elif 'SMILES' in df.columns:
            df['SMILES'] = canonicalize(df['SMILES'].values)
            print(df['SMILES'])
            df.to_csv(os.path.join(output, 'data/%s_canonical.tsv' % name), index=False, sep='\t')
        # error
        else:
            raise KeyError('Dataframe must have an "InChI" or "SMILES" column.')
    
        # vectorize
        # TODO: configurable max length
        # TODO: configurable charsest
        vectors = np.vstack(vectorize(df['SMILES'].values))
        #vectors = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, vectors)
    
        df['vec'] = vectors.tolist()
    
        #df.dropna(how='any', axis=0, inplace=True)
        arr = np.vstack(df['vec'].values)
        
        arr=np.nan_to_num(arr)
    
        # labels
        if 'InChI' in df.columns:
            labels = df.drop(columns=['InChI', 'SMILES', 'vec'])
        else:
            labels = df.drop(columns=['SMILES', 'vec'])
    
        # save
        np.save(os.path.join(output, 'data/%s.npy' % name), arr)
    
        if len(labels.columns) > 0:
            np.save(os.path.join(output, 'data/%s_labels.npy' % name), labels.values)
        
    return arr




