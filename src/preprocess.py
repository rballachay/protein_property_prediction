#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:04:44 2021

@author: RileyBallachay
"""
import os
import re

import numpy as np
import pandas as pd
from data.AADesc import BLOSUM62, VHSE, Z3, Z5

from .fixtures import AMINO_ACIDS, PATTERNS


def get_motif_titles():
    motifs = _motif_generator(X="X")
    return motifs


def create_motif_dataset(haslabels, file_path):

    if haslabels:
        data = list(pd.read_csv("data/DataSet.csv")["seq"])
    else:
        data = list(pd.read_csv("data/prediction_data.csv")["seq"])

    MOTIFS = _motif_generator()
    motifs = [re.compile(m) for m in MOTIFS]

    def fun(motifs, aa_seq):
        matches = np.zeros((len(aa_seq), len(MOTIFS)), dtype=bool)

        for (i, m) in enumerate(motifs):
            ret = list(filter(m.match, aa_seq))
            idxs = [aa_seq.index(s) for s in ret]
            matches[idxs, i] = 1
            print(i / len(motifs))

        return matches

    binSet = fun(motifs, data)
    np.save("data/BinFeatures.npy", binSet)


def read_sequences(data, csv_name="data/sequence_labels.csv"):
    sequences = list(data["seq"])
    datasets = [Z3, Z5, VHSE, BLOSUM62]
    total_values = [list(item.values())[0] for item in datasets]
    e_len = [len(i) for i in total_values]
    t_len = len([i for L in total_values for i in L]) * 4
    feature_array = np.zeros((len(sequences), t_len))

    for (j, seq) in enumerate(sequences):
        index_stop = 0
        for (i, (dset, stride)) in enumerate(zip(datasets, e_len)):
            for (k, pep) in enumerate(seq):
                index_start = index_stop
                index_stop = index_start + stride
                feature_array[j, index_start:index_stop] = datasets[i][pep]

    features_pandas = pd.DataFrame(feature_array, columns=_get_feature_titles(e_len))

    features_pandas.to_csv(os.path.join("", csv_name))
    return features_pandas


def _get_feature_titles(e_len):
    datasets = ["Z3", "Z5", "VHSE", "BLOSUM62"]
    f_titles = []

    for d_set, n in zip(datasets, e_len):
        for m in range(4):
            for i in range(n):
                f_titles.append(f"{d_set}.{str(m + 1)}.{str(i + 1)}")

    return f_titles


def _motif_generator(amino_acids=AMINO_ACIDS, patterns=PATTERNS, X="[A-Z]"):

    monomers, dimers, trimers = _generate_cartesian_products(AMINO_ACIDS)
    all_sequences = []

    for P in patterns["TRIMERS"]:
        for M in monomers:
            A = M
            seq = P.format(X, A)
            all_sequences.append(seq)

    for P in patterns["DIMERS"]:
        for D in dimers:
            A, B = D
            seq = P.format(X, A, B)
            all_sequences.append(seq)

    for P in patterns["MONOMERS"]:
        for D in trimers:
            A, B, C = D
            seq = P.format(X, A, B, C)
            all_sequences.append(seq)

    return all_sequences


def _generate_cartesian_products(AMINO_ACIDS):
    trimers = []
    dimers = []
    monomers = []

    for AA1 in AMINO_ACIDS:
        monomers.append(AA1)
        for AA2 in AMINO_ACIDS:
            dimers.append(AA1 + AA2)
            for AA3 in AMINO_ACIDS:
                trimers.append(AA1 + AA2 + AA3)

    return monomers, dimers, trimers
