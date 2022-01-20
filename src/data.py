#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:01:46 2021

@author: RileyBallachay
"""
import os
import pickle

import numpy as np
import pandas as pd

from src.darkchem import generate_encoded_smiles
from src.preprocess import create_motif_dataset, get_motif_titles, read_sequences


class DataCreation:
    def __init__(self, haslabels=True, csv_features="data/sequence_labels.csv"):
        self.haslabels = haslabels

        if self.haslabels:
            data_raw = pd.read_csv("data/DataSet.csv")
            data_raw = data_raw.rename(columns={"SMILE": "SMILES"})
            self.labels = data_raw["log.label"]
        else:
            data_raw = pd.read_csv("data/prediction_data.csv", index_col=0)

        self.peptides = data_raw.seq

        full_features = read_sequences(data_raw, csv_features)
        smiles_encoded_df = self._smiles_df(
            generate_encoded_smiles(
                pd.DataFrame(data_raw["SMILES"]), haslabels=haslabels
            )
        )
        motifs_data = self._get_motif_dataset()

        self.data = pd.concat([smiles_encoded_df, full_features, motifs_data], axis=1)

        if self.haslabels:
            self.data["label"] = self.labels
            self._drop_outliers()

    def get_data_tuples(self, train_frac=0.95):

        xdata_copy = self._even_out_distribution(self.data.copy())

        y_data = np.array(xdata_copy.label)
        xdata_copy = xdata_copy.drop("label", axis=1).to_numpy()

        idx = int(len(xdata_copy) * train_frac)

        y_train = y_data[:idx]
        y_test = y_data[idx:]

        x1_data = xdata_copy[:, :117]
        x2_data = xdata_copy[:, 117 : (117 + 104)]
        x3_data = xdata_copy[:, (117 + 104) :]

        x_train = (x1_data[:idx], x2_data[:idx], x3_data[:idx])

        x_test = (x1_data[idx:], x2_data[idx:], x3_data[idx:])

        return x_train, y_train, x_test, y_test

    def get_prediction_tuples(self):

        xdata_copy = self.data.to_numpy()

        x1_data = xdata_copy[:, :117]
        x2_data = xdata_copy[:, 117 : (117 + 104)]
        x3_data = xdata_copy[:, (117 + 104) :]

        return (x1_data, x2_data, x3_data)

    def _even_out_distribution(self, xdata_copy):

        xdata_copy = xdata_copy.sort_values(by="label")
        len_df = len(xdata_copy.index)
        top_ind = int(0.1 * len_df)
        bottom_ind = int(0.9 * len_df)
        take_every = 4

        xdata_out = xdata_copy.iloc[:top_ind]
        to_concat1 = xdata_copy[top_ind::take_every].iloc[
            : int((1 / take_every) * 0.8 * len_df)
        ]
        to_concat2 = xdata_copy.iloc[bottom_ind:]
        xdata_out = pd.concat(
            [xdata_out, to_concat1, to_concat2], ignore_index=True
        ).reset_index(drop=True)

        xdata_out = xdata_out.sample(frac=1)
        return xdata_out

    def _drop_outliers(self):

        for outlier in (-4.439897, -4.263806):
            self.data = self.data[self.data["label"] != outlier]

        self.data = self.data.reset_index(drop=True)
        self.data.label = np.array(self.data["label"]).reshape(-1, 1)

    def _smiles_df(self, smiles_encoded):

        smi_titles = []
        for i in range(117):
            title = "smiles." + str(i + 1)
            smi_titles.append(title)

        smiles_dat = pd.DataFrame(smiles_encoded, columns=smi_titles)

        return smiles_dat

    def _get_motif_dataset(self):

        if self.haslabels:
            file_path = "data/BinFeatures.npy"
        else:
            file_path = "data/BinFeatures.npy"

        if os.path.isfile(file_path):
            motifs = np.load(file_path)
        else:
            motifs = create_motif_dataset(self.haslabels, file_path)

        titles = get_motif_titles()

        return self._check_for_motif_csv(motifs, titles)

    def _check_for_motif_csv(self, motifs, titles):

        if self.haslabels:
            file_path = "data/BinFeaturesTrimmed.npy"
            txt_path = "data/BinFeaturesTitles.txt"

            if os.path.isfile(file_path):
                motifs = np.load(file_path)
                with open(txt_path, "rb") as fp:
                    binTitles = pickle.load(fp)
                motifs = pd.DataFrame(motifs, columns=binTitles, dtype=bool)
            else:
                motifs = self._preprocess_motifs(motifs, titles)
                motifs_data = motifs.to_numpy()
                np.save(file_path, motifs_data)
                titles = motifs.columns.tolist()
                with open(txt_path, "wb") as fp:
                    pickle.dump(titles, fp)

        else:
            file_path = "data/BinFeaturesPredictionTrimmed.npy"
            txt_path = "data/BinFeaturesTitles.txt"

            if os.path.isfile(file_path):
                motifs = np.load(file_path)
                with open(txt_path, "rb") as fp:
                    binTitles = pickle.load(fp)
                motifs = pd.DataFrame(motifs, columns=binTitles, dtype=bool)
            else:
                titles = ["motif." + str(i) for i in titles]
                motifs = pd.DataFrame(motifs, columns=titles)
                with open(txt_path, "rb") as fp:
                    titles_new = set(pickle.load(fp))
                motifs = motifs[titles_new]

        return motifs

    def _preprocess_motifs(self, motifs, titles):

        bools = [np.sum(motifs, axis=0) > 20][0]
        bools_mask = np.where(bools)

        motifs = motifs[:, bools_mask].reshape((len(motifs[:, 0]), sum(bools)))
        titles = ["motif." + str(i) for (i, b) in zip(titles, bools) if b]

        return pd.DataFrame(motifs, columns=titles)
