#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 11:03:02 2021

@author: RileyBallachay
"""
import json
from src.data_creation import DataCreation
from src.hypermodel import HyperModel
from src.prediction import make_prediction_data


def main():

    make_prediction_data()

    dc = DataCreation(haslabels=False, csv_features="data/prediction_features.csv")

    predictions = HyperModel().test(dc.get_prediction_tuples())

    with open("data/160k_predictions.json", "w") as f:
        json.dump(dict(zip(dc.peptides, predictions.flatten().tolist())), f)


if __name__ == "__main__":
    main()
