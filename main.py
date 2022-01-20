#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 19:48:11 2021

@author: RileyBallachays
"""

from src.data.data_creation import DataCreation
from src.model.hypermodel import HyperModel


def main():

    data = DataCreation()

    x_train, y_train, x_test, y_test = data.get_data_tuples(train_frac=0.95)

    hypermodel = HyperModel()

    hypermodel.create_model(x_train, y_train)

    hypermodel.train(x_train, y_train)

    hypermodel.test(x_test, y_test)


if __name__ == "__main__":
    main()
