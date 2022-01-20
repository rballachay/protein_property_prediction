#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:50:43 2021

@author: RileyBallachay
"""
import seaborn as sns
import os


def plot_predictions(y, y_hat, time):
    y_hat = y_hat.flatten()
    plot = sns.jointplot(y, y_hat, kind="reg")
    plot.savefig(filesystem(time))
    return


def filesystem(time):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = dir_path + "/data/plots/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory + time + ".png"
