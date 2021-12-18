#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:54:06 2021

@author: RileyBallachay
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import pandas as pd

# Opening JSON file
f = open('data/160k_predictions.json',)
 
# returns JSON object as
# a dictionary
data_pred = json.load(f)

keys = pd.DataFrame.from_dict(data_pred, orient='index').reset_index()

data_true = pd.read_csv('data/DataSet.csv')
temp = [data_pred[d] for d in data_true["seq"]]
data_true['pred'] = temp

plt.figure(dpi=200)

y = data_true['log.label']
y_hat = data_true['pred']
sns.jointplot(y, y_hat, kind="reg")