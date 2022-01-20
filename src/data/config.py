#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:41:54 2021

@author: RileyBallachay
"""

MODEL_CONFIG = {
    "kernels": [3, 5, 7],
    "filters": [3, 5, 7],
    "embedding_dim": 64,
    "latent_dim": 64,
    "epsilon": 0.8,
    "epsilon_std": 0.1,
    "dropout": 0.3,
    "freeze_vae": False,
    "validation": 0.3,
    "batch_size": 128,
    "epochs": 10000,
    "patience": 5000,
    "seed": 777,
    "nlabels": 2,
    "max_length": 117,
    "nchars": 16,
    "data": "/data/",
}
