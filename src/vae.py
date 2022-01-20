#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:39:28 2021

@author: RileyBallachay
"""
import os
from os.path import abspath, dirname

import keras
import numpy as np
from keras import backend as K
from keras import objectives
from keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    Input,
    Lambda,
    Reshape,
)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class VAE(object):
    def create_minimodel(
        self,
        nchars,
        max_length,
        kernels,
        filters,
        embedding_dim,
        latent_dim,
        epsilon_std,
        nlabels,
        dropout,
        freeze_vae,
        **kwargs
    ):
        # initialize variables
        self.nchars = nchars
        self.max_length = max_length
        self.kernels = kernels
        self.filters = filters
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.nlabels = nlabels
        self.dropout = dropout
        self.freeze_vae = freeze_vae

        # define input, embed
        x = Input(shape=(self.max_length,))
        self.x_vae = x

        x_embed = Embedding(
            self.nchars, self.embedding_dim, input_length=self.max_length
        )(x)

        # expose embedding
        self.embedder = Model(inputs=x, outputs=x_embed, name="embedder")

        # construct encoder (input->latent)
        vae_loss, encoded, encoded_variational = self._build_encoder(x_embed)
        self.encoder = Model(inputs=x, outputs=encoded, name="encoder")

        # variational encoder (input->latent+noise)
        self.encoder_variational = Model(
            inputs=x, outputs=encoded_variational, name="v_encoder"
        )

        self.vae_loss = vae_loss

    def create_multitask(
        self,
        nchars,
        max_length,
        kernels,
        filters,
        embedding_dim,
        latent_dim,
        epsilon_std,
        nlabels,
        dropout,
        freeze_vae,
        **kwargs
    ):
        # initialize variables
        self.nchars = nchars
        self.max_length = max_length
        self.kernels = kernels
        self.filters = filters
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.nlabels = nlabels
        self.dropout = dropout
        self.freeze_vae = freeze_vae

        # define input, embed
        x = Input(shape=(self.max_length,))
        x_embed = Embedding(
            self.nchars, self.embedding_dim, input_length=self.max_length
        )(x)

        # expose embedding
        self.embedder = Model(inputs=x, outputs=x_embed, name="embedder")

        # construct encoder (input->latent)
        vae_loss, encoded, encoded_variational = self._build_encoder(x_embed)
        self.encoder = Model(inputs=x, outputs=encoded, name="encoder")

        # variational encoder (input->latent+noise)
        self.encoder_variational = Model(
            inputs=x, outputs=encoded_variational, name="v_encoder"
        )

        # define latent input
        encoded_input = Input(shape=(self.latent_dim,))

        # property predictor
        self.predictor = Model(
            inputs=encoded_input,
            outputs=self._build_predictor(encoded_input),
            name="property_predictor",
        )

        # construct decoder (latent->output)
        self.decoder = Model(
            inputs=encoded_input,
            outputs=self._build_decoder(encoded_input),
            name="decoder",
        )

        # construct variational autoencoder (input->latent+noise->output)
        self.autoencoder = Model(
            inputs=x,
            outputs=[
                self.decoder(self.encoder_variational(x)),
                self.predictor(self.encoder_variational(x)),
            ],
            name="vae",
        )

        # optimizer
        opt = Adam(
            lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=True
        )

        # optionally freeze weights
        if self.freeze_vae is True:
            self.embedder.trainable = False
            self.encoder.trainable = False
            self.encoder_variational.trainable = False
            self.decoder.trainable = False

        # compile autoencoder
        self.autoencoder.compile(
            optimizer=opt,
            loss=[vae_loss, "mean_squared_error"],
            metrics=["accuracy", objectives.categorical_crossentropy],
            loss_weights=[0.1, 0.9],
        )

    def check_for_embedding(self, filename):
        parent_dir = dirname(dirname(abspath(__file__)))
        file_path = parent_dir + filename
        self.file_path = file_path
        if os.path.isfile(file_path):
            self.embedding = np.load(file_path)
            return True
        else:
            return False

    def yield_latent_representation(self, x, y2, filename="/data/embedded_smiles.npy"):
        if self.check_for_embedding(filename):
            x_hat = self.embedding
        else:
            y1 = keras.utils.to_categorical(x, 117)
            y1 = y1.reshape((-1, 117, 117))
            y = [y1, y2]
            self.autoencoder.fit(x, y, epochs=5, shuffle=False, verbose=1)
            x_hat = self.encoder_variational.predict(x)
            np.save(self.file_path, x_hat)

        return x_hat

    def _get_learning_rate(self):
        opt = self.autoencoder.optimizer
        lr0 = opt.lr
        if opt.initial_decay > 0:
            lr = lr0 * (
                1.0 / (1.0 + opt.decay * K.cast(opt.iterations, K.dtype(opt.decay)))
            )

        return lr, lr0

    def _build_encoder(self, x):
        # build filters
        for i, (f, k) in enumerate(zip(self.filters, self.kernels)):
            if i < 1:
                h = Conv1D(f, k, activation="relu", padding="same")(x)
            else:
                h = Conv1D(f, k, activation="relu", padding="same")(h)

        h = Flatten()(h)

        # latent space sampling
        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(
                shape=(batch_size, self.latent_dim), mean=0.0, stddev=self.epsilon_std
            )
            return z_mean_ + K.exp(z_log_var_) * epsilon

        # latent dim
        z_mean = Dense(self.latent_dim, activation="tanh")(h)
        z_log_var = Dense(self.latent_dim, activation="linear")(h)

        # custom loss term
        def vae_loss(y_true, y_pred):
            xent_loss = K.mean(
                objectives.categorical_crossentropy(y_true, y_pred), axis=-1
            )
            kl_loss = K.mean(
                -z_log_var + 0.5 * K.square(z_mean) + K.exp(z_log_var) - 1, axis=-1
            )

            return xent_loss + kl_loss  # + mu_loss + var_loss

        # add noise
        z_mean_variational = Lambda(sampling, output_shape=(self.latent_dim,))(
            [z_mean, z_log_var]
        )

        return (vae_loss, z_mean, z_mean_variational)

    def _build_decoder(self, encoded):
        # connect to latent dim
        h = Reshape((self.latent_dim, 1))(encoded)

        # build filters
        for f, k in zip(self.filters, self.kernels):
            h = Conv1D(f, k, activation="relu", padding="same")(h)

        # prepare output dim
        h = Flatten()(h)
        h = Dense(self.max_length * self.nchars)(h)
        h = Reshape((self.max_length, self.nchars))(h)
        decoded = Activation("softmax")(h)

        return decoded

    def _build_predictor(self, encoded):
        return Dense(50, activation="linear")(encoded)
