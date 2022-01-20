#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:39:28 2021

@author: RileyBallachay
"""
from datetime import datetime

import keras
import keras_tuner as kt
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, concatenate
from keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution

from src.data.config import MODEL_CONFIG
from src.plotting import plot_predictions

from .vae import VAE

disable_eager_execution()


class HyperModel:
    """
    Create hypermodel automatic tuner as described in keras documentation
    https://www.tensorflow.org/tutorials/keras/keras_tuner in order to
    automatically tune keras parameters without intervention.
    """

    def create_model(self, x, y):
        print("The program is creating a model\n")
        self._create_hypertuning(x, y)

        now = datetime.now()
        self.time = now.strftime("%d_%m_%Y_%H_%M")
        self.modelpath = "data/model.h5"

    def train(self, x, y):

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs

        model = self.tuner.hypermodel.build(self.best_hps)

        history = model.fit(x, y, epochs=25, validation_split=0.33, batch_size=32)

        val_acc_per_epoch = history.history["val_loss"]
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
        print("Best epoch: %d" % (best_epoch,))

        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # Retrain the model
        self.hypermodel.fit(x, y, epochs=20, validation_split=0.05)

        self.hypermodel.save(self.modelpath)

    def test(self, x, y=None):
        self.modelpath = "data/model.h5"
        if hasattr(self, "hypermodel"):
            pass
        else:
            try:
                self.hypermodel = keras.models.load_model(self.modelpath)
            except:
                raise Exception("NO MODEL PATH EXISTS, ABORTING...")

        y_hat = self.hypermodel.predict(x)

        if y is not None:
            plot_predictions(y, y_hat, self.time)

        return y_hat

    def _create_hypertuning(self, x, y):
        print("The program is creating hypertuning\n")
        self.tuner = kt.Hyperband(
            _model_builder,
            objective="val_loss",
            max_epochs=2,
            factor=3,
            directory="data",
            project_name="hyperparamter_metadata",
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

        self.tuner.search(
            x,
            y,
            epochs=10,
            validation_split=0.15,
            callbacks=[stop_early],
            batch_size=32,
        )

        # Get the optimal hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        self.best_hps = best_hps


def _model_builder(hp):

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512

    vae = VAE()

    vae.create_minimodel(**MODEL_CONFIG)
    model1 = vae.encoder

    # First section of features
    input2 = Input(shape=(104,))
    hp_units = hp.Int("units layer 1", min_value=64, max_value=128, step=64)
    model2 = Dense(hp_units, activation="sigmoid")(input2)
    model2 = Dropout(0.3)(model2)
    model2 = Model(inputs=input2, outputs=model2)
    dropout1 = hp.Float("dropout 1", min_value=0.0, max_value=0.9, step=0.3)

    # Second section of features
    input3 = Input(shape=(2386,))
    model3 = Dense(256, activation="sigmoid")(input3)
    model3 = Dropout(dropout1)(model3)
    model3 = Model(inputs=input3, outputs=model3)
    combined = concatenate([model1.output, model2.output, model3.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(50, activation="linear")(combined)
    z = Dense(10, activation="linear")(z)
    dropout2 = hp.Float("dropout 2", min_value=0.0, max_value=0.9, step=0.3)
    z = Dropout(dropout2)(z)
    z = Dense(1, activation="linear")(z)

    model = Model(inputs=[model1.input, model2.input, model3.input], outputs=z)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="mean_squared_error",
    )

    return model
