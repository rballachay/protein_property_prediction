#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 08:39:28 2021

@author: RileyBallachay
"""
import numpy as np
import os
from datetime import datetime
import keras
import tensorflow as tf
import keras_tuner as kt
from keras.layers import Input, Conv1D, MaxPooling1D,Embedding, Flatten, Dense, Reshape, Lambda, Activation, Dropout, concatenate
from keras.models import Model
from keras import objectives, backend as K
from keras.optimizers import Adam
from os.path import dirname, abspath
from tensorflow.python.framework.ops import disable_eager_execution
from src.config import MODEL_CONFIG
from src.plotting import plot_predictions

disable_eager_execution()

class VAE(object):
    
    """
    Class VAE is taken and adapted from: 
        https://github.com/pnnl/darkchem 
    """
    def create_minimodel(self, nchars, max_length, kernels, filters,
                         embedding_dim, latent_dim, epsilon_std, nlabels, dropout, freeze_vae, **kwargs):
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
        
        x_embed = Embedding(self.nchars, self.embedding_dim, input_length=self.max_length)(x)

        # expose embedding
        self.embedder = Model(inputs=x,
                              outputs=x_embed,
                              name='embedder')

        # construct encoder (input->latent)
        vae_loss, encoded, encoded_variational = self._build_encoder(x_embed)
        self.encoder = Model(inputs=x,
                             outputs=encoded,
                             name='encoder')

        # variational encoder (input->latent+noise)
        self.encoder_variational = Model(inputs=x,
                                         outputs=encoded_variational,
                                         name='v_encoder')
        
        self.vae_loss = vae_loss
    
    def create_multitask(self, nchars, max_length, kernels, filters,
                         embedding_dim, latent_dim, epsilon_std, nlabels, dropout, freeze_vae, **kwargs):
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
        x_embed = Embedding(self.nchars, self.embedding_dim, input_length=self.max_length)(x)

        # expose embedding
        self.embedder = Model(inputs=x,
                              outputs=x_embed,
                              name='embedder')

        # construct encoder (input->latent)
        vae_loss, encoded, encoded_variational = self._build_encoder(x_embed)
        self.encoder = Model(inputs=x,
                             outputs=encoded,
                             name='encoder')

        # variational encoder (input->latent+noise)
        self.encoder_variational = Model(inputs=x,
                                         outputs=encoded_variational,
                                         name='v_encoder')

        # define latent input
        encoded_input = Input(shape=(self.latent_dim,))

        # property predictor
        self.predictor = Model(inputs=encoded_input,
                               outputs=self._build_predictor(encoded_input),
                               name='property_predictor')

        # construct decoder (latent->output)
        self.decoder = Model(inputs=encoded_input,
                             outputs=self._build_decoder(encoded_input),
                             name='decoder')

        # construct variational autoencoder (input->latent+noise->output)
        self.autoencoder = Model(inputs=x,
                                 outputs=[self.decoder(self.encoder_variational(x)),
                                          self.predictor(self.encoder_variational(x))],
                                 name='vae')

        # optimizer
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1E-8, amsgrad=True)

        # optionally freeze weights
        if self.freeze_vae is True:
            self.embedder.trainable = False
            self.encoder.trainable = False
            self.encoder_variational.trainable = False
            self.decoder.trainable = False

        # compile autoencoder
        self.autoencoder.compile(optimizer=opt,
                                 loss=[vae_loss, 'mean_squared_error'],
                                 metrics=['accuracy', objectives.categorical_crossentropy],
                                 loss_weights=[0.1, 0.9])

    def check_for_embedding(self,filename):
        parent_dir = dirname(dirname(abspath(__file__)))
        file_path = parent_dir+filename
        self.file_path = file_path
        if os.path.isfile(file_path):
            self.embedding = np.load(file_path)
            return True
        else:
            return False
        
        
    def yield_latent_representation(self,x,y2,filename='/data/embedded_smiles.npy'):
        if self.check_for_embedding(filename):
            x_hat = self.embedding  
        else:   
            y1 = keras.utils.to_categorical(x, 117)
            y1 = y1.reshape((-1, 117, 117))
            y=[y1,y2]
            self.autoencoder.fit(x,y,epochs=5,shuffle=False,verbose=1)
            x_hat = self.encoder_variational.predict(x)
            np.save(self.file_path,x_hat)
    
        return x_hat
        
    def _get_learning_rate(self):
        opt = self.autoencoder.optimizer
        lr0 = opt.lr
        if opt.initial_decay > 0:
            lr = lr0 * (1. / (1. + opt.decay * K.cast(opt.iterations,
                                                      K.dtype(opt.decay))))

        return lr, lr0

    def _build_encoder(self, x):
        # build filters
        for i, (f, k) in enumerate(zip(self.filters, self.kernels)):
            if i < 1:
                h = Conv1D(f, k, activation='relu', padding='same')(x)
            else:
                h = Conv1D(f, k, activation='relu', padding='same')(h)
        
        h = Flatten()(h)

        # latent space sampling
        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=self.epsilon_std)
            return z_mean_ + K.exp(z_log_var_) * epsilon

        # latent dim
        z_mean = Dense(self.latent_dim, activation='tanh')(h)
        z_log_var = Dense(self.latent_dim, activation='linear')(h)

        # custom loss term
        def vae_loss(y_true, y_pred):
            xent_loss = K.mean(objectives.categorical_crossentropy(y_true, y_pred), axis=-1)
            kl_loss = K.mean(-z_log_var + 0.5 * K.square(z_mean) + K.exp(z_log_var) - 1, axis=-1)

            return xent_loss + kl_loss  # + mu_loss + var_loss

        # add noise
        z_mean_variational = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        return (vae_loss, z_mean, z_mean_variational)

    def _build_decoder(self, encoded):
        # connect to latent dim
        h = Reshape((self.latent_dim, 1))(encoded)

        # build filters
        for f, k in zip(self.filters, self.kernels):
            h = Conv1D(f, k, activation='relu', padding='same')(h)

        # prepare output dim
        h = Flatten()(h)
        h = Dense(self.max_length * self.nchars)(h)
        h = Reshape((self.max_length, self.nchars))(h)
        decoded = Activation('softmax')(h)

        return decoded

    def _build_predictor(self, encoded):
        return Dense(50, activation='linear')(encoded)

def model_builder(hp):
         
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        
        vae = VAE()
        
        vae.create_minimodel(**MODEL_CONFIG)
        
        model1 = vae.encoder
        
        # Create Model #2 for combined metamodel
        
        input2 = Input(shape=(104))
        
        hp_units = hp.Int('units layer 1', min_value=64, max_value=128, step=64)
        
        model2 = Dense(hp_units, activation="sigmoid")(input2)
        
        model2 = Dropout(0.3)(model2)
        
        model2 = Model(inputs=input2, outputs=model2)
        
        #dropout1 = hp.Float('dropout 1', min_value=0.0, max_value=0.9, step=.3)
        
        #model2 = Dense(64)(model2)
        input3 = Input(shape=(2386,))
        
        model3 = Dense(256, activation="sigmoid")(input3)
        
        model3 = Dropout(0.3)(model3)
        
        model3 = Model(inputs=input3, outputs=model3)
        

        combined = concatenate([model1.output, model2.output, model3.output])
        
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(50, activation="linear")(combined)
        
        z = Dense(10,activation='linear')(z)
        #dropout2 = hp.Float('dropout 2', min_value=0.0, max_value=0.9, step=.3)
        
        #z = Dropout(dropout2)(z)
        
        z = Dense(1, activation="linear")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        
        model = Model(inputs=[model1.input, model2.input, model3.input], outputs=z)
        
        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
        
        return model

class HyperModel:
    """
    Create hypermodel automatic tuner as described in keras documentation 
    https://www.tensorflow.org/tutorials/keras/keras_tuner in order to 
    automatically tune keras parameters without intervention.
    """

    def create_model(self,x,y):
        print("The program is creating a model\n")
        self.__create_hypertuning(x,y)
        
        now = datetime.now()
        self.time = now.strftime("%d_%m_%Y_%H_%M")
        self.modelpath = "data/model.h5"
        
        
    def __create_hypertuning(self,x,y):
        print("The program is creating hypertuning\n")
        self.tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=2,
                     factor=3,
                     directory='data',
                     project_name='hyperparamter_metadata')
    
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        
        self.tuner.search(x, y, epochs=10, validation_split=0.15,
                          callbacks=[stop_early],batch_size=32)

        # Get the optimal hyperparameters
        best_hps=self.tuner.get_best_hyperparameters(num_trials=1)[0]
        
        #print(f"""
        #The hyperparameter search is complete. The optimal number of units in the first densely-connected
        #layer is {best_hps.get('units layer 1')} and the optimal learning rate for the optimizer
        #is {best_hps.get('learning_rate')}.
        #""")
        
        self.best_hps = best_hps
        
        
    def train(self,x,y):
        
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        
        model = self.tuner.hypermodel.build(self.best_hps)
        """
        history = model.fit(x, y, epochs=25, validation_split=0.33,batch_size=32)
        
        val_acc_per_epoch = history.history['val_loss']
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        """
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # Retrain the model
        self.hypermodel.fit(x, y, epochs=20, validation_split=0.05)
        
        self.hypermodel.save(self.modelpath)
        
    def test(self,x,y=None):
        self.modelpath = "data/model.h5"
        if hasattr(self, 'hypermodel'):
            pass
        else:
            try:
                self.hypermodel = keras.models.load_model(self.modelpath)
            except:
                raise Exception("NO MODEL PATH EXISTS, ABORTING...")
                
        y_hat = self.hypermodel.predict(x)
        
        if y is not None:
            plot_predictions(y,y_hat,self.time)

        return y_hat
        