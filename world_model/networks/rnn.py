import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

Z_DIM = 32
ACTION_DIM = 3
REWARD_DIM = 1

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

Z_FACTOR = 1
REWARD_FACTOR = 1

LEARNING_RATE = 0.001


class RNN():
    def __init__(self):
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.hidden_units = HIDDEN_UNITS
        self.gaussian_mixtures = GAUSSIAN_MIXTURES
        self.reward_factor = REWARD_FACTOR
        self.learning_rate = LEARNING_RATE

        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]

    def _build(self):
        # THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM + REWARD_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output_model, _, _ = lstm(rnn_x)
        mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM) + 1)

        mdn_model = mdn(lstm_output_model)

        model = Model(rnn_x, mdn_model)

        # THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))

        lstm_output_forward, state_h, state_c = lstm(
            rnn_x, initial_state=[state_input_h, state_input_c])

        mdn_forward = mdn(lstm_output_forward)

        forward = Model([rnn_x, state_input_h, state_input_c],
                        [mdn_forward, state_h, state_c])

        # LOSS FUNCTION
        def rnn_z_loss(y_true, y_pred):
            z_true, rew_true = self.get_responses(y_true)

            d = GAUSSIAN_MIXTURES * Z_DIM
            z_pred = y_pred[:, :, :(3*d)]
            z_pred = tf.reshape(z_pred, [-1, GAUSSIAN_MIXTURES * 3])

            log_pi, mu, log_sigma = self.get_mixture_coef(z_pred)

            flat_z_true = tf.reshape(z_true, [-1, 1])

            z_loss = log_pi + self.tf_log_normal(flat_z_true, mu, log_sigma)
            z_loss = -tf.math.log(tf.reduce_sum(tf.exp(z_loss), 1, keepdims=True))
            z_loss = tf.reduce_mean(z_loss)

            return z_loss

        def rnn_rew_loss(y_true, y_pred):
            z_true, rew_true = self.get_responses(y_true)  # , done_true

            reward_pred = y_pred[:, :, -1]

            rew_loss = tf.keras.metrics.binary_crossentropy(
                rew_true, reward_pred, from_logits=True)
            rew_loss = tf.reduce_mean(rew_loss)

            return rew_loss

        def rnn_loss(y_true, y_pred):
            z_loss = rnn_z_loss(y_true, y_pred)
            rew_loss = rnn_rew_loss(y_true, y_pred)

            return Z_FACTOR * z_loss + REWARD_FACTOR * rew_loss

        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss=rnn_loss, optimizer=optimizer, metrics=[
                      rnn_z_loss, rnn_rew_loss])

        return (model, forward)

    def get_responses(self, y_true):
        z_true = y_true[:, :, :Z_DIM]
        rew_true = y_true[:, :, -1]

        return z_true, rew_true

    def get_mixture_coef(self, z_pred):
        log_pi, mu, log_sigma = tf.split(z_pred, 3, 1)
        # axis 1 is the mixture axis
        log_pi = log_pi - tf.math.log(tf.reduce_sum(tf.exp(log_pi), axis=1, keepdims=True))

        return log_pi, mu, log_sigma

    def tf_log_normal(self, z_true, mu, log_sigma):
        log_sqrt_two_PI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((z_true - mu) / tf.exp(log_sigma)) ** 2 - log_sigma - log_sqrt_two_PI

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output):
        self.model.fit(rnn_input, rnn_output,
                       shuffle=False,
                       epochs=1,
                       batch_size=len(rnn_input))

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
