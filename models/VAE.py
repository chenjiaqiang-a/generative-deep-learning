import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from utils import CustomCallback, step_decay_schedule


disable_eager_execution()


class VariationalAutoEncoder:
    def __init__(self,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_strides,
                 z_dim,
                 use_batch_norm=False,
                 use_dropout=False):
        self.name = "VariationalAutoEncoder"

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()

    def _build(self):
        encoder_input = tf.keras.Input(
            shape=self.input_dim, name="encoder_input")

        x = encoder_input
        for i in range(self.n_layers_encoder):
            conv_layer = tf.keras.layers.Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding="same",
                name=f"encoder_conv_{i}"
            )

            x = conv_layer(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            if self.use_dropout:
                x = tf.keras.layers.Dropout(rate=0.25)(x)

        shape_before_flatten = tf.shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)
        self.mu = tf.keras.layers.Dense(self.z_dim, name="mu")(x)
        self.log_var = tf.keras.layers.Dense(self.z_dim, name="log_var")(x)
        self.encoder_mu_log_var = tf.keras.Model(
            encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = tf.random.normal(
                shape=tf.shape(mu), mean=0.0, stddev=1.0)
            return mu + tf.exp(log_var / 2) * epsilon
        encoder_output = tf.keras.layers.Lambda(
            sampling, name="encoder_output")([self.mu, self.log_var])

        self.encoder = tf.keras.Model(encoder_input, encoder_output)

        # THE DECODER
        decoder_input = tf.keras.Input(
            shape=(self.z_dim,), name="decoder_input")

        x = tf.keras.layers.Dense(np.prod(shape_before_flatten))(decoder_input)
        x = tf.keras.layers.Reshape(shape_before_flatten)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = tf.keras.layers.Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding="same",
                name=f"decoder_conv_t_{i}"
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.LeakyReLU()(x)
                if self.use_dropout:
                    x = tf.keras.layers.Dropout(rate=0.25)(x)
            else:
                x = tf.keras.layers.Activation("sigmoid")(x)

        decoder_output = x
        self.decoder = tf.keras.Model(decoder_input, decoder_output)

        # THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = tf.keras.Model(model_input, model_output)

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        # COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * \
                tf.reduce_sum(1 + self.log_var -
                              tf.square(self.mu) - tf.exp(self.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss,
                           metrics=[vae_r_loss, vae_kl_loss])

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, "viz"))
            os.makedirs(os.path.join(folder, "weights"))
            os.makedirs(os.path.join(folder, "images"))

        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout
            ], f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self,
              x_train,
              batch_size,
              epochs,
              run_folder,
              steps_per_epoch=None,
              print_every_n_batches=100,
              initial_epoch=0,
              lr_decay=1):
        custom_callback = CustomCallback(run_folder,
                                         print_every_n_batches,
                                         initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate,
                                       decay_factor=lr_decay,
                                       step_size=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_folder, "weights/weights.h5"),
                                                        save_weights_only=True,
                                                        verbose=1)

        callback_list = [checkpoint, custom_callback, lr_sched]

        self.model.fit(
            x_train, x_train,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callback_list
        )

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.model,
                                  to_file=os.path.join(
                                      run_folder, "viz/model.png"),
                                  show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.encoder,
                                  to_file=os.path.join(
                                      run_folder, "viz/encoder.png"),
                                  show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder,
                                  to_file=os.path.join(
                                      run_folder, "viz/decoder.png"),
                                  show_shapes=True, show_layer_names=True)
