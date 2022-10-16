import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf


class WGAN:
    def __init__(self, input_dim,
                 critic_conv_filters,
                 critic_conv_kernel_size,
                 critic_conv_strides,
                 critic_batch_norm_momentum,
                 critic_activation,
                 critic_dropout_rate,
                 critic_learning_rate,
                 generator_initial_dense_layer_size,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernel_size,
                 generator_conv_strides,
                 generator_batch_norm_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 optimizer, z_dim):
        self.name = "WGAN"

        self.input_dim = input_dim
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_conv_strides = critic_conv_strides
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.optimizer = optimizer
        self.z_dim = z_dim

        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_critic()
        self._build_generator()
        self._build_adversarial()

    def wasserstein(self, y_true, y_pred):
        return -tf.reduce_mean(y_true * y_pred)

    def _get_activation(self, activation):
        if activation == "leaky_relu":
            layer = keras.layers.LeakyReLU(alpha=0.2)
        else:
            layer = keras.layers.Activation(activation)
        return layer

    def _build_critic(self):
        # THE CRITIC
        critic_input = keras.Input(
            shape=self.input_dim, name="critic_input")

        x = critic_input
        for i in range(self.n_layers_critic):
            x = keras.layers.Conv2D(
                filters=self.critic_conv_filters[i],
                kernel_size=self.critic_conv_kernel_size[i],
                strides=self.critic_conv_strides[i],
                padding="same",
                kernel_initializer=self.weight_init,
                name=f"critic_conv_{i}"
            )(x)
            if self.critic_batch_norm_momentum and i > 0:
                x = keras.layers.BatchNormalization(
                    momentum=self.critic_batch_norm_momentum)(x)
            x = self._get_activation(self.critic_activation)(x)
            if self.critic_dropout_rate:
                x = keras.layers.Dropout(rate=self.critic_dropout_rate)(x)

        x = keras.layers.Flatten()(x)

        critic_output = keras.layers.Dense(
            1, activation=None, kernel_initializer=self.weight_init)(x)

        self.critic = keras.Model(critic_input, critic_output)

    def _build_generator(self):
        # THE GENERATOR
        generator_input = keras.Input(
            shape=(self.z_dim,), name="generator_input")

        x = generator_input
        x = keras.layers.Dense(np.prod(self.generator_initial_dense_layer_size),
                               kernel_initializer=self.weight_init)(x)
        if self.generator_batch_norm_momentum:
            x = keras.layers.BatchNormalization(
                momentum=self.generator_batch_norm_momentum)(x)
        x = self._get_activation(self.generator_activation)(x)
        x = keras.layers.Reshape(self.generator_initial_dense_layer_size)(x)
        if self.generator_dropout_rate:
            x = keras.layers.Dropout(rate=self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                x = keras.layers.UpSampling2D()(x)
                x = keras.layers.Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding="same",
                    kernel_initializer=self.weight_init,
                    name=f"generator_conv_{i}"
                )(x)
            else:
                x = keras.layers.Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    strides=self.generator_conv_strides[i],
                    padding="same",
                    kernel_initializer=self.weight_init,
                    name=f"generator_conv_{i}"
                )(x)
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = keras.layers.BatchNormalization(
                        momentum=self.generator_batch_norm_momentum)(x)
                x = self._get_activation(self.generator_activation)(x)
            else:
                x = keras.layers.Activation("tanh")(x)

        generator_output = x
        self.generator = keras.Model(generator_input, generator_output)

    def _get_optim(self, lr):
        if self.optimizer == "adam":
            optim = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimizer == "rmsprop":
            optim = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optim = keras.optimizers.Adam(learning_rate=lr)
        return optim

    def _set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):
        # COMPILE CRITIC
        self.critic.compile(
            optimizer=self._get_optim(self.critic_learning_rate),
            loss=self.wasserstein
        )

        # COMPILE THE FULL GAN
        self._set_trainable(self.critic, False)

        model_input = keras.Input(shape=(self.z_dim,), name="model_input")
        model_output = self.critic(self.generator(model_input))
        self.model = keras.Model(model_input, model_output)

        self.model.compile(
            optimizer=self._get_optim(self.generator_learning_rate),
            loss=self.wasserstein
        )

        self._set_trainable(self.critic, True)

    def _train_critic(self, x_train, batch_size, clip_threshold, using_generator):
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real = self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold)
                       for w in weights]
            l.set_weights(weights)

        return [d_loss, d_loss_real, d_loss_fake]

    def _train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder,
              print_every_n_batches=10, n_critic=5,
              clip_threshold=0.01, using_generator=False):
        for epoch in range(self.epoch, self.epoch+epochs):
            for _ in range(n_critic):
                d_loss = self._train_critic(
                    x_train, batch_size, clip_threshold, using_generator)

            g_loss = self._train_generator(batch_size)

            # print the progress
            print("Epoch {:03d} -- [D loss: {:.3f}(R {:.3f}, F {:.3f})] [G loss: {:.3f}]".format(
                epoch+1, d_loss[0], d_loss[1], d_loss[2], g_loss
            ))

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # if at save interval => save generated image samples
            if (epoch+1) % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(
                    run_folder, f"weights/weights-{epoch:d}.h5"))
                self.save_model(run_folder)

            self.epoch += 1

        self.save_model(run_folder)

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # rescale image 0 - 1
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]))
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(os.path.join(
            run_folder, f"images/sample_{self.epoch:d}.png"))
        plt.close()

    def plot_model(self, run_folder):
        keras.utils.plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'),
                               show_shapes=True, show_layer_names=True)
        keras.utils.plot_model(self.critic, to_file=os.path.join(run_folder, 'viz/critic.png'),
                               show_shapes=True, show_layer_names=True)
        keras.utils.plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png'),
                               show_shapes=True, show_layer_names=True)

    def save(self, folder):
        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump([
                self.input_dim,
                self.critic_conv_filters,
                self.critic_conv_kernel_size,
                self.critic_conv_strides,
                self.critic_batch_norm_momentum,
                self.critic_activation,
                self.critic_dropout_rate,
                self.critic_learning_rate,
                self.generator_initial_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate,
                self.optimizer,
                self.z_dim
            ], f)
        self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, "model.h5"))
        self.critic.save(os.path.join(run_folder, "critic.h5"))
        self.generator.save(os.path.join(run_folder, "generator.h5"))
        pickle.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
