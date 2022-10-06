import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class GAN:
    def __init__(self, input_dim,
                 discriminator_conv_filters,
                 discriminator_conv_kernel_size,
                 discriminator_conv_strides,
                 discriminator_batch_norm_momentum,
                 discriminator_activation,
                 discriminator_dropout_rate,
                 discriminator_learning_rate,
                 generator_initial_dense_layer_size,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernel_size,
                 generator_conv_strides,
                 generator_batch_norm_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 optimizer,
                 z_dim):
        self.name = "GAN"

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

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

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()
        self._build_adversarial()

    def _get_activation(self, activation):
        if activation == "leaky_relu":
            layer = tf.keras.layers.LeakyReLU(alpha=0.2)
        else:
            layer = tf.keras.layers.Activation(activation)
        return layer

    def _build_discriminator(self):
        # THE DISCRIMINATOR
        discriminator_input = tf.keras.Input(
            shape=self.input_dim, name="discriminator_input")

        x = discriminator_input
        for i in range(self.n_layers_discriminator):
            x = tf.keras.layers.Conv2D(
                filters=self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernel_size[i],
                strides=self.discriminator_conv_strides[i],
                padding="same",
                kernel_initializer=self.weight_init,
                name=f"discriminator_conv_{i}"
            )(x)
            if self.discriminator_batch_norm_momentum and i > 0:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.discriminator_batch_norm_momentum)(x)
            x = self._get_activation(self.discriminator_activation)(x)
            if self.discriminator_dropout_rate:
                x = tf.keras.layers.Dropout(
                    rate=self.discriminator_dropout_rate)(x)

        x = tf.keras.layers.Flatten()(x)
        discriminator_output = tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer=self.weight_init)(x)

        self.discriminator = tf.keras.Model(
            discriminator_input, discriminator_output)

    def _build_generator(self):
        # THE GENERATOR
        generator_input = tf.keras.Input(
            shape=(self.z_dim,), name="generator_input")

        x = generator_input
        x = tf.keras.layers.Dense(np.prod(
            self.generator_initial_dense_layer_size), kernel_initializer=self.weight_init)(x)
        if self.generator_batch_norm_momentum:
            x = tf.keras.layers.BatchNormalization(
                momentum=self.generator_batch_norm_momentum)(x)
        x = self._get_activation(self.generator_activation)(x)
        x = tf.keras.layers.Reshape(self.generator_initial_dense_layer_size)(x)
        if self.generator_dropout_rate:
            x = tf.keras.layers.Dropout(rate=self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                x = tf.keras.layers.UpSampling2D()(x)
                x = tf.keras.layers.Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding="same",
                    kernel_initializer=self.weight_init,
                    name=f"generator_conv_{i}"
                )(x)
            else:
                x = tf.keras.layers.Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    strides=self.generator_conv_strides[i],
                    padding="same",
                    kernel_initializer=self.weight_init,
                    name=f"generator_conv_{i}"
                )(x)
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = tf.keras.layers.BatchNormalization(
                        momentum=self.generator_batch_norm_momentum)(x)
                x = self._get_activation(self.generator_activation)(x)
            else:
                x = tf.keras.layers.Activation("tanh")(x)

        generator_output = x
        self.generator = tf.keras.Model(generator_input, generator_output)

    def _get_optim(self, lr):
        if self.optimizer == "adam":
            optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimizer == "rmsprop":
            optim = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optim = tf.keras.optimizers.Adam(learning_rate=lr)
        return optim

    def _set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):
        # COMPILE DISCRIMINATOR
        self.discriminator.compile(
            optimizer=self._get_optim(self.discriminator_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # COMPILE THE FULL GAN
        self._set_trainable(self.discriminator, False)

        model_input = tf.keras.Input(shape=(self.z_dim,), name="model_input")
        model_output = self.discriminator(self.generator(model_input))
        self.model = tf.keras.Model(model_input, model_output)

        self.model.compile(
            optimizer=self._get_optim(self.generator_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self._set_trainable(self.discriminator, True)

    def _train_discriminator(self, x_train, batch_size, using_generator):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(
            true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(
            gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def _train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder,
              print_every_n_batches=50, using_generator=False):
        for epoch in range(self.epoch, self.epoch + epochs):
            d = self._train_discriminator(x_train, batch_size, using_generator)
            g = self._train_generator(batch_size)

            print(f"{epoch:d} [D loss: {d[0]:.3f}(R {d[1]:.3f}, F{d[2]:.3f})] [D acc: {d[3]:.3f}(R {d[4]:.3f}, F {d[5]:.3f})] [G loss: {d[0]:.3f}] [G acc: {g[1]:.3f}]")

            self.d_losses.append(d)
            self.g_losses.append(g)

            if (epoch + 1) % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(
                    run_folder, f"weights/weights-{epoch:d}.h5"))
                self.model.save_weights(os.path.join(
                    run_folder, f"weights/weights.h5"))
                self.save_model(run_folder)

            self.epoch += 1

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(
                    gen_imgs[cnt, :, :, :]), cmap="Greys")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(os.path.join(
            run_folder, f"images/sample_{self.epoch:d}.png"))
        plt.close()

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(run_folder, "viz/model.png"),
                                  show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(run_folder, "viz/discriminator.png"),
                                  show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(run_folder, "viz/generator.png"),
                                  show_shapes=True, show_layer_names=True)

    def save(self, folder):
        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pkl.dump([
                self.input_dim,
                self.discriminator_conv_filters,
                self.discriminator_conv_kernel_size,
                self.discriminator_conv_strides,
                self.discriminator_batch_norm_momentum,
                self.discriminator_activation,
                self.discriminator_dropout_rate,
                self.discriminator_learning_rate,
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
        self.discriminator.save(os.path.join(run_folder, "discriminator.h5"))
        self.generator.save(os.path.join(run_folder, "generator.h5"))
        pkl.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
