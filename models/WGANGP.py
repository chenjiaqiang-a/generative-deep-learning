import os
import pickle
from functools import partial
from this import d
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import _Merge
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.keras.utils.vis_utils import plot_model


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP:
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
                 optimizer,
                 grad_weight,
                 z_dim,
                 batch_size) -> None:
        self.name = "WGANGP"

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

        self.weight_init = RandomNormal(mean=0.0, stddev=0.02)
        self.grad_weight = grad_weight
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self._build_critic()
        self._build_generator()
        self._build_adversarial()

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring
        gradients_sqr = K.square(gradients)
        # summing over the rows
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        # and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)
    
    def _get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer
    
    def _build_critic(self):
        # THE CRITIC
        critic_input = Input(shape=self.input_dim, name='critic_input')

        x = critic_input
        for i in range(self.n_layers_critic):
            x = Conv2D(
                filters=self.critic_conv_filters[i],
                kernel_size=self.critic_conv_kernel_size[i],
                strides=self.critic_conv_strides[i],
                padding='same',
                kernel_initializer=self.weight_init,
                name=f'critic_conv_{i}'
            )(x)
            if self.critic_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.critic_batch_norm_momentum)(x)
            x = self._get_activation(self.critic_activation)(x)
            if self.critic_dropout_rate:
                x = Dropout(rate=self.critic_dropout_rate)(x)
        
        x = Flatten()(x)
        critic_output = Dense(1, activation=None, kernel_initializer=self.weight_init)(x)

        self.critic = Model(critic_input, critic_output)
    
    def _build_generator(self):
        # THE GENERATOR
        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input
        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer=self.weight_init)(x)
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
        x = self._get_activation(self.generator_activation)(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)
        
        for i in range(self.n_layers_generator):
            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    padding='same',
                    kernel_initializer=self.weight_init,
                    name=f'generator_conv_{i}'
                )(x)
            else:
                x = Conv2DTranspose(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernel_size[i],
                    strides=self.generator_conv_strides[i],
                    padding='same',
                    kernel_initializer=self.weight_init,
                    name=f'generator_conv_{i}'
                )(x)
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
                x = self._get_activation(self.generator_activation)(x)
            else:
                x = Activation('tanh')(x)
        
        generator_output = x
        self.generator = Model(generator_input, generator_output)
    
    def _get_optim(self, lr):
        if self.optimizer == 'adam':
            optim = Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimizer == 'rmsprop':
            optim = RMSprop(learning_rate=lr)
        else:
            optim = Adam(learning_rate=lr)
        return optim
    
    def _set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val
    
    def _build_adversarial(self):
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self._set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = Input(shape=self.input_dim)

        # Fake image
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # critic determins validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional 'interpolated_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
            optimizer=self._get_optim(self.critic_learning_rate),
            loss_weights=[1, 1, self.grad_weight]
        )

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self._set_trainable(self.critic,  False)
        self._set_trainable(self.generator, True)

        # Sampled noise for input to generator
        model_input = Input(shape=(self.z_dim,))
        # Generate images based of noise
        img = self.generator(model_input)
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines generator model
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=self._get_optim(self.generator_learning_rate),
            loss=self.wasserstein
        )

        self._set_trainable(self.critic, True)
    
    def _train_critic(self, x_train, batch_size, using_generator):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32)

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        d_loss = self.critic_model.train_on_batch([true_imgs, noise], [valid, fake, dummy])
        return d_loss
    
    def _train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)
    
    def train(self, x_train, batch_size, epochs, run_folder,
            print_every_n_batches=10, n_critic=5, using_generator=False):
        for epoch in range(self.epoch, self.epoch + epochs):
            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic
            
            for _ in range(critic_loops):
                d_loss = self._train_critic(x_train, batch_size, using_generator)
            
            g_loss = self._train_generator(batch_size)

            print(f"{epoch:d} ({critic_loops:d}, {1:d}) "
                  f"[D loss: {d_loss[0]:.1f} (R {d_loss[1]:.1f}, F {d_loss[2]:.1f}, G {d_loss[3]:.1f})]"
                  f"[G loss: {g_loss:.1f}]")

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, f"weights/weights-{epoch:d}.h5"))
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
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='Grey')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, f"images/sample_{self.epoch}.png"))
        plt.close()
    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.critic, to_file=os.path.join(run_folder ,'viz/critic.png'),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'),
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
                self.grad_weight,
                self.z_dim,
                self.batch_size
            ], f)
        self.plot_model(folder)
    
    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, "model.h5"))
        self.critic.save(os.path.join(run_folder, "critic.h5"))
        self.generator.save(os.path.join(run_folder, "generator.h5"))
        pickle.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
