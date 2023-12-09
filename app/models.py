import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, Input, Flatten, Reshape, Embedding, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model, save_model

import numpy as np

class CGAN(tf.keras.Model):
  def __init__(self, latent_dim, n_classes=10):
    super(CGAN, self).__init__()

    self.latent_dim = latent_dim
    self.n_classes = n_classes

    # create generator
    self.generator = self.build_generator(latent_dim, n_classes)

    # create discriminator
    self.discriminator = self.build_discriminator(n_classes=n_classes)

    # add tracker
    self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
    self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

  @property
  def metrics(self):
    return [self.gen_loss_tracker, self.disc_loss_tracker]

  def compile(self, d_optimizer, g_optimizer, loss_fn):
    tf.config.run_functions_eagerly(True)

    # set optimizers
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer

    # set loss function
    self.loss_fn = loss_fn

    super(CGAN, self).compile()

  def generate_latent_points(self, n_samples):
    x_input = np.random.randn(self.latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, self.latent_dim)
    x_labels = np.random.randint(0, self.n_classes, n_samples)
    return [x_input, x_labels]

  def generate_fake_samples(self, n_samples):
    # generate points in latent space
    x_input, x_labels = self.generate_latent_points(n_samples)

    # predict outputs
    imgs = self.generator([x_input, x_labels])

    # create class labels
    y = np.zeros((n_samples, 1))
    return [imgs, x_labels], y

  def generate_images(self, x_labels):
    # generate points in latent space
    x_input, _ = self.generate_latent_points(x_labels.size)

    # predict outputs
    imgs = self.generator.predict([x_input, x_labels])

    # return generated images
    return imgs

  @tf.function
  def train_step(self, data):
    # Unpack the data
    real_images, real_labels = data

    # Get batch size
    batch_size = tf.shape(real_images)[0]

    # Get randomly selected 'real' samples, with labels
    x_real, y_real = [real_images, real_labels], np.ones((batch_size, 1))

    # Update discriminator model weights
    with tf.GradientTape() as tape:
      predictions = self.discriminator(x_real)
      d_loss_real = self.loss_fn(y_real, predictions)
    grads = tape.gradient(d_loss_real, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

    # Generate 'fake' examples, with labels
    x_fake, y_fake = self.generate_fake_samples(batch_size)

    # Update discriminator model weights
    with tf.GradientTape() as tape:
      predictions = self.discriminator(x_fake)
      d_loss_fake = self.loss_fn(y_fake, predictions)
    grads = tape.gradient(d_loss_fake, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

    # Calculate loss and accuracy
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # Prepare points in latent space as input for the generator
    x_gan = self.generate_latent_points(2 * batch_size)

    # Create inverted labels for the fake samples
    y_gan = np.ones((2 * batch_size, 1))

    # Update the generator via the discriminator's error
    with tf.GradientTape() as tape:
      fake_imgs = self.generator(x_gan)
      fake_imgs_and_labels = [fake_imgs, x_gan[1]]
      predictions = self.discriminator(fake_imgs_and_labels)
      g_loss = self.loss_fn(y_gan, predictions)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

    # Monitor loss
    self.gen_loss_tracker.update_state(g_loss)
    self.disc_loss_tracker.update_state(d_loss)

    return {
        "g_loss": self.gen_loss_tracker.result(),
        "d_loss": self.disc_loss_tracker.result(),
    }

  def build_discriminator(self, in_shape=(28, 28, 1), n_classes=10):
    # label input
    i_label = Input(shape=(1, ))
    x_label = Embedding(n_classes, 50)(i_label)
    x_label = Dense(in_shape[0] * in_shape[1], activation='tanh')(x_label)
    x_label = Reshape((in_shape[0], in_shape[1], 1))(x_label)

    # image input
    i_img = Input(shape=in_shape)

    # concatenate
    x = Concatenate()([x_label, i_img])

    # conv layers
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='tanh')(i_img)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='tanh')(x)
    x = Flatten()(x)
    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([i_img, i_label], x)
    return model

  def build_generator(self, latent_dim, n_classes=10):
    # label input
    i_label = Input(shape=(1, ))
    x_label = Embedding(n_classes, 50)(i_label)
    x_label = Dense(7 * 7 * 3, activation='tanh')(x_label)
    x_label = Reshape((7, 7, 3))(x_label)

    # foundation for 7x7 image
    i_lat = Input(shape=(latent_dim, ))
    x_lat = Dense(7 * 7 * 125, activation='tanh')(i_lat)
    x_lat = Reshape((7, 7, 125))(x_lat)

    # concatenate
    x = Concatenate()([x_lat, x_label])

    # upsample to 14x14
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)

    # upsample to 28x28
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)

    # make only one color channel, values in (-1, +1)
    x = Conv2D(1, (7, 7), activation='tanh', padding='same')(x)
    return Model([i_lat, i_label], x)
    if not os.path.exists('gan_images'):
      os.makedirs('gan_images')

    data, _ = self.generate_fake_samples(25)
    imgs, labels = data
    rows, cols = 5, 5

    # Rescale images (-1, +1) -> (0, 1)
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols, figsize=(9, 10))
    idx = 0
    for i in range(rows):
      for j in range(cols):
        axs[i, j].imshow(imgs[idx], cmap='gray')
        axs[i, j].set_title(f'num: {labels[idx]}')
        axs[i, j].axis('off')
        idx += 1

    if method == 'show':
      plt.show()
    elif method == 'save':
      fig.savefig(f'gan_images/sample_{e:003d}_{b:0004d}.png')
      plt.close()

  # save models
  def save(self, filepath, overwrite=True, save_format=None, **kwargs):
    self.generator.save(filepath + 'generator.h5')
    self.discriminator.save(filepath + 'discriminator.h5')

  # load models
  def load_models(self, generator_path, discriminator_path):
    self.generator = load_model(generator_path, compile=False)
    self.discriminator = load_model(discriminator_path, compile=False)