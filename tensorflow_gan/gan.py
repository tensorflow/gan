"core gan class"

from collections import defaultdict
import numpy as np
from termcolor import cprint
from tensorflow.keras import layers  # pylint: disable=import-error
from tensorflow.keras.models import Model   # pylint: disable=import-error

from tensorflow_gan.utils import display_gray_images

# Check if we are in a ipython/colab environement
try:
    class_name = get_ipython().__class__.__name__
    if "Terminal" in class_name:
        IS_NOTEBOOK = False
    else:
        IS_NOTEBOOK = True

except NameError:
    IS_NOTEBOOK = False

if IS_NOTEBOOK:
    from tqdm import tqdm_notebook as tqdm
    from IPython.display import HTML
else:
    from tqdm import tqdm


class GAN():
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        # track epoch metrics
        self.metrics = defaultdict(list)

    def compile(self,
                optimizer='adam',
                loss='binary_crossentropy',

                discriminator_loss=None,
                discriminator_optimizer=None,
                discriminator_metrics=['accuracy'],

                gan_loss=None,
                gan_optimizer=None,
                gan_metrics=None):

        # [default values]
        if not discriminator_optimizer:
            discriminator_optimizer = optimizer
        if not gan_optimizer:
            gan_optimizer = optimizer

        if not discriminator_loss:
            discriminator_loss = loss
        if not gan_loss:
            gan_loss = loss

        # FIXME: metrics

        # [Discriminator]

        self.discriminator.compile(loss=discriminator_loss,
                                   optimizer=discriminator_optimizer,
                                   metrics=discriminator_metrics)
        self.discriminator_metric_names = [m.name for m in self.discriminator.metrics]  # nopep8
        self.discriminator_input_shape = self.discriminator.inputs[0].shape[1:]

        # FIXME: discriminator weight reload


        # [Generator]
        # self.generator.compile(loss=generator_loss,
                            #    optimizer=generator_optimizer,
                            #    metrics=generator_metrics)

        # self.generator_metric_names = [m.name for m in self.generator.metrics]
        self.generator_input_shape = self.generator.inputs[0].shape[1:]
        # FIXME: generator weight reload

        # [GAN]
        self.gan = self._build_gan(self.discriminator, self.generator)
        # ! must be set before gan compile AND after discriminator compile
        self.discriminator.trainable = False
        self.gan.compile(loss=gan_loss, optimizer=gan_optimizer,
                         metrics=gan_metrics)
        self.gan_metric_names = [m.name for m in self.gan.metrics]

    def fit(self, x, y=None, epochs=100, batch_size=32, sample_interval=10,
            generator_train_multiplier=1):
        self.batch_size = batch_size
        batches = int(len(x) / batch_size) * 2

        for epoch in range(epochs):
            self.metrics = defaultdict(list)
            # progress bar
            display_metrics = {}

            pb = tqdm(total=batches, desc="%s/%s epochs" %
                      (epoch + 1, epochs), unit='batch')

            for _ in range(batches):
                # numpy support passing multiples indices to collect data
                indices = np.random.randint(0, len(x), int(batch_size/2))
                if y:
                    y_batch = y[indices]
                else:
                    y_batch = None
                x_batch = x[indices]

                batch_metrics, x_fake = self.train_batch(x_batch, y_batch,
                                                         generator_train_multiplier=generator_train_multiplier  # nopep8
                                                        )

                # record metrics
                for m, v in batch_metrics.items():
                    self.metrics[m].append(v)
                    display_metrics[m] = "%.4f" % np.mean(self.metrics[m])
                # update progress bar
                pb.update()
                pb.set_postfix(display_metrics)
            pb.close()
            # print('[epoch:%s/%s]' % (epoch, epochs))
            if not epoch % sample_interval:
                display_gray_images(x_fake[:16])

        #final sampling
        display_gray_images(x_fake[:16])

    def gen_batch_data(self, x_batch, y_real=None):
        "generate the needed data"

        batch_size = self.batch_size
        # generate fake data with the generator
        x_generator = np.random.randn(batch_size,
                                      int(self.generator_input_shape[0]))
        x_fake = self.generator.predict(x_generator)
        y_fake = np.zeros((batch_size, 1))

        # label for the disriminator
        y_real = np.ones((batch_size, 1))

        return x_generator, x_fake, y_fake, y_real

    def train_batch(self, x_real, y_real=None, generator_train_multiplier=1):
        "Train a single batch"

        # generate data
        x_gen, x_fake, y_fake, y_real = self.gen_batch_data(x_real, y_real)

        # train discriminator
        half_size = int(self.batch_size / 2)  # only use half for discri
        rloss = self.discriminator.train_on_batch(x_real, y_real[:half_size])
        floss = self.discriminator.train_on_batch(x_fake[:half_size],
                                                  y_fake[:half_size])
        dloss = np.add(rloss, floss) * 0.5

        # FIXME: not working for some unknown reason
        # concatenating might improve backprop as you have examples of both per backprop
        # x_batch = np.concatenate((x_real, x_fake[:half_size]))
        # y_batch = np.concatenate((y_real[:half_size], y_fake[:half_size]))
        # dloss = self.discriminator.train_on_batch(x_batch, y_batch)

        # train generator
        for _ in range(generator_train_multiplier):
            gloss = self.gan.train_on_batch(x_gen, y_real)

        # compute metrics
        metrics = {}
        if len(self.discriminator_metric_names):
            metrics['D_loss'] = dloss[0]
            for idx, name in enumerate(self.discriminator_metric_names):
                metrics['D_' + name] = dloss[idx + 1]  # 0 is loss
        else:
            metrics['D_loss'] = dloss

        if len(self.gan_metric_names):
            metrics['G_loss'] = gloss[0]
            for idx, name in enumerate(self.gan_metric_names):
                metrics['G_' + name] = gloss[idx + 1]  # 0 is loss
        else:
            metrics['G_loss'] = gloss

        return metrics, x_fake

    def _build_gan(self, generator, discriminator,  generator_weights=None,
                   discriminator_weight=None):
        "build gan"

        # extract shape from the discriminator model
        input_shape = self.generator_input_shape
        x_in = layers.Input(shape=input_shape)
        x_g = self.generator(x_in)
        x_d = self.discriminator(x_g)
        return Model(x_in, x_d)

    def summary(self):
        "Display a summary of the various gan model"
        cprint('[Discriminator]', 'cyan')
        self.discriminator.summary()

        cprint('[Generator]', 'yellow')
        self.generator.summary()

        cprint('[GAN]', 'green')
        self.gan.summary()
