from tenforflow_gan.application import ImageSuperResolution
from tenforflow_gan.application import ImageSampler
from tensorflow.keras.callbacks import Tensorboard
from tenforflow_gan.reference import SAGAN
from tensorflow_gan.losses import perceptual_loss

# not ideal names - might need special cases for different use?
from tensorflow_gan.generator import discriminator_generator
from tensorflow_gan.generator import generator_generator

# callback works the same as TF2
callbacks = [
    Tensorboard('logs/'),
    # Many of the features such as showing images are  moved to normal callback
    ImageSampler(interval=5)

]

# instanciate application
gan = ImageSuperResolution()

# swap the discriminator
gan.discriminator = SAGAN.discriminator

# change optimize
generator_optimizer = SASGAN.generator_optimizer()
generator_optimizer.loss = perceptual_loss


#compile it with everything initalized by default
gan.compile(generator_optimizer=generator_optimizer)

# summary shows both network
gan.summary()

# fit as  you would normally do with TF2
gan.fit(discriminator_generator, generator_generator, epochs=100, batch_size=128, sample_interval=5)