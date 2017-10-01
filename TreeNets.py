from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D
from keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from GAN import GAN

def discriminator(size):
    '''Builds discriminator CN'''
    discriminator = Sequential()
    depth = 64
    dropout = 0.4
    input_shape = (size, size, 1)

    discriminator.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))
    discriminator.add(Conv2D(128, 5, strides=2, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))
    discriminator.add(Conv2D(256, 5, strides=2, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))
    discriminator.add(Conv2D(512, 5, strides=1, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))
    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    return discriminator

def generator(size, baseFeatures):
    '''Builds Generator CN'''
    generator = Sequential()
    dropout = 0.4
    depth = 256
    dim = int(size/4)

    generator.add(Dense(dim*dim*depth, input_dim=baseFeatures))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Reshape((dim, dim, depth)))
    generator.add(Dropout(dropout))
    generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(128, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(64, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Conv2DTranspose(32, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Conv2DTranspose(1, 5, padding='same'))
    generator.add(Activation('sigmoid'))

    return generator

if __name__ == '__main__':
    '''Creates nets and rins GAN'''
    #Number of features to be encoded
    baseFeatures = 100
    #Number of batches
    batches = 100
    #Resolution of the pictures
    size = 72
    #Which array to load
    path = "SavedData/treesDat72.npy"
    #Optimisers. Uncomment for other optimisers
    discOptimiser = SGD()
    advOptimiser = Adam(lr=0.0002, beta_1=0.5)

    x_tr = np.load(path)
    gan = GAN(discriminator(size), generator(size, baseFeatures), discOptimiser, advOptimiser, baseFeatures)
    gan.train(batches, 256, x_tr)
    gan.testGen(10, "Final")
    plt.show()
    #Save models
    gan.discNet.save('SavedData/discriminator.h5')
    gan.gen.save('SavedData/generator.h5')
