
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from time import sleep

class GAN(object):
    def __init__(self, discriminator, generator, discOptimiser, advOptimiser, baseFeatures):
        '''Setup and prepare training models'''
        self.discNet= discriminator
        self.gen = generator
        self.discriminatorCompile(discOptimiser)
        self.adversarialCompile(advOptimiser)
        self.feats = baseFeatures

    def discriminatorCompile(self, optimiser):
        '''Compiles the provided discriminator for training'''
        #optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.dis = Sequential()
        self.dis.add(self.discNet)
        self.dis.compile(loss='mean_squared_error', optimizer=optimiser)
        return self.dis

    def adversarialCompile(self, optimiser):
        '''Compiles the adversarial net'''
        #optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.adv = Sequential()
        self.adv.add(self.gen)
        self.adv.add(self.discNet)
        self.adv.compile(loss='mean_squared_error', optimizer=optimiser)
        return self.adv

    def train(self, batches, batch_size, x_tr):
        '''Trains both models in batches.
         Will run until the generator has been trained on desired number of batches'''
        c = 0
        i = 0
        prevI = 0
        while i < batches:
            #Allow discriminator training
            self.discNet.trainable = True
            real = x_tr[np.random.randint(0, x_tr.shape[0], size=batch_size), :, :, :]
            #Generate based on gaussian noise
            ran = np.random.normal(-1.0, 1.0, size=[batch_size, self.feats])
            fake = self.gen.predict(ran)
            #Uncomment for pure noise input
            #fake = np.random.normal(-1.0, 1.0, size=[batch_size, x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]])
            x_dis = np.concatenate((real, fake))
            y_dis = np.ones([2*batch_size, 1])
            #Smooth Labels
            y_dis[:batch_size, :] = np.random.uniform(0.7, 1.2, size=[batch_size,1])
            y_dis[batch_size:, :] = np.random.uniform(0.0, 0.3, size=[batch_size,1])
            dis_result = self.dis.train_on_batch(x_dis, y_dis)

            #Only traini generator when discriminator is performing well
            if dis_result<0.2:
                i = i+1
                #Stop discriminator training
                self.discNet.trainable = False
                ran = np.random.normal(-1.0, 1.0, size=[batch_size, self.feats])
                x_gen = ran
                #Smooth Labels
                y_gen = np.random.uniform(0.7, 1.2, size=[batch_size,1])
                gen_result = self.adv.train_on_batch(x_gen, y_gen)
            else:
                gen_result = -1

            #Ends if it looks like discriminator is diverging
            if dis_result>0.3:
                c = c+1
                if c>5:
                    break
            else:
                c = 0

            #Show a couple of images from the generator 5 times during the training
            if i%int(batches/5) == 0 and prevI != i:
                self.testGen(2, str(i))
                prevI = i

            print("%d Dis loss %f    Gen loss %f" %(i, dis_result, gen_result))

    def testGen(self, number, name= ""):
        '''Displays a number of generated images'''
        for i in range(number):
            ran = np.random.uniform(-1.0, 1.0, size=[1,self.feats])
            fake = self.gen.predict(ran)
            fake = fake.reshape(fake.shape[1], fake.shape[2])
            plt.figure()
            plt.title(name)
            plt.imshow(fake, cmap='gray')
