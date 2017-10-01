from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

'''Generates 10 images from the GAN'''

features = 100
name = "Generated Trees"

m = load_model('SavedData/generator.h5')
for i in range(10):
    ran = np.random.uniform(-1.0, 1.0, size=[1,features])
    fake = m.predict(ran)
    fake = fake.reshape(fake.shape[1], fake.shape[2])
    plt.figure()
    plt.title(name)
    plt.imshow(fake, cmap='gray')
plt.show()
