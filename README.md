# Tree GAN
A Deep Convolutional Network for generating images of trees. At the moment it is data and memory limited and is therefore inconsistent. However, it can produce low-quality textures and outlines of trees and branches.

Different optimisers, training data and archetecture produce very different results.

I built this project to explore DCGANs and to assess how useful they will be in the generation of outdoor landscapes.

# Dependancies
keras
tensorflow (theano may require tweaking the code)
scipy
h5py

# Data
72x72 and 140x140 greyscale images of trees from ImageNet are stored as numpy arrays in the SavedData folder.

# Generating Trees
In order to run the latest generator:
```
python GenerateTrees.py
```
If you want to retrain the network on the saved data:
```
python TreeNets.py
```
If you want to recreate the numpy arrays from a new data set:

1. Place images in the top level of the repositry in a folder named Trees
2. Run:
```python PrepTreeData.py```
