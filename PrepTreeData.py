import numpy as np
from scipy import misc
import matplotlib.pylab as plt
from PIL import Image
import glob

number = 10
x = 72
y= 72
types = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".gif", ".pdf"]
files = []
for extension in types:
    files.extend(glob.glob("Trees/*"+extension))
trees = np.zeros([len(files)*2, x, y, 1])

#Converts all jpg images in trees folder into greyscale and desired resolution
for i in range(len(files)):
    im = Image.open(files[i]).convert("L")
    im2 = im.resize((x,y), Image.ANTIALIAS)
    tree = np.asarray(im2)
    tree2 = np.flip(tree, 1)
    trees[i, :, :, 0] = tree
    trees[i+len(files), :, :, 0] = tree2


#Saves as numpy array
np.save("SavedData/treesDat72.npy", trees)

#Displays some of the images
pics = trees[np.random.randint(0, trees.shape[0], size=number), :, :, 0]
for pic in pics:
    plt.figure()
    plt.imshow(pic, cmap='gray')
plt.show()
