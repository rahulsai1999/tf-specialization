## Week 2 - Augmenting Images

- Image augmentation is one of the most widely used tools to increase the size of your dataset and increase the accuracy of your neural network with limited data. It basically involves transforms the images i.e. rotation, mirroring, skewing, cropping or warping the image.
- TensorFlow allows us to augment these images in memory thereby saving us lots of space and time.
- Augmenting images helps us in avoiding overfitting of the training data.

### Coding Augmentation

Augmentation is performed by using the ImageDataGenerator class which we have been using to rescale our images on-the-fly.

**Code**

```py
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```

**Purpose**

- **rescale** is used for normalizing the image as in the previous sections.
- **rotation_range** is set a range from 0 to N degrees(upto 180) from which a random value of rotation is selected.
- **width_shift and height_shift** are used to move the image around the frame as we don't want to overfit the centered subject onto the network.
- **shear_range**: shearing/skewing along the x-axis
- **zoom_range**: zooming a relative part of the image.
- **horizontal_flip**: mirroring or flipping the image.
- **fill_mode**: a parameter to deal with lost or corrupted pixels, here a nearest neighbour approach is used.
