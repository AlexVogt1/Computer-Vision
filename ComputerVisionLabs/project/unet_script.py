#########################
# Matthew Dacre 2091295 #
#                       #
# Joshua Wacks 2143116  #
#                       #
# Alex Vogt 20152320    #
#########################

import numpy as np
import imageio
from glob import glob
from skimage import img_as_float32
from natsort import natsorted
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from skimage import img_as_float32
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate
from skimage.util import random_noise

def split(imgs, msks, seed=42):

    ''' Splitting data using indexing slicing'''

    random.seed(seed)
    c = list(zip(imgs, msks))
    random.shuffle(c)
    imgs, msks = zip(*c)

    train = 34
    val = 7
    test = 7

    x_train, y_train = imgs[:train], msks[:train]
    x_val, y_val = imgs[train:train+val], msks[train:train+val]
    x_test, y_test = imgs[-1*test:], msks[-1*test:]

    return x_train, y_train, x_val, y_val, x_test, y_test

# Reading in data
path_pairs = list(zip(
natsorted(glob('./puzzle_corners_1024x768/images-1024x768/*.png')),
natsorted(glob('./puzzle_corners_1024x768/masks-1024x768/*.png')),
))

imgs = np.array([cv2.resize(img_as_float32(imageio.imread(ipath)), (192, 256)) for ipath, _ in path_pairs])
msks = np.array([cv2.resize(img_as_float32(imageio.imread(mpath)), (192, 256)) for _, mpath in path_pairs])

x_train, y_train, x_val, y_val, x_test, y_test = split(imgs, msks)


# Getting pretrained VGG16 weights from tensorflow
vgg16_pretrained = VGG16(include_top=False, weights='imagenet', input_shape=(256, 192, 3))

def augment_data(x, y, k = (3, 3), sx = 1, size=(192, 256)):

    ''' Augmenting data using blurring, flipping and noise '''

    def brightness(img, scale):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * scale

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def sat(img, scale):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * scale

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    images = list(x)
    masks = list(y)

    

    for i in range(len(x)):
        # Flip
        images.append(cv2.flip(x[i], 0))
        images.append(cv2.flip(x[i], 1))
        masks.append(cv2.flip(y[i], 0))
        masks.append(cv2.flip(y[i], 1))
    
    for i in range(len(images)):
        # Blur and Noise
        images.append(cv2.GaussianBlur(images[i], k, sx))
        images.append(random_noise(images[i], mode='s&p'))
        masks.append(masks[i])
        masks.append(masks[i])

        
    for i in range(len(masks)):
        images[i] = cv2.resize(images[i], size)
        masks[i] = cv2.resize(masks[i], size)



    return np.array(images), np.array(masks)

def get_data(x_train, y_train, x_val, y_val, x_test, y_test):

    ''' Augmenting and placing data in tensorflow Dataset for training '''

    x_train, y_train = augment_data(x_train, y_train)
    x_val = np.array(list(x_val))
    y_val = np.array(list(y_val))
    x_test = np.array(list(x_val))
    y_test = np.array(list(y_val))

    

    data_train = tf.data.Dataset.from_tensor_slices((

            {
                'x' : np.expand_dims(x_train, axis=0)
            },
            {
                'y' : tf.one_hot(np.expand_dims(y_train, axis=0), depth=2, dtype=tf.int32)
            }
        )
    )

    data_val = tf.data.Dataset.from_tensor_slices((

            {
                'x' : np.expand_dims(x_val, axis=0)
            },
            {
                'y' : tf.one_hot(np.expand_dims(y_val, axis=0), depth=2, dtype=tf.int32)
            }
        )
    )

    data_test = tf.data.Dataset.from_tensor_slices((

            {
                'x' : np.expand_dims(x_test, axis=0)
            },
            {
                'y' : tf.one_hot(np.expand_dims(y_test, axis=0), depth=2, dtype=tf.int32)
            }
        )
    )
    return data_train, data_val, data_test

# Getting trainig/testing data
train, val, test = get_data(x_train, y_train, x_val, y_val, x_test, y_test)


# Model definition
class UnetVGG():
    def __init__(self, input: tuple, num_classes: int, lr: float):
        self.input_shape = input
        self.num_classes = num_classes
        self.learning_rate = lr

        self.optimizer = Nadam(learning_rate=lr)
        self.model = self.build_model(self.input_shape, self.num_classes)

        self.model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])

    def build_model(self, input_shape, num_classes):
        encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        encoder.layers[0]._name  = 'x'

        encoder_output = encoder.output
        conv0 = Conv2D(512, 3, activation='relu', padding='same')(encoder_output)
        conv0_trans = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(conv0)

        concat1 = concatenate([encoder.get_layer('block5_conv3').output, conv0_trans])
        conv1 = Conv2D(512, 3, activation='relu', padding='same')(concat1)
        conv1_trans = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(conv1)

        concat2 = concatenate([encoder.get_layer('block4_conv3').output, conv1_trans])
        conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat2)
        conv2_trans = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(conv2)

        concat3 = concatenate([encoder.get_layer('block3_conv3').output, conv2_trans])
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat3)
        conv3_trans = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv3)

        concat_4 = concatenate([encoder.get_layer('block2_conv2').output, conv3_trans])
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_4)
        conv4_trans = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(conv4)

        concat5 = concatenate([encoder.get_layer('block1_conv2').output, conv4_trans])
        conv5 = Conv2D(32, 3, activation='relu', padding='same')(concat5)

        out = Conv2D(num_classes, 1, activation='softmax', name='y')(conv5)

        return Model(inputs = encoder.layers[0].input, outputs=out, name="UnetVGG")

    def train(self, train, val, epochs):
        self.model.fit(train, validation_data = val, epochs=epochs)

    def predict(self, data):
        return np.squeeze(self.model.predict(data))

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.model.save(path)
# Flusing python output to ensure training notifications are printed
print("Flushed", flush=True)

HYPER_PARAMS = {
    'shape' : (256, 192, 3), 
    'num_classes' : 2,
    'lr' : 1e-3,
    'epochs' : 1000,
    'savedir' : './model'
}

model = UnetVGG(HYPER_PARAMS['shape'], HYPER_PARAMS['num_classes'], HYPER_PARAMS['lr'])
model.train(train, val, HYPER_PARAMS['epochs'])
model.save(HYPER_PARAMS['savedir'])