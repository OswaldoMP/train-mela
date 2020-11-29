import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_train = './data/train'
data_test = './data/test'

epoca = 10
length, width = 100, 100
batchSize = 10
iterEpoca = 100
iterTrain = 200
filter1Cnn = 32
filter2Cnn = 64
filter1SizeCnn = (3, 3)
filter2SizeCnn = (2, 2)
poolSize = (2, 2)
classes = 1
lr = 0.0005

dataGenerateTrain = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True    
)

dataGenerateTest = ImageDataGenerator(
    rescale=1./255
)

imageTrain = dataGenerateTrain.flow_from_directory(
    data_test,
    target_size=(length, width),
    batch_size=batchSize,
    class_model='categorical'
)

imageTest = dataGenerateTest.flow_from_directory(
    data_test,
    target_size=(length, width),
    batch_size=batchSize,
    class_mode='categorical'
)

cnn = Sequential()

cnn.add(Convolution2D(filter1SizeCnn, filter1Cnn, padding='same', input_shape=(length, width, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=poolSize))

cnn.add(Convolution2D(filter2SizeCnn, filter2Cnn, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=poolSize))

cnn.add(Flatten())
cnn.add(Dense(150, activation='relu'))
cnn.add(Dropout(0.5))
ccn.add(Dense(classes, activation='softmax'))

cnn.compile(loss='categorical_crossentropy')
cnn.fit(imageTrain, steps_per_epoch=iterEpoca, epochs=epoca, validation_data=imageTest, validation_steps=iterTrain)

_dir = './model/'

if not os.path.exists(_dir):
    os.mkdir(_dir)

cnn.save(_dir+'/model.h5')
cnn.save_weights(_dir+'/pesos.h5')