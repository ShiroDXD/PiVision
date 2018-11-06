import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = 'Data\Train'
valid_path = 'Data\Valid'
test_path = 'Data\Test'


train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    valid_path, target_size=(224, 224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    test_path, target_size=(224, 224), batch_size=10, shuffle=False)


mobile = keras.applications.mobilenet_v2.MobileNetV2()

mobile.summary()

x=mobile.layers[-6].output
x=Flatten(x)
predictions = Dense(14, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)

model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False


model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=18,
                    validation_data=valid_batches, validation_steps=3, epochs=60, verbose=2)


test_labels = test_batches.classes
test_labels
test_batches.class_indices

predictions = model.predict_generator(test_batches, steps=5, verbose=0)


model.save('retrainedMobileNet.h5')
