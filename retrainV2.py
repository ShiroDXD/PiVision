import keras
from keras.applications import MobileNetV2
from keras import models
from keras import layers
from keras import optimizers
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




mobile = keras.applications.mobilenet_v2.MobileNetV2()

for layer in mobile.layers[:-23]:
    layer.trainable = False

#for layer in mobile.layers:
  #  print(layer, layer.trainable)

train_path = 'Data\Train'
valid_path = 'Data\Valid'
test_path = 'Data\Test'


train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    valid_path, target_size=(224, 224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    test_path, target_size=(224, 224), batch_size=10, shuffle=False)