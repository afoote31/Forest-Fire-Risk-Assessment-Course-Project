# %%
# Import packages
import os
from tensorflow import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Prepare the data generators
base_dir = '/Users/aaronfoote/COURSES/ME315/FireRisk'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'val')


train_High_dir = os.path.join(train_dir, 'High')
train_Low_dir = os.path.join(train_dir, 'Low')
train_Moderate_dir = os.path.join(train_dir, 'Moderate')
train_Non_Burnable_dir = os.path.join(train_dir, 'Non-burnable')
train_Very_High_dir = os.path.join(train_dir, 'Very_High')
train_Very_Low_dir = os.path.join(train_dir, 'Very_Low')
train_Water_dir = os.path.join(train_dir, 'Water')

test_High_dir = os.path.join(test_dir, 'High')
test_Low_dir = os.path.join(test_dir, 'Low')
test_Moderate_dir = os.path.join(test_dir, 'Moderate')
test_Non_Burnable_dir = os.path.join(test_dir, 'Non-burnable')
test_Very_High_dir = os.path.join(test_dir, 'Very_High')
test_Very_Low_dir = os.path.join(test_dir, 'Very_Low')
test_Water_dir = os.path.join(test_dir, 'Water')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainValDataGenerator = ImageDataGenerator(rescale = 1./255., validation_split = 0.25)
testDataGenerator = ImageDataGenerator(rescale = 1./255.)

trainGenerator = trainValDataGenerator.flow_from_directory(
    directory = train_dir,
    target_size = (270,270),
    class_mode = "categorical",
    subset = 'training',
    batch_size = 16
    )
valGenerator = trainValDataGenerator.flow_from_directory(
    directory = train_dir,
    target_size = (270,270),
    class_mode = 'categorical',
    subset = 'validation',
    batch_size = 16
)

testGenerator = testDataGenerator.flow_from_directory(
    directory = test_dir,
    target_size = (270,270),
    class_mode = 'categorical'
)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# %%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model

# %% Define the Models
# Start off with a brutally large single dense layer. No convolution to speak of. This is the baseline
baseline = Sequential([
    Flatten(input_shape=(270, 270,3)),
    Dense(units = 16, activation = 'relu'),
    BatchNormalization(),
    Dropout(rate = 0.4),
    Dense(7, activation = 'softmax')
])
print(baseline.summary())
plot_model(baseline, to_file = "baseline2.png")

baselineCNN = Sequential(
    [
        Conv2D(16, (3, 3), activation='relu',input_shape=(270, 270, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size = 2, strides = 3),
        Flatten(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(rate = 0.4),
        Dense(7, activation='softmax')
    ]
)
print(baselineCNN.summary())
plot_model(baselineCNN,to_file = "simpleCNN2.png")

vggMini = Sequential(
    [
        Conv2D(16, (3, 3), activation='relu',input_shape=(270, 270, 3), padding = 'same'),
        Conv2D(16, (3, 3), activation='relu', padding = 'same'),
        MaxPooling2D(pool_size = 2),
        Conv2D(32, (3, 3), activation='relu', padding = 'same'),
        Conv2D(32, (3, 3), activation='relu', padding = 'same'),
        MaxPooling2D(pool_size = 2),
        Flatten(),
        Dense(16, activation = 'relu'),
        BatchNormalization(),
        Dropout(rate = 0.4),
        Dense(7, activation='softmax')
    ]
)

print(vggMini.summary())
plot_model(vggMini, to_file = "vggMini2.png")


inputLayer = Input(shape = (270,270,3))

block1column1 = Conv2D(filters = 32, kernel_size = 1, activation = 'relu', padding = 'same')(inputLayer)
block1column2 = Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(inputLayer)
block1column3 = Conv2D(filters = 32, kernel_size = 5, activation = 'relu', padding = 'same')(inputLayer)

outputLayerB1 = Concatenate()([block1column1,block1column2,block1column3])

block2column1 = Conv2D(filters = 16, kernel_size = 1, activation = 'relu', padding = 'same')(outputLayerB1)
block2column2 = Conv2D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(outputLayerB1)
block2column3 = Conv2D(filters = 16, kernel_size = 5, activation = 'relu', padding = 'same')(outputLayerB1)

outputLayerB2 = Concatenate()([block2column1,block2column2,block2column3])

maxPooling = MaxPooling2D(pool_size = 5)(outputLayerB2)

flattening = Flatten()(maxPooling)

dense1 = Dense(units = 16, activation = 'relu')(flattening)

batchNormalize = BatchNormalization()(dense1)

dropout = Dropout(rate = 0.4)(batchNormalize)

outputLayerFinal = Dense(units = 7, activation = 'softmax')(dropout)

googleLeNetMini = Model(inputs = inputLayer, outputs = outputLayerFinal)

print(googleLeNetMini.summary())
plot_model(googleLeNetMini, to_file = 'googLeNetMini2.png')

# %% Compile the models

# compile the model with a cross-entropy loss and specify the given optimizer
baseline.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-5),
    loss = CategoricalCrossentropy(),
    metrics=['accuracy', TopKCategoricalAccuracy(k = 2, name = 'Top 2 Accuracy')]
)

baselineCNN.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-5),
    loss = CategoricalCrossentropy(),
    metrics=['accuracy', TopKCategoricalAccuracy(k = 2, name = 'Top 2 Accuracy')]
)

vggMini.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-5),
    loss = CategoricalCrossentropy(),
    metrics=['accuracy', TopKCategoricalAccuracy(k = 2, name = 'Top 2 Accuracy')]
)

googleLeNetMini.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-5),
    loss = CategoricalCrossentropy(),
    metrics = ['accuracy', TopKCategoricalAccuracy(k = 2, name = 'Top 2 Accuracy')]
)


# %% Fit Baseline
baselineHistory = baseline.fit(trainGenerator,validation_data = valGenerator, epochs = 25, verbose = 1)

baselineResults = baseline.evaluate(testGenerator)
print("Baseline performance:", baselineResults)

pd.DataFrame(baselineHistory.history).to_csv('baselineResults.csv')

# %% Simple CNN

baselineCNNHistory = baselineCNN.fit(trainGenerator,validation_data = valGenerator, epochs = 25, verbose = 1)

simpleCNNResults = baselineCNN.evaluate(testGenerator)
print("Simple CNN performance:", simpleCNNResults)

pd.DataFrame(baselineCNNHistory.history).to_csv('baselineCNNResults.csv')

# %% Fit Mini VGG

vggMiniHistory = vggMini.fit(trainGenerator,validation_data = valGenerator, epochs = 25, verbose = 1)

vggMiniResults = vggMini.evaluate(testGenerator)
print("VGG Mini performance:", vggMiniResults)

pd.DataFrame(vggMiniHistory.history).to_csv('vggMiniResults.csv')


# %% Fit Mini GoogLeNet

googleLeNetMiniHistory = googleLeNetMini.fit(trainGenerator,validation_data = valGenerator, epochs = 25, verbose = 1)

googLeNetMiniResults = googleLeNetMini.evaluate(testGenerator)
print("GoogLeNet performance:", googLeNetMiniResults)

pd.DataFrame(googleLeNetMiniHistory.history).to_csv('googLeNetMiniResults.csv')

