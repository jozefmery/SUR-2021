import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
import utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


class Model(keras.models.Sequential):
    def __init__(self, inputs: tuple, num_labels: int):
        super(Model, self).__init__()
        self.add(Conv2D(32, (3, 3), padding='same', input_shape=inputs))
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))
        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.5))
        self.add(Conv2D(128, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(Conv2D(128, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_labels, activation='softmax'))
        self.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])


model = Model((64,64,3), 31)

print(model.summary())

datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_directory(
    directory="data\\voice\\spectogram\\train\\",
    batch_size=32, seed=42, shuffle=True,
    class_mode="categorical",
    target_size=(64,64)
)

valid_generator=datagen.flow_from_directory(
    directory="data\\voice\\spectogram\\dev\\",
    batch_size=32, seed=42, shuffle=True,
    class_mode="categorical",
    target_size=(64,64)
)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=200
)

# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Set figure size.
plt.figure(figsize=(12, 8))
# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
# Set title
plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,200,5), range(0,200,5))
plt.legend(fontsize = 18);
plt.savefig("training.png")
