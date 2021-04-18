
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import utils
# Build a simple dense model with early stopping and softmax for categorical classification, remember we have 30 classes


class Model(keras.models.Sequential):
    def __init__(self, inputs: tuple, num_labels: int):
        super(Model, self).__init__()
        self.add(Dense(inputs[0], input_shape=inputs, activation = 'relu'))
        self.add(Dropout(0.1))
        self.add(Dense(128, activation = 'relu'))
        self.add(Dropout(0.25))
        self.add(Dense(128, activation = 'relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_labels, activation = 'softmax'))
        self.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


model = Model((193,), num_labels=31)

print(model.summary())

train_data, train_target = utils.load_train_data()
dev_data, dev_target = utils.load_dev_data()

# Hot encoding y
lb = LabelEncoder()
train_target = to_categorical(lb.fit_transform(train_target))
dev_target = to_categorical(lb.fit_transform(dev_target))

ss = StandardScaler()
train_data = ss.fit_transform(train_data)
dev_data = ss.transform(dev_data)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
mcp_save = ModelCheckpoint('model/voice/speechnet.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(
    train_data, train_target,
    batch_size=32, epochs=200,
    validation_data=(dev_data, dev_target),
    callbacks=[early_stop, mcp_save]
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
plt.xticks(range(0,100,5), range(0,100,5))
plt.legend(fontsize = 18);
plt.savefig("training.png")
