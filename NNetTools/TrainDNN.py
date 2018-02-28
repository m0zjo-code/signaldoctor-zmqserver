'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *

batch_size = 64
#num_classes = 10
epochs = 10000

# the data, shuffled and split between train and test sets

input_data = np.load("PsdTrainingData.npz")
x_train = input_data['X_train']
y_train = input_data['y_train']
x_test = input_data['X_test']
y_test = input_data['y_test']

num_classes = len(np.unique(y_train))
print("No. Classes:", num_classes)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(2**12, activation='tanh', input_shape=(256,)))
model.add(Dropout(0.2))
model.add(Dense(2**7, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2**7, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2**7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2**6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2**5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5, verbose=0, mode='auto')


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adamax(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# serialize model to JSON
model_json = model.to_json()
with open("psdmodel.nn", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("psdmodel.h5")
print("Saved model to disk")
