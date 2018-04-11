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

batch_size = 128
#num_classes = 10
epochs = 1000

# the data, shuffled and split between train and test sets

input_data = np.load("nnetsetup/PsdTrainingData.npz")
x_train = input_data['X_train']
y_train = input_data['y_train']
x_test = input_data['X_test']
y_test = input_data['y_test']


from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=0)


num_classes = len(np.unique(y_train))
print("No. Classes:", num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train) # Normalise data to [0, 1] range
x_test /= np.max(x_test) # Normalise data to [0, 1] range

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(2**3, activation='tanh', input_shape=(256,)))
model.add(Dropout(0.2))
model.add(Dense(2**3, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(2**3, activation='tanh'))
#model.add(Dense(2**7, activation='tanh'))
#model.add(Dense(2**7, activation='tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(2**6, activation='tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(2**5, activation='tanh'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=50, verbose=1, mode='auto')
callbacks_list = [earlystop]


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    shuffle = True,
                    callbacks=callbacks_list,
                    validation_split=0.2)


score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# serialize model to JSON
model_json = model.to_json()
with open("psdmodel.nn", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("psdmodel.h5")
print("Saved model to disk")
print("Training Complete")
