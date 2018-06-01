from keras.models import model_from_json
from keras.models import load_model

from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from sklearn.metrics import confusion_matrix
import numpy as np
import configparser

import signaldoctorlib as sdl
import os, shutil
import glob, time
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read('sdl_config.ini')

import sys

network_definition_location = "/home/jonathan/SIGNAL_CNN_TRAIN_KERAS/MAGNOISE31052018_Adadelta_4_1_1527790557.nn"
network_weights_location = "/home/jonathan/SIGNAL_CNN_TRAIN_KERAS/MAGNOISE31052018_Adadelta_4_1_1527790557.h5"

data_test_set = "/mnt/datastore/FYP/training_sets/training31052018/MagSpecTrainingData.npz"

def norm_data(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))


input_data = np.load(data_test_set)

X_test = input_data['X_test']
y_test = input_data['y_test']


try:
    num_train, width, depth = X_test.shape # there are 50000 training examples in CIFAR-10
except ValueError:
    depth = 1
    num_train, width = X_test.shape
    print("Single Channel Detected")
    

num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_test).shape[0] # there are 10 image classes

X_test = X_test.astype('float32')

X_test = norm_data(X_test)

Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
X_test = np.expand_dims(X_test, axis=-1)


def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

try:
    loaded_model = load_model(network_weights_location)
    print("Loaded model from disk - V2 Single File H5 Model")
except ValueError:
    json_file = open(network_definition_location, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(network_weights_location)
    print("Loaded model from disk - V1 JSON Model")
    loaded_model.compile(loss='categorical_crossentropy',
        optimizer='Adamax',
        metrics=['accuracy'])

scores = loaded_model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

Y_predict = loaded_model.predict(X_test)
conf_matx = confusion_matrix(Y_test.argmax(axis=1), Y_predict.argmax(axis=1))
print(conf_matx)
print(type(loaded_model))
print(loaded_model.metrics_names)
print(scores)

### Now to generate 
input_folder = "/mnt/datastore/FYP/training_sets/reducedset31052018"


class_index = {}
class_index['CARRIER'] = 0
class_index['SSB'] = 1
class_index['AM'] = 2
class_index['FSK'] = 3
class_index['CW'] = 4

filename_prefix = str(int(time.time())) + "_MAG"

def awgn(iq_data, snr):
    no_samples = iq_data.shape[0]
    
    print(no_samples)
    
    abs_iq = np.abs(iq_data*iq_data.conj())
    signal_power = np.sum(abs_iq)/no_samples
    k = signal_power * 10**(-snr/20)
    noise_output = np.empty((no_samples), dtype=np.complex)
    noise_output.real = np.random.normal(0,1,no_samples) * np.sqrt(k)
    noise_output.imag = np.random.normal(0,1,no_samples) * np.sqrt(k)
    return iq_data+noise_output

def feature_gen(file_list, spec_size, config = None, snr = None):
    output_list = []
    
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        #print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        #print("We have %f s of data" % (len(iq_data)/fs))
        print("Reading-->>",filename)
        
        ## Do noise addition here!
        iq_data = awgn(iq_data, snr)

        feature_dict = sdl.generate_features(fs, iq_data, spec_size, plot_features = False, config = config)
        
        #tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
        output_list.append(feature_dict['magnitude'])
        
        #plt.pcolormesh(feature_dict['magnitude'])
        #plt.show()
    return output_list

with open('results_%s.log'%filename_prefix, "a") as f:
    f.write("%s, %s, %s\n"%("Class", "SNR", "Accuracy"))


for snr in range(-20, 20+1):
    for sig in os.listdir(input_folder):
        sig_input_folder = input_folder + "/" + sig
        if os.path.isdir(sig_input_folder):
            print("Processing from --->>>>>>", sig_input_folder, sig)
            filename_list = []
            for fileZ in os.listdir(sig_input_folder):
                #print(fileZ)
                if fileZ.endswith(".npz"):
                    filename_list.append(os.path.join(sig_input_folder, fileZ))
            SPEC_SIZE = 256
            
            
            data_list = feature_gen(filename_list, SPEC_SIZE, config = config, snr = snr)
            
            data_list = np.asarray(data_list)
            #run Classification
            y_test_SNR = np.ones((data_list.shape[0])) * class_index[sig]
            Y_test_SNR = np_utils.to_categorical(y_test_SNR, num_classes)
            data_list = data_list.astype('float32')
            data_list = norm_data(data_list)
            data_list = np.expand_dims(data_list, axis=-1)
            
            scores = loaded_model.evaluate(data_list, Y_test_SNR, verbose=1)  # Evaluate the trained model on the test set!

            print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

            #Y_predict = loaded_model.predict(data_list)
            #conf_matx = confusion_matrix(Y_test_SNR.argmax(axis=1), Y_predict.argmax(axis=1))
            #print(conf_matx)
            #print(type(loaded_model))
            #print(loaded_model.metrics_names)
            print(scores)
            with open('results_%s.log'%filename_prefix, "a") as f:
                f.write("%s, %f, %f\n"%(sig, snr, scores[1]*100))




