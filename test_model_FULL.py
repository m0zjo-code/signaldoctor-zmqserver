from keras.models import model_from_json
from keras.models import load_model

from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import configparser

import signaldoctorlib as sdl
import signaldoctorlib_class as sdlc
import os, shutil, argparse
import glob, time
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read('sdl_config.ini')

#parser = argparse.ArgumentParser(description='Train a CNN')
#parser.add_argument('--weights', help='Location of Network Weights', action="store", dest="nweights")
#parser.add_argument('--arch', help='Location of Network Arch File', action="store", dest="narch")
#parser.add_argument('--test', help='Location of Network Test Set', action="store", dest="testloc")
#parser.add_argument('--analysis', help='Location of Analysis Files', action="store", dest="analysisfiles")
#parser.add_argument('--mode', help='Feature Generation Mode', action="store", dest="mode")
#args = parser.parse_args()

import sys

#network_definition_location = "/home/jonathan/SIGNAL_CNN_TRAIN_KERAS/MAGNOISE31052018_Adadelta_4_1_1527790557.nn"
#network_weights_location = "/home/jonathan/SIGNAL_CNN_TRAIN_KERAS/MAGNOISE31052018_Adadelta_4_1_1527790557.h5"

#data_test_set = "/mnt/datastore/FYP/training_sets/training31052018/MagSpecTrainingData.npz"

network_prefix = "/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/prototype_networks/"
network_prefix = "/mnt/datastore/FYP/prototypenetworks/"



network_definition_location = [
    network_prefix+"CecSpec_Adadelta_4_1_1527900104",
    network_prefix+"FFTSpec_Adadelta_4_1_1527917966",
    network_prefix+"MagSpec_Adadelta_4_1_1527866143",
    network_prefix+"MaxPSD_Adamax_1_2_1527865830",
    network_prefix+"MeanPSD_Adamax_1_2_1527865687",
    network_prefix+"MinPSD_Adamax_1_2_1527865966",
    network_prefix+"VarPSD_Adamax_1_2_1527866038"
    ]

class_dec = network_prefix+"MagSpecTrainingData"


network_list = []
for name in network_definition_location:
    network_list.append(sdlc.get_spec_model(modelname=name, indexname=class_dec))

print(network_list[0][1])

def norm_data(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))

def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

### Now to generate 
input_folder = "/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/HF_Reduced/HF_Dataset"
input_folder = "/mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced"

feature_gen_modes = [
    "corrcoef",
    "fft_spectrum",
    "magnitude",
    "max_spectrum",
    "psd",
    "min_spectrum",
    "variencespectrum"
    ]
    
num_classes = 6

filename_prefix = str(int(time.time())) + "_" + "FULL"

def awgn(iq_data, snr):
    no_samples = iq_data.shape[0]
    
    #print(no_samples)
    
    abs_iq = np.abs(iq_data*iq_data.conj())
    signal_power = np.sum(abs_iq)/no_samples
    k = signal_power * 10**(-snr/20)
    noise_output = np.empty((no_samples), dtype=np.complex)
    noise_output.real = 2 * np.random.normal(0,1,no_samples) * np.sqrt(k)
    noise_output.imag = 2 * np.random.normal(0,1,no_samples) * np.sqrt(k)
    return iq_data+noise_output

def feature_gen(file_list, spec_size, config = None, snr = None, mode = None):
    output_list = []
    
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        #print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        #print("We have %f s of data" % (len(iq_data)/fs))
        #print("Reading-->>",filename)
        
        ## Do noise addition here!
        iq_data = awgn(iq_data, snr)
        
        
        feature_dict = sdl.generate_features(fs, iq_data, spec_size, plot_features = False, config = config)
        
        #tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
        output_list.append(feature_dict[mode])
        
        #plt.pcolormesh(feature_dict['magnitude'])
        #plt.show()
    return output_list

with open('results_%s.log'%filename_prefix, "a") as f:
    f.write("%s, %s, %s\n"%("Class", "SNR", "Accuracy"))


for snr in range(10, 20+1):
    full_score = [[],[]]
    
    for sig in os.listdir(input_folder):
        sig_input_folder = input_folder + "/" + sig
        
        if os.path.isdir(sig_input_folder):
            print("Processing from --->>>>>>", sig_input_folder, sig, "SNR:", snr)
            filename_list = []
            for fileZ in os.listdir(sig_input_folder):
                #print(fileZ)
                if fileZ.endswith(".npz"):
                    filename_list.append(os.path.join(sig_input_folder, fileZ))
            SPEC_SIZE = 256
            
            
            score_output = []
            for i in range(0, len(feature_gen_modes)):
                data_list = feature_gen(filename_list, SPEC_SIZE, config = config, snr = snr, mode = feature_gen_modes[i])
                data_list = np.asarray(data_list)
            
                #run Classification
                data_list = data_list.astype('float32')
                data_list = norm_data(data_list)
                data_list = np.expand_dims(data_list, axis=-1)
                
                #scores = network_list[i][0].evaluate(data_list, Y_test_SNR, verbose=1)  # Evaluate the trained model on the test set!
                #print("Mode:", modes[i], "SNR:", snr)
                #print("%s: %.2f%%" % (network_list[i][0].metrics_names[1], scores[1]*100))
                network_prediction = network_list[i][0].predict(data_list, verbose=1)
                score_output.append(network_prediction)
                
            score_output = np.asarray(score_output)
            
            # If we want to weight, do it here
            # weightings = np.array([0, 1, 2, 3, 4, 5, 6])
            # score_output = np.transpose(score_output.transpose()*weightings)
            
            # Initially we sum the array across the feature axis to get an aggregated sum
            # For an equally weighted set, the max a result can get is 6
            score_output = np.sum(score_output, axis = 0)
            
            # Get the max value argument across the signal sample axis
            aggregated_output = np.argmax(score_output, axis=1)
            print(aggregated_output)
            
            # Generate correct class vector
            y_test_SNR = np.ones((data_list.shape[0])) * network_list[i][1][sig]
            #Y_test_SNR = np_utils.to_categorical(y_test_SNR, num_classes)
            
            score = accuracy_score(y_test_SNR, aggregated_output)
            
            full_score[0].append(y_test_SNR)
            full_score[1].append(aggregated_output)
            
            print("Score, ", score)
            
            with open('results_%s.log'%filename_prefix, "a") as f:
                f.write("%s, %f, %f\n"%(sig, snr, score*100))
                
    # Process whole set of classes to get conf_matrix
    y_true = np.asarray(full_score[0]).flatten()
    y_pred = np.asarray(full_score[1]).flatten()
    
    score = accuracy_score(y_true, y_pred)
    print("Score, ", score)
    conf_matx = confusion_matrix(y_true, y_pred)
    print(conf_matx)
    
    with open('%s_acc.log'%filename_prefix, "a") as f:
        f.write("acc\n")
        f.write("%.4f%%" % (score*100))
        f.write('\n')
        f.write(str(conf_matx))



