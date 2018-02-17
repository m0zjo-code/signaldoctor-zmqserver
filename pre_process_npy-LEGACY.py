import numpy as np
import signaldoctorlib as sdl
import os
import matplotlib.pyplot as plt

## Takes training IQ data and generates training features


data_output_folder_spec = "specdata"
data_output_folder_psd = "psddata"

data_lib = [data_output_folder_spec, data_output_folder_psd]

def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

def feature_gen(file_list, spec_size):
    output_list_spec = [] ## This is going to be very big!
    output_list_psd = []
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        print("We have %f s of data" % (len(iq_data)/fs))

        Zxx_dat, PSD = sdl.generate_features(fs, iq_data, spec_size, False)
        output_list_spec.append(Zxx_dat)
        output_list_psd.append(PSD)
        #plt.pcolormesh(Zxx_dat)
        #plt.show()
    return output_list_spec, output_list_psd


input_folder = "/mnt/datastore/FYP/training_sets/TF_Train_V3/iq"
filename_list = []

for sig in os.listdir(input_folder):
    sig_input_folder = input_folder + "/" + sig
    if os.path.isdir(sig_input_folder):
        print(sig_input_folder)
        for fileZ in os.listdir(sig_input_folder):
            print(fileZ)
            if fileZ.endswith(".npz"):
                filename_list.append(os.path.join(sig_input_folder, fileZ))
    
        spec_list, psd_list = feature_gen(filename_list, 256)
        
        spec_aray = np.asarray(spec_list)
        psd_aray = np.asarray(psd_list)
        
        np.save(data_output_folder_spec + "/" + sig + '.npy', spec_aray)
        np.save(data_output_folder_psd + "/" + sig + '.npy', psd_aray)


## Splits training data into training and test data

##### SPECTROGRAM ########

input_folder = data_output_folder_spec

train_testsplit = 0.6 ##

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)
    
    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)
    
    index = index + 1

y_train = np.asarray(y_train)


X_train = np.asarray(X_train)
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    
np.savez("%sLib.npz" % (input_folder), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


##### PSD ########

input_folder = data_output_folder_psd

train_testsplit = 0.6 ##

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)
    
    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)
    
    index = index + 1

y_train = np.asarray(y_train)


X_train = np.asarray(X_train)
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
    
np.savez("%sLib.npz" % (input_folder), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
