import numpy as np
import signaldoctorlib as sdl
import os
import matplotlib.pyplot as plt

## Takes training IQ data and generates training features

def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

def feature_gen(file_list, spec_size):
    output_list_spec = [] ## This is going to be very big!
    output_list_psd = []
    output_list_phi = []
    output_list_cec = []
    
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        #print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        #print("We have %f s of data" % (len(iq_data)/fs))
        print(filename)

        Zxx_mag_rs, Zxx_phi_rs, Zxx_cec_rs, PSD = sdl.generate_features(fs, iq_data, spec_size, False)
        output_list_spec.append(Zxx_mag_rs)
        output_list_psd.append(PSD)
        output_list_phi.append(Zxx_phi_rs)
        output_list_cec.append(Zxx_cec_rs)
        #plt.pcolormesh(Zxx_dat)
        #plt.show()
    return [output_list_spec, output_list_phi, output_list_cec, output_list_psd]

input_folder = "/home/jonathan/TF_Train_1_0"

for sig in os.listdir(input_folder):
    sig_input_folder = input_folder + "/" + sig
    if os.path.isdir(sig_input_folder):
        print("Processing from --->>>>>>", sig_input_folder)
        filename_list = []
        for fileZ in os.listdir(sig_input_folder):
            #print(fileZ)
            if fileZ.endswith(".npz"):
                filename_list.append(os.path.join(sig_input_folder, fileZ))

        data_list = feature_gen(filename_list, 256)

        spec_aray = np.asarray(data_list[0])
        phi_array = np.asarray(data_list[1])
        cec_array = np.asarray(data_list[2])
        psd_aray = np.asarray(data_list[3])
        np.save('nnetsetup/specdata/' + sig + '.npy', spec_aray)
        np.save('nnetsetup/psddata/' + sig + '.npy', psd_aray)
        np.save('nnetsetup/phidata/' + sig + '.npy', phi_array)
        np.save('nnetsetup/cecdata/' + sig + '.npy', cec_array)



## Splits training data into training and test data

## $$$$$$$ SPECTROGRAM $$$$$$$$

train_testsplit = 0.8

input_folder = "nnetsetup/specdata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/spec_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()

y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

np.savez("nnetsetup/SpecTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)




## $$$$$$$ Phi $$$$$$$$

train_testsplit = 0.8

input_folder = "nnetsetup/phidata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/phi_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()

y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

np.savez("nnetsetup/PhiTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


## $$$$$$$ CEC $$$$$$$$

train_testsplit = 0.8

input_folder = "nnetsetup/cecdata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/cec_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()

y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

np.savez("nnetsetup/CecTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)



## $$$$$$$ PSD $$$$$$$$

train_testsplit = 0.8 ##

input_folder = "nnetsetup/psddata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/psd_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()


y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

np.savez("nnetsetup/PsdTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
