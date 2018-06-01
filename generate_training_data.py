import numpy as np
import signaldoctorlib as sdl
import os, shutil, configparser
import glob, random
import matplotlib.pyplot as plt

## Takes training IQ data and generates training features

SPEC_SIZE = 256 ## NxN input tensor size

ADD_RANDOM_NOISE_LEVEL = True

#input_folder = "/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/HF_Reduced/HF_Dataset"
#input_folder = "/home/jonathan/HF_Dataset"
input_folder = "/mnt/datastore/FYP/training_sets/HF_SetV4_NOISE"


LOAD_FROM_DATASTORE = True

def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

def awgn(iq_data, snr):
    no_samples = iq_data.shape[0]
    abs_iq = np.abs(iq_data*iq_data.conj())
    signal_power = np.sum(abs_iq)/no_samples
    k = signal_power * 10**(-snr/20)
    noise_output = np.empty((no_samples), dtype=np.complex)
    noise_output.real = np.random.normal(0,1,no_samples) * np.sqrt(k)
    noise_output.imag = np.random.normal(0,1,no_samples) * np.sqrt(k)
    return iq_data+noise_output

def feature_gen(file_list, spec_size, config = None):
    output_list_mean = []
    output_list_max = []
    output_list_min = []
    output_list_var = []
    
    output_list_spec = [] ## This is going to be very big!
    output_list_cec = []
    output_list_fft = []
    
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        #print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        #print("We have %f s of data" % (len(iq_data)/fs))
        print("Reading-->>",filename)
        if ADD_RANDOM_NOISE_LEVEL:
            snr = random.triangular(-20, 20)
            iq_data = awgn(iq_data, snr)
            print("Applied SNR value of %f dB"%snr)

        feature_dict = sdl.generate_features(fs, iq_data, spec_size, plot_features = False, config = config)
        
        #tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
        output_list_mean.append(feature_dict['psd'])
        output_list_max.append(feature_dict['max_spectrum'])
        output_list_min.append(feature_dict['min_spectrum'])
        output_list_var.append(feature_dict['variencespectrum'])
        
        output_list_spec.append(feature_dict['magnitude'])
        output_list_cec.append(feature_dict['corrcoef'])
        output_list_fft.append(feature_dict['fft_spectrum'])
        
        #print(feature_dict['psd'].shape)
        #print(feature_dict['variencespectrum'].shape)
        #print(feature_dict['differentialspectrumdensity'].shape)
        #print(feature_dict['min_spectrum'].shape)
        #print(feature_diect['min_spectrum'].shape)
        
        #tmp_psd = np.stack((feature_dict['psd'], feature_dict['variencespectrum'], feature_dict['differentialspectrumdensity'], feature_dict['min_spectrum'], feature_dict['min_spectrum']), axis=-1)
        

        #plt.pcolormesh(feature_dict['magnitude'])
        #plt.show()
    return [output_list_mean, output_list_max, output_list_min, output_list_var, output_list_spec, output_list_cec, output_list_fft]


if LOAD_FROM_DATASTORE:
    config = configparser.ConfigParser()
    config.read('sdl_config.ini')

    feature_list = ['meandata', 'maxdata', 'mindata', 'vardata', 'specdata', 'cecdata', 'fftdata']
    folder = 'nnetsetup/'

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    for i in feature_list:
        os.mkdir(folder+i)


    for sig in os.listdir(input_folder):
        sig_input_folder = input_folder + "/" + sig
        if os.path.isdir(sig_input_folder):
            print("Processing from --->>>>>>", sig_input_folder)
            filename_list = []
            for fileZ in os.listdir(sig_input_folder):
                #print(fileZ)
                if fileZ.endswith(".npz"):
                    filename_list.append(os.path.join(sig_input_folder, fileZ))

            data_list = feature_gen(filename_list, SPEC_SIZE, config = config)

            np.save('nnetsetup/meandata/' + sig + '.npy', np.asarray(data_list[0]))
            np.save('nnetsetup/maxdata/' + sig + '.npy', np.asarray(data_list[1]))
            np.save('nnetsetup/mindata/' + sig + '.npy', np.asarray(data_list[2]))
            np.save('nnetsetup/vardata/' + sig + '.npy', np.asarray(data_list[3]))
            
            np.save('nnetsetup/specdata/' + sig + '.npy', np.asarray(data_list[4]))
            np.save('nnetsetup/cecdata/' + sig + '.npy', np.asarray(data_list[5]))
            np.save('nnetsetup/fftdata/' + sig + '.npy', np.asarray(data_list[6]))
        



def package_data_1D(input_folder, prefix):
    ## $$$$$$$ PSD $$$$$$$$
    print("~@~@~@~@~@ Processing", prefix, "at", input_folder, "@~@~@~@~@~@~@~")
    train_testsplit = 0.8 ##

    #input_folder = "nnetsetup/psddata"

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

    thefile = open('nnetsetup/'+prefix+'.csv', 'w')
    for i in index_data:
        thefile.write("%s\n" % i)
    thefile.close()


    y_train = np.asarray(y_train)

    X_train = np.asarray(X_train)
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    y_test = np.asarray(y_test)

    X_test = np.asarray(X_test)
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    np.savez('nnetsetup/'+prefix+'.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("########", prefix, "########")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)


def package_data_2D(input_folder, prefix):

    ## Splits training data into training and test data

    ## $$$$$$$ SPECTROGRAM CNN $$$$$$$$

    train_testsplit = 0.8

    #input_folder = "nnetsetup/specdata"

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

    thefile = open('nnetsetup/'+prefix+'.csv', 'w')
    for i in index_data:
        thefile.write("%s\n" % i)
    thefile.close()

    y_train = np.asarray(y_train)

    X_train = np.asarray(X_train)
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    y_test = np.asarray(y_test)

    X_test = np.asarray(X_test)
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))

    np.savez('nnetsetup/'+prefix+'.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("########", prefix, "########")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

package_data_1D("nnetsetup/meandata", "MeanPSDTrainingData")
package_data_1D("nnetsetup/maxdata", "MaxPSDTrainingData")
package_data_1D("nnetsetup/mindata", "MinPSDTrainingData")
package_data_1D("nnetsetup/vardata", "VarTrainingData")

package_data_1D("nnetsetup/specdata", "MagSpecTrainingData")
package_data_1D("nnetsetup/cecdata", "CecTrainingData")
package_data_1D("nnetsetup/fftdata", "FFTTrainingData")
