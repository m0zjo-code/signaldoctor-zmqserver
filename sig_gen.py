import numpy as np
import scipy
import configparser
import signaldoctorlib as sdl

config = configparser.ConfigParser()
config.read('sdl_config.ini')

#output_folder_noise = "/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/NOISE/"
#output_folder_carrier = "/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/CARRIER/"

output_folder_noise = "/mnt/datastore/FYP/training_sets/HF_SetV4_NOISE/NOISE/"
output_folder_carrier = "/mnt/datastore/FYP/training_sets/HF_SetV4_NOISE/CARRIER/"

no_sigs = 1500

for i in range(0, no_sigs):
    noise_vec = np.empty((8192), dtype = np.complex)
    noise_vec.real = np.random.normal(0, 1, 8192)
    noise_vec.imag = np.random.normal(0, 1, 8192)

    spec_size = 256
    feature_dict = sdl.generate_features(1, noise_vec, spec_size, plot_features = False, config = config)
    feature_dict['iq_data'] = noise_vec
    feature_dict['local_fs'] = 1
    sdl.save_IQ_buffer(feature_dict, output_folder = output_folder_noise,  config = config)
    print("Generated NOISE")


for i in range(0, no_sigs):
    noise_vec = np.empty((8192), dtype = np.complex)
    noise_vec.real = np.random.normal(1, 1, 8192)
    noise_vec.imag = np.random.normal(1, 1, 8192)
    
    buffer_fft = sdl.fft_wrap(noise_vec, mode = 'scipy')
    buffer_fft_rolled = np.roll(buffer_fft, int(len(buffer_fft)/2))
    noise_vec = sdl.ifft_wrap(buffer_fft_rolled, mode = 'scipy')

    spec_size = 256
    feature_dict = sdl.generate_features(1, noise_vec, spec_size, plot_features = False, config = config)
    feature_dict['iq_data'] = noise_vec
    feature_dict['local_fs'] = 1
    sdl.save_IQ_buffer(feature_dict, output_folder = output_folder_carrier,  config = config)
    print("Generated CARRIER")
