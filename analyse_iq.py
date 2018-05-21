import os
import numpy as np
import math, configparser
import matplotlib.pyplot as plt
import signaldoctorlib as sdl

load_dir = 'MATLAB/sample_logs/'

iq_file = ['0SA1QT', '4Y0NXH', 'AITEIJ', 'LMZ1V1']
sig = ['CW', 'Static Carrier', 'SSB', 'AM']
config = configparser.ConfigParser()
config.read('sdl_config.ini')

for i in range(0, 4):
    data = np.load(load_dir+iq_file[i]+'.npz')
    #print(data.keys())
    channel_iq = data['channel_iq']
    fs = data['fs']
    iq_len = len(channel_iq)
    print("IQ Len: ", iq_len, iq_len/fs, "s")
    feature_list = sdl.generate_features(fs,  channel_iq, config=config)

    ##### FYI::
        ###output_dict = {}
        ###output_dict['magnitude'] = Zxx_mag_rs
        ###output_dict['phase'] = Zxx_phi_rs
        ###output_dict['corrcoef'] = Zxx_cec_rs
        ###output_dict['cov'] = Zxx_cov_rs
        ###output_dict['psd'] = PSD
        ###output_dict['variencespectrum'] = Varience_Spectrum
        ###output_dict['differentialspectrumdensity'] = Differential_Spectrum
        ###output_dict['differentialspectrum_freq'] = diff_array0_rs
        ###output_dict['differentialspectrum_time'] = diff_array1_rs
        ###output_dict['min_spectrum'] = Min_Spectrum
        ###output_dict['max_spectrum'] = Max_Spectrum
        ###output_dict['fft_spectrum'] = normalise_spectrogram(Zxx_mag_fft, spec_size, spec_size)
        ###output_dict['hilb_spectrum'] = normalise_spectrogram(Zxx_mag_hilb, spec_size, spec_size)
    #####
    plt.subplot(2, 2, i+1)
    plt.title('Magnitude Spectrogram - '+sig[i])
    plt.pcolormesh(np.linspace(0, iq_len/fs, 256), np.linspace(-fs/2, fs/2, 256), feature_list['magnitude'])
    plt.xlabel('Time/s')
    plt.ylabel('Frequency/Hz')

plt.tight_layout()
plt.show()


