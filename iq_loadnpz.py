"""
Jonathan Rawlinson 2018
"""

import numpy as np
import signaldoctorlib as sdl
import os
import matplotlib.pyplot as plt


def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

def feature_gen(file_list, spec_size):
    output_list = [] ## This is going to be very big!
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        print("We have %f s of data" % (len(iq_data)/fs))

        Zxx_dat = sdl.generate_features(fs, iq_data, 256, False)
        output_list.append(Zxx_dat)
        #plt.pcolormesh(Zxx_dat)
        #plt.show()
    return output_list


input_folder = "exampleIQ"
filename_list = []

for file in os.listdir("exampleIQ"):
    if file.endswith(".npz"):
        filename_list.append(os.path.join("exampleIQ", file))
        
filename_list = ["exampleIQ/tankcontroller33.npz","exampleIQ/tankcontroller33.npz","exampleIQ/tankcontroller33.npz","exampleIQ/tankcontroller33.npz"]

spec_list = feature_gen(filename_list, 256)
spec_aray = np.asarray(spec_list)
np.save('SpecArray.npy', spec_aray)
