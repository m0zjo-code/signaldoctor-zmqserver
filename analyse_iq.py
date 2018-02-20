import os
import numpy as np
import math
import matplotlib.pyplot as plt
import signaldoctorlib as sdl

load_dir = 'logs/'

items = os.listdir(load_dir)

iqlist = []
for names in items:
    if names.endswith(".npz"):
        iqlist.append(names)
print(iqlist)

def savefig_mat(mat, fname):
    plt.pcolormesh(mat)
    #plt.show()
    plt.savefig(fname+".jpg", dpi=100, bbox_inches='tight')

for iq_file in iqlist:
    data = np.load(load_dir+iq_file)
    #print(data.keys())
    channel_iq = data['channel_iq']
    fs = data['fs']
    iq_len = len(channel_iq)
    print("IQ Len: ", iq_len, iq_len/fs, "s")
    feature_list = sdl.generate_features(fs,  channel_iq)
    savefig_mat(feature_list[0], load_dir+iq_file)
