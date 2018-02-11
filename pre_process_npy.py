import numpy as np
import os

train_testsplit = 0.6 ##

input_folder = "specdata"

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
