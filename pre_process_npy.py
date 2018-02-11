import numpy as np

train_testsplit = 0.6 ##

input_folder = "specdata"

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, file))

for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    numpy.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
