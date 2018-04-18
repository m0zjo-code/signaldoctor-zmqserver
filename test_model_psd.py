from keras.models import model_from_json
import numpy as np
import signaldoctorlib as sdl

json_file = open('psdmodel.nn', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("psdmodel.h5")
print("Loaded model from disk")

loaded_model.summary()

input_file = "/home/jonathan/3EI0V2.npz"

def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

print("Loaded data file")
fs, iq_data = load_npz(input_file)

feature_dict = sdl.generate_features(fs, iq_data, 299, plot_features = False)
#tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
tmp_psd = np.stack((feature_dict['psd'], feature_dict['variencespectrum'], feature_dict['differentialspectrumdensity'], feature_dict['min_spectrum'], feature_dict['min_spectrum']), axis=-1)

tmp_psd = np.reshape(tmp_psd, (1, tmp_psd.shape[0],  tmp_psd.shape[1]))

print("Data Shape:", tmp_psd.shape)
prediction = loaded_model.predict(tmp_psd)
#prediction = prediction.flat[0]
print(prediction)

