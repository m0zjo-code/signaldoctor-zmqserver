from keras.models import model_from_json
import tensorflow as tf
import csv
import numpy as np


def get_spec_model(modelname=None, indexname=None):
    """ 
    Load models from file 
    Returns model and indexes 
    """
    
    json_file = open('%s.nn'%(modelname), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5"%(modelname))
    print("Loaded -->> %s model from disk" % modelname)
    ## https://stackoverflow.com/questions/6740918/creating-a-dictionary-from-a-csv-file
    with open('%s.csv'%indexname, mode='r') as ifile:
        reader = csv.reader(ifile)
        d = {}
        for row in reader:
            k, v = row
            d[int(v)] = k
    
    return loaded_model, d


def classify_buffer1d(feature_dict, loaded_model):
    
    tmp_psd = np.stack((feature_dict['psd'], feature_dict['variencespectrum'], feature_dict['differentialspectrumdensity'], feature_dict['min_spectrum'], feature_dict['min_spectrum']), axis=-1)
    tmp_psd = np.reshape(tmp_psd, (1, tmp_psd.shape[0],  tmp_psd.shape[1]))
    #print("Data Shape:", tmp_psd.shape)
    prediction = loaded_model.predict(tmp_psd)[0]
    #prediction = prediction.flat[0]
    #print(prediction)
    prediction = np.argmax(prediction, axis=-1)
    #print(index_dict[prediction])
    
    return prediction


def classify_buffer2d(feature_dict, loaded_model, spec_size = 256):
    
    tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
    tmp_spec = np.reshape(tmp_spec, (1, tmp_spec.shape[0],  tmp_spec.shape[1], tmp_spec.shape[2]))
    #print("Data Shape:", tmp_spec.shape)
    prediction = loaded_model.predict(tmp_spec)[0]
    #print(prediction)
    prediction = np.argmax(prediction, axis=-1)
    #print(index_dict[prediction])
    
    return prediction


