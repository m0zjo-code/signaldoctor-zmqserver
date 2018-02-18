import sys
import zmq
import numpy as np
import sys
port = "5555"

import signaldoctorlib as sdl

MODEL_NAME = "spec_model"

from keras.models import model_from_json


def classify_spectrogram(input_array, model):
    
    print("Classify Spec")
    print("Data Shape:", input_array.shape)
    
    print(input_array)
    input_array = input_array.reshape((1, input_array.shape[0], input_array.shape[1], 1))
    
    prediction = loaded_model.predict(input_array)
    
    print(prediction)
    
    
    prediction = prediction.flat[0]
    


if __name__ == "__main__":
    
    
    print("Running: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    
    if  (sys.argv[0] == "-i"):
        print("Reading IQ")
        sys.exit(1)
    
    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("Collecting updates from IQ server...")
    socket.connect ("tcp://127.0.0.1:5555")

    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    
    ## LOAD SPECTROGRAM NETWORK ##
    json_file = open('%s.nn'%(MODEL_NAME), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5"%(MODEL_NAME))
    print("Loaded model from disk")
    
    while True:
        string = socket.recv()
        buffer_data = np.fromstring(string, dtype = 'complex64')
        extracted_features = sdl.process_buffer(buffer_data)
        
        
        print(type(loaded_model))
        
        spec = np.asarray(extracted_features[0][0])
        
        classify_spectrogram(spec, loaded_model)
        
        sys.exit(1)
    


    
