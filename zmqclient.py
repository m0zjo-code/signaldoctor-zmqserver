import sys
import zmq
import numpy as np
import sys, getopt
port = "5555"

import signaldoctorlib as sdl

MODEL_NAME = "spec_model"

from keras.models import model_from_json

LOG_IQ = True


## DEBUG
import matplotlib.pyplot as plt

def classify_spectrogram(input_array, model):

    print("Classify Spec")
    print("Data Shape:", input_array.shape)
    input_array = input_array.reshape((1, input_array.shape[0], input_array.shape[1], 1))
    prediction = model.predict(input_array)
    print(prediction)
    prediction = prediction.flat[0]



def main(argv):
    IQ_LOCAL = None
    IQ_FILE = None
    
    print("Running: ", sys.argv[0])
    
    try:
        opts, args = getopt.getopt(argv,"hli:",["ifile="])
    except getopt.GetoptError:
        print('%s -i <inputfilepath>'%sys.argv[0])
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('%s -i <inputfilepath>'%sys.argv[0])
            sys.exit()
        if opt == '-l':
            IQ_LOCAL = False
        elif opt in ("-i", "--ifile"):
            IQ_LOCAL = True
            IQ_FILE = arg
    
    if (len(opts) == 0):
        print("Using default parameters ->> Loading data from ZMQ PUB")
    
    if  IQ_LOCAL:
        print("Reading local IQ")
        print("Input file: ", IQ_FILE)
        sdl.process_iq_file(IQ_FILE)
        sys.exit()

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
        extracted_features, extracted_iq = sdl.process_buffer(buffer_data)

        # We now have the features and iq data
        if LOG_IQ:
            print("Logging.....")
            for iq_channel in extracted_iq:
                sdl.save_IQ_buffer(iq_channel[0], iq_channel[1])

        spec = np.asarray(extracted_features[0][0])
        #plt.pcolormesh(extracted_features[0][0])
        #plt.show()
        classify_spectrogram(spec, loaded_model)

        #sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
