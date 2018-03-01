import sys
import zmq
import numpy as np
import sys, getopt
port = "5555"

import signaldoctorlib as sdl

LOG_IQ = True
MODEL_NAME = "specmodel"

## DEBUG
import matplotlib.pyplot as plt



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
        sdl.process_iq_file(IQ_FILE,LOG_IQ)
        sys.exit()

    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("Collecting updates from IQ server...")
    socket.connect ("tcp://127.0.0.1:5555")

    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    loaded_model, index_dict = sdl.get_spec_model(MODEL_NAME)

    while True:
        string = socket.recv()
        buffer_data = np.fromstring(string, dtype = 'complex64')
        sdl.classify_buffer(buffer_data, fs=1, LOG_IQ=LOG_IQ, loaded_model=loaded_model, loaded_index=index_dict)


if __name__ == "__main__":
    main(sys.argv[1:])
