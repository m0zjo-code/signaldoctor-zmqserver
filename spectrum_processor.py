import sys
import zmq
import numpy as np
import sys, getopt
port = 5555
pubport = 5556
pubport_global = 5558

import signaldoctorlib as sdl

LOG_IQ = True
#MODEL_NAME = "specmodel"

## DEBUG
import matplotlib.pyplot as plt

fs = 2.09715e6

metadata = {'radio':'HackRF', 'cf':3.9e6}

def main(argv):
    IQ_LOCAL = None
    IQ_FILE = None
    
    print("Running: ", sys.argv[0])
    
    pubcontext = zmq.Context()
    pubsocket = pubcontext.socket(zmq.PUB)
    pubsocket.bind('tcp://127.0.0.1:%i' % pubport)
    
    pubcontext_global = zmq.Context()
    pubsocket_global = pubcontext_global.socket(zmq.PUB)
    pubsocket_global.bind('tcp://127.0.0.1:%i' % pubport_global)
    
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
        sdl.process_iq_file(IQ_FILE,LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata)
        sys.exit()

    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("Collecting updates from IQ server...")
    socket.connect ("tcp://127.0.0.1:%i" % port)

    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    #loaded_model, index_dict = sdl.get_spec_model(MODEL_NAME)

    while True:
        string = socket.recv()
        buffer_data = np.fromstring(string, dtype = 'complex64')
        sdl.classify_buffer(buffer_data, fs=fs, LOG_IQ=LOG_IQ, pubsocket=pubsocket, metadata=metadata)


if __name__ == "__main__":
    main(sys.argv[1:])
