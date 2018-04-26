import sys
import zmq
import numpy as np
import sys, getopt
port = 5555
pubport = 5556
pubport_global = 5558

import signaldoctorlib as sdl
import argparse

LOG_IQ = True
#MODEL_NAME = "specmodel"

## DEBUG
import matplotlib.pyplot as plt

fs = 2.09715e6

metadata = {'radio':'HackRF', 'cf':3.9e6}


def main(args):
    IQ_LOCAL = None
    IQ_FILE = None
    
    print("Running: ", sys.argv[0])
    
    pubcontext = zmq.Context()
    pubsocket = pubcontext.socket(zmq.PUB)
    pubsocket.bind('tcp://127.0.0.1:%i' % pubport)
    
    pubcontext_global = zmq.Context()
    pubsocket_global = pubcontext_global.socket(zmq.PUB)
    pubsocket_global.bind('tcp://127.0.0.1:%i' % pubport_global)
    
    if args.input_file != None:
        print("Reading local IQ")
        print("Input file: ", args.input_file)
        sdl.process_iq_file(args.input_file, LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata)
        sys.exit()
    
    if args.d:
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
            sdl.classify_buffer(buffer_data, fs=fs, LOG_IQ=LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata)
    
    if args.zmq_src != None:
        # Socket to talk to server
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        print("Collecting updates from IQ server...")
        socket.connect ("tcp://%s" % args.zmq_src)

        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        #loaded_model, index_dict = sdl.get_spec_model(MODEL_NAME)

        while True:
            string = socket.recv()
            buffer_data = np.fromstring(string, dtype = 'complex64')
            sdl.classify_buffer(buffer_data, fs=fs, LOG_IQ=LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='### Radio Interface for Signal Doctor. Both GRC and *.WAV Inputs Supported - Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")

    parser.add_argument('-i', help='Process WAV file (single or dual channel)', action="store", dest="input_file")
    parser.add_argument('-z', help='RX from GRC -->> source_ip:port', action="store", dest="zmq_src")
    parser.add_argument('-d', help='RX from GRX with default settings (127.0.0.1:5555)', action="store_true")
    
    args = parser.parse_args()
    main(args)
