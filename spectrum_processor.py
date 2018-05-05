"""
Jonathan Rawlinson - 2018
Imperial College EEE Department
Spectrum Processor for RF Signal Classification Project
"""

import sys
import zmq
import numpy as np
import sys, getopt
port = 5555
pubport = 5556
pubport_global = 5558

import signaldoctorlib as sdl
import argparse
import configparser

## DEBUG
import matplotlib.pyplot as plt

## Setup Data - needs to match GNURadio setup to make sense
fs = 2.09715e6
metadata = {'radio':'HackRF', 'cf':3.9e6}


def main(args, config):
    # Main function
    print("Running: ", sys.argv[0])
    
    # Setup ZMQ contexts and sockets
    pubcontext = zmq.Context()
    pubsocket = pubcontext.socket(zmq.PUB)
    pubsocket.bind('tcp://127.0.0.1:%i' % pubport)
    pubcontext_global = zmq.Context()
    pubsocket_global = pubcontext_global.socket(zmq.PUB)
    pubsocket_global.bind('tcp://127.0.0.1:%i' % pubport_global)
    LOG_IQ = config['DETECTION_OPTIONS'].getboolean('LOG_IQ')
    # Checks to see if we are reading from a pre-recorded file
    if args.input_file != None:
        print("Reading local IQ")
        print("Input file: ", args.input_file)
        sdl.process_iq_file(args.input_file, LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata, config=config)
        sys.exit()
    
    # Checks to see if we are using default settings
    if args.d:
        # Socket to talk to IQ server
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        print("Collecting updates from IQ server...")
        socket.connect ("tcp://127.0.0.1:%i" % port)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Keep recieving data and processing
        while True:
            string = socket.recv()
            buffer_data = np.fromstring(string, dtype = 'complex64')
            sdl.analyse_buffer(buffer_data, fs=fs, LOG_IQ=LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata, config=config)
    
    # Custom settings - need to compleet full setup TODO
    if args.zmq_src != None:
        # Socket to talk to server
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        print("Collecting updates from IQ server...")
        socket.connect ("tcp://%s" % args.zmq_src)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Keep recieving data and processing
        while True:
            string = socket.recv()
            buffer_data = np.fromstring(string, dtype = 'complex64')
            sdl.analyse_buffer(buffer_data, fs=fs, LOG_IQ=LOG_IQ, pubsocket=[pubsocket, pubsocket_global], metadata=metadata, config=config)
    print("No arguments supplied - exiting. For help please run with -h flag")
    
if __name__ == "__main__":
    # Setup arg-parser
    parser = argparse.ArgumentParser(description='### Radio Interface for Signal Doctor. Both GRC and *.WAV Inputs Supported - Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")

    parser.add_argument('-i', help='Process WAV file (single or dual channel)', action="store", dest="input_file")
    parser.add_argument('-z', help='RX from GRC -->> source_ip:port', action="store", dest="zmq_src")
    parser.add_argument('-d', help='RX from GRX with default settings (127.0.0.1:5555)', action="store_true")
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('sdl_config.ini')
    main(args, config)
