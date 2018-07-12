"""
Jonathan Rawlinson - 2018
Imperial College EEE Department
Spectrum Processor for RF Signal Classification Project
"""

import sys
import zmq
import numpy as np
import sys, getopt
port = 5550
pubport = 5556
pubport_global = 5558

import signaldoctorlib as sdl
import argparse
import configparser
from numpy.fft import fftshift

from scipy import signal
import matplotlib.pyplot as plt

import skimage.io as io
from sklearn.preprocessing import MinMaxScaler
## Setup Data - needs to match GNURadio setup to make sense

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
        process_iq_file(args.input_file, LOG_IQ, pubsocket=[pubsocket, pubsocket_global], config=config)
        sys.exit()

def process_iq_file(filename, LOG_IQ, pubsocket=None, config=None):
    """
    Loads IQ data from a .WAV file from sdr# and classifies
    """
    # Load data
    fs, file_len, iq_file = sdl.load_IQ_file(filename)
    print("###Loaded IQ File###, Len: %i, Fs %i" % (file_len, fs))

    # Calculate buffer length
    iq_buffer_len = float(config['DETECTION_OPTIONS']['iq_buffer_len'])
    length = (fs/1000)*iq_buffer_len*4
    length = sdl.power_bit_length(length)
    buf_no = int(np.floor(file_len/(length)))
    
    # Iterate over the full buffer
    print("Length of buffer: ", length/fs, "s")
    for i in range(0, buf_no):
        print("Processing buffer %i of %i" % (i+1 , buf_no))
        # Read IQ section into memory
        in_frame, fs = sdl.import_buffer(iq_file, fs, i*length, (i+1)*length)
        generate_spectrum(fs=fs, in_frame=in_frame)
        
def generate_spectrum(fs = None, in_frame = None):
    print("Loaded frame:", len(in_frame))
    NFFT = 2**14
    f, t, Zxx = signal.stft(in_frame, fs, nperseg=NFFT, noverlap = NFFT/2)
    f = fftshift(f)
    Yxx = np.absolute(np.square(Zxx), dtype=np.float32)
    Yxx = np.log10(Yxx)
    Yxx_t = np.transpose(Yxx)
    Yxx_t = (2*normalise(Yxx_t)-1)
    
    cm = plt.get_cmap('jet')
    colored_image = cm(Yxx_t)
    
    print("###########")
    print(colored_image.shape)
    print("###########")
    #io.imshow(Yxx)
    #io.show()
    io.imsave(sdl.id_generator(size = 12)+".png", colored_image)
    #sys.exit(1)
    #plt.pcolormesh(f, t, Yxx_t, cmap='jet')
    #plt.title('STFT Magnitude')
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('Time [sec]')
    #plt.show()

def normalise(input_data):
    return (input_data - np.min(input_data))/(np.max(input_data) - np.min(input_data))
    
if __name__ == "__main__":
    # Setup arg-parser
    parser = argparse.ArgumentParser(description='### Magnitude Spectrum - Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")

    parser.add_argument('--input', help='Process WAV file (single or dual channel)', action="store", dest="input_file")

    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('sdl_config_gen.ini')
    main(args, config)
