"""
Jonathan Rawlinson 2018
"""
import numpy as np
import math
import string
import random
import csv

from scipy import signal
from scipy.io.wavfile import read as wavfileread
from scipy.io import savemat
from scipy.misc import imresize, imsave
from scipy.stats import zscore
from numpy.fft import fftshift

from detect_peaks import detect_peaks

from keras.models import model_from_json
import tensorflow as tf


## FFT Config
import pyfftw
from scipy.fftpack import fft, ifft

## SETUP
import os.path
import pickle

## DEBUG
import matplotlib.pyplot as plt

## TODO ADD TO CFG FILE
energy_threshold = -5
peak_threshold = 0.2
smooth_stride = 1024
fs = 2e6
MaxFFTN = 22
wisdom_file = "fftw_wisdom.wiz"
iq_buffer_len = 2000 ##ms
OSR = 1
MODEL_NAME = ["specmodel", "psdmodel"]
plot = True
resample_ratio = 256


## Help from -->> https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list
def npmax(l):
    """ 
    Find Max value and index of wall
    """
    max_idx = int(np.argmax(l))
    max_val = l[max_idx]
    return (max_idx, max_val)


def get_spec_model(modelname):
    """ 
    Load models from file 
    Returns model and indexes 
    """
    loaded_model = []
    for i in MODEL_NAME:
        json_file = open('%s.nn'%(i), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model.append(model_from_json(loaded_model_json))
        # load weights into new model
        loaded_model[-1].load_weights("%s.h5"%(i))
        print("Loaded model from disk")
    ## https://stackoverflow.com/questions/6740918/creating-a-dictionary-from-a-csv-file
    with open('spec_data_index.csv', mode='r') as ifile:
        reader = csv.reader(ifile)
        d = {}
        for row in reader:
            k, v = row
            d[int(v)] = k
    return loaded_model, d

def save_IQ_buffer(channel_iq, fs, output_format = 'npy', output_folder = 'logs/'):
    """
    Write IQ data into npy file
    The MATLAB *.mat file can also be used
    """
    filename = id_generator()
    if (output_format == 'npy'):
        np.savez(output_folder+filename+".npz",channel_iq=channel_iq, fs=fs)
    else:
        savemat(output_folder+filename+".mat", {'channel_iq':channel_iq, 'fs':fs})
    
    f, t, Zxx_cmplx = signal.stft(channel_iq, 1, nperseg=512, return_onesided=False)
    Zxx_mag = np.abs(np.power(Zxx_cmplx, 2))
    imsave(output_folder+filename+".png", Zxx_mag)



## From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate random string
    """
    return ''.join(random.choice(chars) for _ in range(size))

def load_IQ_file(input_filename):
    """Read .wav file containing IQ data from SDR#"""
    fs, samples = wavfileread(input_filename)
    return fs, len(samples), samples

def import_buffer(iq_file,fs,start,end):
    """Extract buffer from array and balance the IQ streams"""
    #fs, samples = scipy.io.wavfile.read(input_filename)
    #print("\nFile Read:", input_filename, " - Fs:", fs)
    #print(len(samples))
    input_frame = iq_file[int(start):int(end)]
    input_frame_iq = np.empty(input_frame.shape[:-1], dtype=np.complex)
    input_frame_iq.real = input_frame[..., 0]
    input_frame_iq.imag = input_frame[..., 1]
    # Balance IQ file
    input_frame_iq = IQ_Balance(input_frame_iq)

    return input_frame_iq, fs


## From https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
def power_bit_length(x):
    x = int(x)
    return 2**((x-1).bit_length()-1)

def process_iq_file(filename, LOG_IQ, pubsocket=None):

    #loaded_model, index_dict = get_spec_model(MODEL_NAME)

    fs, file_len, iq_file = load_IQ_file(filename)
    print("Len file: ", file_len )

    ##Calculate buffer length
    length = (fs/1000)*iq_buffer_len
    length = power_bit_length(length)

    buf_no = int(np.floor(file_len/(length)))

    print("Length of buffer: ", length/fs, "s")
    for i in range(0, buf_no):
        print("Processing buffer %i of %i" % (i+1 , buf_no))
        ## Read IQ data into memory
        in_frame, fs = import_buffer(iq_file, fs, i*length, (i+1)*length)
        print("IQ Len: ", len(in_frame))
        classify_buffer(in_frame, fs=fs, LOG_IQ=LOG_IQ,  pubsocket=pubsocket)


def classify_buffer(buffer_data, fs=1, LOG_IQ = True, pubsocket=None):
    """
    This function generates the features and acts as an entry point into the processing framework
    The features are generated, logged (if required) and then published to the next block
    """
    extracted_features, extracted_iq = process_buffer(buffer_data, fs)

    # We now have the features and iq data
    if LOG_IQ:
        print("Logging.....")
        for iq_channel in extracted_iq:
            save_IQ_buffer(iq_channel[0], iq_channel[1])
    send_features(extracted_features, pubsocket)
    
def send_features(extracted_features, pubsocket):
    """
    Publish the feature arrays over the network
    The ZMQ socket is passed from the calling function
    """
    print(pubsocket)
    for data in extracted_features:
        data = np.asarray(data)
        pubsocket.send_pyobj(data)
        #print(data)

def classify_spectrogram(input_array, model, index):
    """
    This function takes the input features and runs them through the classification networks
    """
    print("LEN:", len(input_array))
    for i in range(0, len(input_array)):
        tmpspec = input_array[i][0]
        tmppsd = input_array[i][1]
        
        #tmpspec = tmpspec.reshape((1, tmpspec.shape[0], tmpspec.shape[1], 1))
        tmpspec_z = np.zeros((1, tmpspec.shape[0], tmpspec.shape[1], 3))
        for j in range(0, 3):
            tmpspec_z[0,:,:,j] = tmpspec

        
        tmpspec_z = tmpspec_z.astype('float32')
        tmpspec_z /= np.max(tmpspec_z)
        prediction_spec = model[0].predict(tmpspec_z)

        idx_spec = npmax(prediction_spec[0])[0]
        print("Classified signal (spec) -->>", index[idx_spec])
        
        tmppsd = tmppsd.astype('float32')
        tmppsd /= np.max(tmppsd) 
        tmppsd = tmppsd.reshape((1, tmppsd.shape[0]))
        prediction_psd= model[1].predict(tmppsd)

        idx_psd = npmax(prediction_psd[0])[0]
        print("Classified signal (psd) -->>", index[idx_psd])
        #prediction = prediction.flat[0]

def fft_wrap(iq_buffer, mode = 'pyfftw'):
    """
    Compute FFT - either using pyfftw (default) or scipy
    """
    if mode == 'pyfftw':
        return pyfftw.interfaces.numpy_fft.fft(iq_buffer)
    elif mode == 'scipy':
        return fft(iq_buffer) #### TODO CLEAN

def ifft_wrap(iq_buffer, mode = 'pyfftw'):
    """
    Compute IFFT - either using pyfftw (default) or scipy
    """
    if mode == 'pyfftw':
        return pyfftw.interfaces.numpy_fft.ifft(iq_buffer)
    elif mode == 'scipy':
        return ifft(iq_buffer)
    

def process_buffer(buffer_in, fs=1):
    """
    Analyse input buffer, extract signals and pass onto the classifiers
    """
    buffer_len = len(buffer_in)
    print("Processing signal. Len:", buffer_len)
    #Do FFT - get it out of the way!
    buffer_fft = fft_wrap(buffer_in, mode = 'scipy')
    
    ## Will now unroll FFT for ease of processing
    buffer_fft_rolled = np.roll(buffer_fft, int(len(buffer_fft)/2))
    #Is there any power there?
    buffer_abs = np.abs(buffer_fft_rolled*buffer_fft_rolled.conj())
    buffer_energy = np.log10(buffer_abs.sum()/buffer_len)

    #If below threshold return nothing
    if (buffer_energy<energy_threshold):
        print("Below threshold - returning nothing")
        return None

    #TODO TODO TODO - Fix narrow band peak detection
    ## "Bin" FFT
    ## Also smooth the buffer

    buffer_abs = np.reshape(buffer_abs, (resample_ratio, int(len(buffer_abs)/resample_ratio)), order='F')
    buffer_abs = np.mean(buffer_abs, axis=0)
    #buffer_fft_smooth = signal.resample(buffer_abs, smooth_stride)
    #buffer_fft_smooth = smooth(np.log2(buffer_abs), window_len=10)
    buffer_abs[buffer_abs == 0] = 0.01 #Remove all zeros
    
    buffer_log2abs = buffer_abs
    for i in range(0,25):
        buffer_log2abs = smooth(buffer_log2abs, window_len=10, window ='bartlett')
    
    ## Normalise
    buffer_log2abs = zscore(buffer_log2abs)
    #buffer_log2abs = signal.detrend(buffer_log2abs)
    buffer_log2abs = buffer_log2abs - np.min(buffer_log2abs)
    
    
      
    buffer_peakdata = detect_peaks(buffer_log2abs, mph=0.005 , mpd=100, edge='rising', show=True)
    #Search for signals of interest
    output_signals = []
    for peak_i in buffer_peakdata:
        #Decimate the signal in the frequency domain
        #Please note that the OSR value is from the 3dB point of the signal - if there is a long roll off (spectrally) some of the signal may be cut
        output_signal_fft, bandwidth = fd_decimate(buffer_fft_rolled, resample_ratio, buffer_log2abs, peak_i, OSR)
        print("Bandwidth--->>>", bandwidth)
        buf_len = len(buffer_fft_rolled)
        if len(output_signal_fft) ==0:
            continue
        #Compute IFFT and add to list
        td_channel = ifft_wrap(output_signal_fft, mode = 'scipy')
        output_signals.append(td_channel)
    
    #TODO TODO TODO - Fix narrow band peak detection
    
    print("We have %i signals!" % (len(output_signals)))
    
    ## Generate Features ##
    output_features = []
    output_iq = []
    for i in output_signals:
        local_fs = fs * len(i)/buffer_len
        features = generate_features(local_fs, i, plot=plot)
        features.append(local_fs)
        output_features.append(features)
        output_iq.append([i, local_fs])

    return output_features, output_iq

def log_enhance(input_array, order=1):
    input_array_tmp = input_array
    for i in range(0, order):
        min_val = np.min(input_array_tmp)
        input_array_shift = input_array_tmp-min_val+1
        input_array_tmp = np.log2(input_array_shift)
    return input_array_tmp
    

def generate_features(local_fs, iq_data, spec_size=256, roll = True, plot = False):
    """
    Generate classification features
    """
    
    ## Generate normalised spectrogram
    NFFT = math.pow(2, int(math.log(math.sqrt(len(iq_data)), 2) + 0.5)) #Arb constant... Need to analyse TODO
    #print("NFFT:", NFFT)
    f, t, Zxx_cmplx = signal.stft(iq_data, local_fs, nperseg=NFFT, return_onesided=False)
    f = fftshift(f)
    #f = np.roll(f, int(len(f)/2))
    Zxx_cmplx = fftshift(Zxx_cmplx, axes=0)
    if roll:
        Zxx_cmplx = np.roll(Zxx_cmplx, int(len(f)/2), axis=0)

    Zxx_mag = np.abs(np.power(Zxx_cmplx, 2))
    Zxx_phi = np.abs(np.angle(Zxx_cmplx))
    Zxx_cec = np.abs(np.corrcoef(Zxx_mag, Zxx_phi))
    Zxx_mag_rs = normalise_spectrogram(Zxx_mag, spec_size, spec_size)
    Zxx_phi_rs = normalise_spectrogram(Zxx_phi, spec_size, spec_size)
    Zxx_cec_rs = normalise_spectrogram(Zxx_cec, spec_size, spec_size)
    
    # We have a array suitable for NNet
    ## Generate spectral info by taking mean of spectrogram ##
    PSD = np.mean(Zxx_mag_rs, axis=1)
    
    kk = np.log2(Zxx_mag_rs+1)
    
    if plot:
        plt.subplot(2, 2, 1)
        plt.pcolormesh(log_enhance(Zxx_mag_rs+1, order=1)) ## +1 to stop 0s
        plt.subplot(2, 2, 2)
        plt.pcolormesh(Zxx_phi_rs)
        plt.subplot(2, 2, 3)
        plt.plot(PSD)
        plt.subplot(2, 2, 4)
        plt.pcolormesh(Zxx_cec_rs)
        plt.show()

    output_list = [Zxx_mag_rs, Zxx_phi_rs, Zxx_cec_rs, PSD]

    return output_list

def smooth(x,window_len=12,window='flat'):
    ## From http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    Smooth the data using a window with variable size
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[int(window_len/2-1):int(-window_len/2)]

def normalise_spectrogram(input_array, newx, newy):
    """
    Interpolate NxN array into newx and newy array
    """
    arr_max = input_array.max()
    arr_min = input_array.min()
    input_array = (input_array-arr_min)/(arr_max-arr_min)

    return imresize(input_array, (newx, newy))

def IQ_Balance(IQ_File):
    """
    Remove DC offset from input data file
    """
    DC_Offset_Real = np.mean(np.real(IQ_File))
    DC_Offset_Imag = np.mean(np.imag(IQ_File))
    return IQ_File - (DC_Offset_Real + DC_Offset_Imag * 1j)

#def pad_fft(input_fft):
    ##Pad to 2^n to speed up ifft
    #output_len = len(input_fft)
    #required_len = 2**np.ceil(np.log2(output_len))
    #out_deficit = required_len-output_len
    #zeros_add = int(out_deficit/2)
    #output_fft = np.pad(input_fft, (zeros_add, zeros_add), 'constant', constant_values=(0, 0))
    #return output_fft



def fd_decimate(fft_data, resample_ratio, fft_data_smoothed, peakinfo, osr):
    """
    Decimate Buffer in Frequency domain to extract signal from 3db Bandwidth
    """
    #Calculate 3dB peak bandwidth
    #TODO
    
    cf = peakinfo
    bw = find_3db_bw_JR_single_peak(fft_data_smoothed, peakinfo)
    bw = 100
    slicer_lower = int(cf-(bw/2)*osr)*resample_ratio
    slicer_upper = int(cf+(bw/2)*osr)*resample_ratio
    #print("Slicing Between: ", slicer_lower, slicer_upper, "BW: ", bw, "CF: ", cf, peakinfo)
    #Slice FFT
    output_fft = fft_data[slicer_lower:slicer_upper]
    return output_fft, bw

def find_3db_bw_JR_single_peak(data, peak):
    """ 
    Find bandwidth of single peak 
    """
    max_bw = find_3db_bw_max(data, peak)
    min_bw = find_3db_bw_min(data, peak)
    bw1 = (max_bw - peak)*2
    bw2 = (peak - min_bw)*2
    bw = max(bw1, bw2)
    print("BW max min:", bw, max_bw, min_bw)
    return bw

#def find_3db_bw_max(data, peak, step=10):
    #""" 
    #Find 3dB point going up the array 
    #"""
    #max_height = data[peak]
    #min_global = np.min(data)
    #thresh_3db = max_height - (np.abs(max_height - min_global))/2
    
    #tdb = np.argmax(data[peak:] < thresh_3db)
    #return tdb ##Nothing found - should probably raise an error here. TODO check error

#def find_3db_bw_min(data, peak, step=10):
    #""" 
    #Find 3dB point going down the array
    #"""
    #max_height = data[peak]
    #min_global = np.min(data)
    #thresh_3db = max_height - (np.abs(max_height - min_global))/2
    #tdb = np.argmax(np.flip(data[len(data)-peak:] < thresh_3db, axis=0))
    #return tdb ##Nothing found - should probably raise an error here


def find_3db_bw_max(data, peak, step=10):
    """ 
    Find 3dB point going up the array 
    """
    max_height = data[peak]
    min_global = np.min(data)
    thresh_3db = max_height - (np.abs(max_height - min_global))/2
    previous_value = max_height
    for i in range(peak, len(data)-1, step):
        if (data[i] < thresh_3db):
            return i
        elif (data[i] > previous_value):
            return i
        previous_value = data[i]
    return len(data)-1 ##Nothing found - should probably raise an error here. TODO check error

def find_3db_bw_min(data, peak, step=10):
    """ 
    Find 3dB point going down the array
    """
    max_height = data[peak]
    min_global = np.min(data)
    thresh_3db = max_height - (np.abs(max_height - min_global))/2
    previous_value = max_height
    for i in range(peak, 0-1, -step):
        if (data[i] < thresh_3db):
            return i
        elif (data[i] > previous_value):
            return i
    return 0 ##Nothing found - should probably raise an error here


def find_channels(data, min_height, min_distance):
    """
    Locate channels within FFT magnitude data - this function does not care about sampling rates etc. It will return the raw sample values. Smoothing is recommended
    """
    ## Input data numpy array of the data to be searched
    ## min_height is a number between 0 and 1 (e.g. 0.454) that defines the treshold of where a channel starts

    psd_len = len(data)

    max_data = np.max(data) #find max value
    min_data = np.mean(data) #find mean value TEST - Using mean as min to smooth out dips
    #min_data = np.min(data) #find min value
    #print("Min Data Val:",min_data," Max Data Val: ", max_data)

    thresh = (max_data-min_data)*min_height + min_data #Calculate the threshold for channel detection. TODO Look at delta mean instead?
    #print("Threshold Val: ",thresh)
    peaklist = []

    if data[0] < thresh: #Get initial state
        onPeak = False
    elif data[0] >= thresh:
        onPeak = True

    prevStatus = onPeak
    start_val = 0
    #peak_id = 0;
    for i in range(1, len(data)): ## Run over PSD and return peaks above threshold
        if data[i] < thresh:
            onPeak = False
        elif data[i] >= thresh:
            onPeak = True

        if onPeak == True and prevStatus == False:
            # Just arrived on peak
            start_val = i;
        elif onPeak == False and prevStatus == True:
            # Just left Peak
            width = i - start_val
            location = np.floor(start_val+width/2)
            #power = calculate_power(data[start_val:i])
            #peaklist.append([peak_id, [start_val, i], int(location)])
            peaklist.append([start_val, i, int(location)])
            #peak_id = peak_id + 1;
        prevStatus = onPeak
    
    #now we check to see if any peaks should be combined
    #for i in range(1,len(peaklist)):
    #    if ((peaklist[i][1][0] - peaklist[i-1][1][1])<min_distance):
    #        print(i)
    #        peaklist[i-1][1][1] = peaklist[i][1][1]
    #        peaklist[i-1][2] = (peaklist[i-1][2]+peaklist[i][2])/2
    #        del peaklist[i]
    return peaklist

def generate_wisdom(N, wisdom_f):
    """
    Used to generate a .wiz file to speed up boot up times
    """
    for n in range(1, N+1):
        n = 2**n
        a = np.ones(n, dtype=np.complex64)
        a.real = np.random.rand(n)
        a.imag = np.random.rand(n)

        print("Vector of Len: %i generated! :)" % (len(a)))
        fft_a = pyfftw.interfaces.numpy_fft.fft(a)
        print("Generated FFT of len: %i" %(len(a)))

    with open(wisdom_f, "wb") as f:
        pickle.dump(pyfftw.export_wisdom(), f, pickle.HIGHEST_PROTOCOL)

    print("Exported Wisdom file")


def import_wisdom(wisdom_f):
    """
    Load Wisdom file
    """
    with open(wisdom_f, "rb") as f:
        dump = pickle.load(f)
        # Now you can use the dump object as the original one
        pyfftw.import_wisdom(dump)


def test_generated_wisdom(N, wisdom_f):
    """
    Wisdom tester
    """
    for n in range(1, N+1):
        n = 2**n
        a = np.ones(n, dtype=np.complex64)
        a.real = np.random.rand(n)
        a.imag = np.random.rand(n)
        fft_a = pyfftw.interfaces.numpy_fft.fft(a)
    print("FFT tested up to %i" % (len(a)))


### Leave at bottom of file ##
#if os.path.isfile(wisdom_file):
    #import_wisdom(wisdom_file)
    #print("PyFFTW Wisdom File Exists - Loaded")

#else:
    #print("PyFFTW Widom File Not Found - Generating up to %i" %(2**MaxFFTN))
    #generate_wisdom(MaxFFTN, wisdom_file)
    #print("PyFFTW Wisdom File Generated")
