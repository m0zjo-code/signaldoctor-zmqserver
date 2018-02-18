"""
Jonathan Rawlinson 2018
"""
import numpy as np
import pyfftw
#import pywt
import math
from scipy import signal
from scipy.io.wavfile import read as wavfileread
from scipy.misc import imresize
from numpy.fft import fftshift

## SETUP
import os.path
import pickle

## DEBUG
import matplotlib.pyplot as plt

## TODO ADD TO CFG FILE
energy_threshold = -5
smooth_stride = 1024
fs = 2**21
MaxFFTN = 22
wisdom_file = "fftw_wisdom.wiz"
iq_buffer_len = 500 ##ms 

def load_IQ_file(input_filename):
    """Read .wav file containing IQ data from SDR#"""
    fs, samples = wavfileread(input_filename)
    return fs, len(samples), samples

def import_buffer(iq_file,fs,start,end):
    """Extract buffer from array and balance the IQ streams"""
    #fs, samples = scipy.io.wavfile.read(input_filename)
    #print("\nFile Read:", input_filename, " - Fs:", fs)
    #print(len(samples))
    input_frame = np.transpose(iq_file[int((fs/1000)*start):int((fs/1000)*end)])
    input_frame_iq = np.empty(input_frame.shape[:-1], dtype=np.complex)
    input_frame_iq.real = input_frame[..., 0]
    input_frame_iq.imag = input_frame[..., 1]
    # Balance IQ file
    input_frame_iq = IQ_Balance(input_frame_iq)

    return input_frame_iq, fs

def process_iq_file(filename):
    fs, file_len, iq_file = load_IQ_file(filename)
    print("Len file: ", file_len ) 
    length = iq_buffer_len
    buf_no = int(np.floor(file_len/(length)))
    print("Length of buffer: ", length/fs, "s")
    for i in range(0, buf_no):
        print("Processing buffer %i of %i" % (i+1 , buf_no))
        ## Read IQ data into memory
        in_frame = import_buffer(iq_file, fs, i*length, (i+1)*length)
        print("IQ Len: ", len(in_frame))
        extracted_features = process_buffer(in_frame)

def process_buffer(buffer_in):
    buffer_len = len(buffer_in)
    print("Processing signal. Len:", buffer_len)
    #buffer_in = IQ_Balance(buffer_in)
    #Do FFT - get it out of the way!
    buffer_fft = pyfftw.interfaces.numpy_fft.fft(buffer_in)
    ## Will now unroll FFT for ease of processing
    buffer_fft = np.roll(buffer_fft, int(buffer_len/2))
    
    #Is there any power there?
    buffer_abs = np.abs(buffer_fft*buffer_fft.conj())
    buffer_energy = np.log10(buffer_abs.sum()/buffer_len)
    
    #If below threshold return nothing
    if (buffer_energy<energy_threshold):
        print("Below threshold - returning nothing")
        return None
    
    #Smooth buffer
    #buffer_fft_smooth = buffer_abs
    
    ## "Bin" FFT
    ## Should probably compute moving avg here too
    buffer_fft_smooth = signal.resample(buffer_abs, smooth_stride)
    
    print(buffer_fft_smooth.shape)
    #buffer_fft_smooth =  buffer_abs.reshape(-1, smooth_stride).mean(axis=1)
        
    print("Smoothed buffer len:", len(buffer_fft_smooth))
    #plt.plot(buffer_fft_smooth)
    #plt.show()
    
    #Search for signals of interest
    buffer_peakdata = find_channels(buffer_fft_smooth, 0.5, 1)
    
    output_signals = []
    for peak_i in buffer_peakdata:
        #Decimate the signal in the frequency domain
        output_signal_fft, bandwidth = fd_decimate(buffer_fft, buffer_fft_smooth, peak_i, smooth_stride, 4)
        
        buf_len = len(buffer_fft)
        
        if len(output_signal_fft) ==0:
            continue
        #Pad FFT for faster computation
        #output_signal_fft = pad_fft(output_signal_fft)
        #bandwidth = ((peak_i[1]-peak_i[0])/smooth_stride) * fs
        #print(bandwidth)
        #Compute IFFT and add to list
        output_signals.append(pyfftw.interfaces.numpy_fft.ifft(output_signal_fft))
    
    print("We have %i signals!" % (len(output_signals)))
    
    ## Generate Features ##
    
    output_features = []
    for i in output_signals:
        local_fs = fs * len(i)/buffer_len
        print("Resampled FS: ", local_fs)
        features = generate_features(local_fs, i)
        features.append(local_fs)
        output_features.append(features)
    
    return output_features
    
    
def generate_features(local_fs, iq_data, spec_size=256, roll = True):
    
    ## Generate normalised spectrogram
    NFFT = math.pow(2, int(math.log(math.sqrt(len(iq_data)), 2) + 0.5)) #Arb constant... Need to analyse TODO
    #print("NFFT:", NFFT)
    
    f, t, Zxx_cmplx = signal.stft(iq_data, local_fs, nperseg=NFFT, return_onesided=False)
    
    f = fftshift(f)
    #f = np.roll(f, int(len(f)/2))
    Zxx_cmplx = fftshift(Zxx_cmplx, axes=0)
    if roll:
        Zxx_cmplx = np.roll(Zxx_cmplx, int(len(f)/2), axis=0)
    
    Zxx_abs = np.abs(Zxx_cmplx)
    
    Zxx_rs = normalise_spectrogram(Zxx_abs, spec_size, spec_size)
    
    # We have a array suitable for NNet
    ## Generate spectral info by taking mean of spectrogram ##
    PSD = np.mean(Zxx_rs, axis=1)
    
    #TD_pwr = np.sqrt(np.square(iq_data.real) + np.square(iq_data.imag))
    
    #plt.plot(TD_pwr)
    #plt.show()
    
    ## Generate CWT ##
    #widths = np.arange(1, 101)
    #cwtmatr_real = signal.cwt(i.real, signal.ricker, widths)
    #cwtmatr_imag = signal.cwt(i.imag, signal.ricker, widths)
    #plt.pcolormesh(cwtmatr_real) 
    #plt.show()
    
    #plt.pcolormesh(Zxx_rs)
    #plt.show()
    
    output_list = [Zxx_rs, PSD]
    
    return output_list
        
def normalise_spectrogram(input_array, newx, newy):
    arr_max = input_array.max()
    arr_min = input_array.min()
    input_array = (input_array-arr_min)/(arr_max-arr_min)   
    
    return imresize(input_array, (newx, newy))

def IQ_Balance(IQ_File):
    """Remove DC offset from input data file"""
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
    
def fd_decimate(fft_data, fft_data_smoothed, peakinfo, smooth_stride, osr):
    #Calculate 3dB peak bandwidth
    cf = peakinfo[2]*smooth_stride
    bw = find_3db_bw_JR_single_peak(fft_data_smoothed, peakinfo[2])*smooth_stride
    slicer_lower = int(cf-(bw/2)*osr)
    slicer_upper = int(cf+(bw/2)*osr)
    
    #Slice FFT
    output_fft = fft_data[slicer_lower:slicer_upper]
    return output_fft, bw

def find_3db_bw_JR_single_peak(data, peak):
    """ Find bandwidth of single peak """
    max_bw = find_3db_bw_max(data, peak)
    min_bw = find_3db_bw_min(data, peak)
    bw = max_bw - min_bw
    return bw

def find_3db_bw_max(data, peak, step=1):
    """ Find 3dB point going up the array """
    max_height = data[peak]
    min_global = np.min(data)
    thresh_3db = max_height - (np.abs(max_height - min_global))/2
    for i in range(peak, len(data)-1, step):
        if (data[i] < thresh_3db):
            return i
    return len(data)-1 ##Nothing found - should probably raise an error here. TODO check error

def find_3db_bw_min(data, peak, step=1):
    """ Find 3dB point going down the array""" 
    max_height = data[peak]
    min_global = np.min(data)
    thresh_3db = max_height - (np.abs(max_height - min_global))/2
    for i in range(peak, 0-1, -step):
        if (data[i] < thresh_3db):
            return i
    return 0 ##Nothing found - should probably raise an error here

    
def find_channels(data, min_height, min_distance):
    """Locate channels within FFT magnitude data - this function does not care about sampling rates etc. It will return the raw sample values. Smoothing is recommended""" 
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
    with open(wisdom_f, "rb") as f:
        dump = pickle.load(f)
        # Now you can use the dump object as the original one  
        pyfftw.import_wisdom(dump)
    
    
def test_generated_wisdom(N, wisdom_f):
    for n in range(1, N+1):
        n = 2**n
        a = np.ones(n, dtype=np.complex64)
        a.real = np.random.rand(n)
        a.imag = np.random.rand(n)
        fft_a = pyfftw.interfaces.numpy_fft.fft(a)
    print("FFT tested up to %i" % (len(a)))


## Leave at bottom of file ##
if os.path.isfile(wisdom_file):
    import_wisdom(wisdom_file)
    print("PyFFTW Wisdom File Exists - Loaded")
    
else:
    print("PyFFTW Widom File Not Found - Generating up to %i" %(2**MaxFFTN))
    generate_wisdom(MaxFFTN, wisdom_file)
    print("PyFFTW Wisdom File Generated")
