import numpy as np
import scipy.signal
import detect_peaks
import peakutils.peak
import peakdetect
import findpeaks
import tb_detect_peaks
import timeit


def plot_peaks(x, indexes, algorithm=None, mph=None, mpd=None, setup=''):
    """Plot results of the peak dectection."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
        return
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b', lw=1)
    if indexes.size:
        label = 'peak'
        label = label + 's' if indexes.size > 1 else label
        ax.plot(indexes, x[indexes], '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (indexes.size, label))
        #ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02*x.size, x.size*1.02-1)
    ax.set_xlim(3000, 4000)
    
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)  
    ax.set_ylim(-70, 35)
    ax.set_xlabel('Search Vector', fontsize=14)
    ax.set_ylabel('10log(Amplitude/WHz^-1)', fontsize=14)
    ax.set_title('%s %s' % (algorithm, setup))
    plt.show()


def scipy_cwt(vector, plot=False):
    
    indexes = scipy.signal.find_peaks_cwt(
        vector,
        np.arange(1, 10),
        #max_distances=np.arange(1, 20)*2,
        min_snr = 1.00
    )
    indexes = np.array(indexes) - 1
    
    if plot:
        print('Detect peaks without any filters.')
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='scipy.signal.find_peaks_cwt',
            setup = 'Wavelet lengths 1-10, min_snr = 1'
        )
    
def scipy_argrelextrema(vector, plot=False):
    indexes = scipy.signal.argrelextrema(
        np.array(vector),
        comparator=np.greater,order=2)
    
    if plot:
        print('Peaks are: %s' % (indexes[0]))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='scipy.signal.argrelextrema',
            setup = 'comparator=np.greater, order=2'
        )
        
def scipy_findpeaks(vector, plot=True):
    indexes, _ = scipy.signal.find_peaks(
        np.array(vector),
        prominence=10,
        height=-50,
        distance=5)
    
    if plot:
        print('Peaks are: %s' % (indexes[0]))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='scipy.signal.find_peaks',
            setup = 'prominence=10,height=-50,distance=5'
        )
    
def detect_peaks_test(vector, plot=False):
    indexes = detect_peaks.detect_peaks(
        vector, 
        mph=-40, 
        mpd=5, 
        edge='rising', 
        show=False)
    
    if plot:
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='detect_peaks.detect_peaks',
            setup = 'mph=-40, mpd=5, edge=rising edge'
        )

def peakutils_test(vector, plot=False):
    indexes = peakutils.peak.indexes(
        np.array(vector),
        thres=0.5, 
        min_dist=5)
    
    if plot:
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='peakutils.peak.indexes',
            setup = 'threshold=0.5, min_dist=2'
        )

def peakdetect_test(vector, plot=False):
    
    peaks = peakdetect.peakdetect(
        np.array(vector), 
        lookahead=5, 
        delta=5)
    # peakdetect returns two lists, respectively positive and negative peaks,
    # with for each peak a tuple of (indexes, values).
    indexes = []
    for posOrNegPeaks in peaks:
        for peak in posOrNegPeaks:
            indexes.append(peak[0])
    
    if plot:
        print('Detect peaks with distance filters.')
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='peakdetect',
            setup = 'lookahead=5, delta=5'
        )

def findpeaks_test(vector, plot=False):
    indexes = findpeaks.findpeaks(
        np.array(vector), 
        spacing=1, 
        limit=-40)
    
    if plot:
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='findpeaks',
            setup = 'spacing=1, limit=-40'
        )


def tb_detect_peaks_test(vector, plot=False):
    indexes = tb_detect_peaks.detect_peaks(vector, -0.5)
    if plot:
        print('Detect peaks with height threshold.')
        print('Peaks are: %s' % (indexes))
        plot_peaks(
            np.array(vector),
            np.array(indexes),
            algorithm='tb_detect_peaks.detect_peaks',
            setup = 'threshold = -0.5'
        )
    

    
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


filename = 'search_psd.npz'

npzload = np.load(filename)
buffer_abs = npzload['buffer_abs']

f = open('pd_logs.csv','w')

number = 10
buffer_abs = 10*np.log(buffer_abs+0.00001)

scipy_findpeaks(buffer_abs, plot=True)

#print("Mode: min(ts)/number")
#print("Mode: min(ts)/number", file=f)

#wrapped_fft_wrap = wrapper(scipy_cwt, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("scipy_cwt:", min(ts)/number)
#print("scipy_cwt:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(scipy_argrelextrema, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("scipy_argrelextrema:", min(ts)/number)
#print("scipy_argrelextrema:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(scipy_argrelextrema, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("scipy_argrelextrema:", min(ts)/number)
#print("scipy_argrelextrema:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(detect_peaks_test, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("detect_peaks_test:", min(ts)/number)
#print("detect_peaks_test:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(peakutils_test, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("peakutils_test:", min(ts)/number)
#print("peakutils_test:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(peakdetect_test, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("peakdetect_test:", min(ts)/number)
#print("peakdetect_test:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(findpeaks_test, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("findpeaks_test:", min(ts)/number)
#print("findpeaks_test:", min(ts)/number, file=f)

#wrapped_fft_wrap = wrapper(tb_detect_peaks_test, buffer_abs, plot=False)
#ts = timeit.repeat(wrapped_fft_wrap, number=number)
#print("tb_detect_peaks_test:", min(ts)/number)
#print("tb_detect_peaks_test:", min(ts)/number, file=f)
