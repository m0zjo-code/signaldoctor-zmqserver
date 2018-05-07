import numpy as np
import scipy.signal
from detect_peaks import detect_peaks
import peakutils.peak
import peakdetect
import findpeaks


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
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('%s %s' % (algorithm, setup))
    plt.show()


def scipy_cwt(vector):
    print('Detect peaks without any filters.')
    indexes = scipy.signal.find_peaks_cwt(
        vector,
        np.arange(1, 10),
        #max_distances=np.arange(1, 20)*2,
        min_snr = 1.00
    )
    indexes = np.array(indexes) - 1
    print('Peaks are: %s' % (indexes))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='scipy.signal.find_peaks_cwt',
        setup = 'Wavelet lengths 1-10, min_snr = 1'
    )
    
def scipy_argrelextrema(vector):
    indexes = scipy.signal.argrelextrema(
        np.array(vector),
        comparator=np.greater,order=2)
    print('Peaks are: %s' % (indexes[0]))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='scipy.signal.argrelextrema'
    )
    
def detect_peaks_test(vector):
    indexes = detect_peaks(
        vector, 
        mph=-40, 
        mpd=5, 
        edge='rising', 
        show=False)
    print('Peaks are: %s' % (indexes))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='detect_peaks'
    )

def peakutils_test(vector):
    indexes = peakutils.peak.indexes(
        np.array(vector),
        thres=0.5, 
        min_dist=2)
    print('Peaks are: %s' % (indexes))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='peakutils.peak.indexes'
    )

def peakdetect_test(vector):
    print('Detect peaks with distance filters.')
    peaks = peakdetect.peakdetect(
        np.array(vector), 
        lookahead=20, 
        delta=10)
    # peakdetect returns two lists, respectively positive and negative peaks,
    # with for each peak a tuple of (indexes, values).
    indexes = []
    for posOrNegPeaks in peaks:
        for peak in posOrNegPeaks:
            indexes.append(peak[0])
    print('Peaks are: %s' % (indexes))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='peakutils.peak.indexes'
    )

def findpeaks_test(vector):
    indexes = findpeaks.findpeaks(
        np.array(vector), 
        spacing=1, 
        limit=-40)
    print('Peaks are: %s' % (indexes))
    plot_peaks(
        np.array(vector),
        np.array(indexes),
        algorithm='peakutils.peak.indexes'
    )


filename = 'search_psd.npz'

npzload = np.load(filename)
buffer_abs = npzload['buffer_abs']

scipy_cwt(10*np.log(buffer_abs+0.00001))