import timeit
from scipy.fftpack import fft, ifft
import pyfftw
import numpy as np


def fft_wrap(iq_buffer, mode = 'pyfftw'):
    """
    Compute FFT - either using pyfftw (default) or scipy
    """
    if mode == 'pyfftw':
        return pyfftw.interfaces.numpy_fft.fft(iq_buffer)
    elif mode == 'scipy':
        return fft(iq_buffer) #### TODO CLEAN
    
def wrapper(func, *args, **kwargs):
    def wrapped():
        return fft_wrap(*args, **kwargs)
    return wrapped
    
test_lens = [
    2**8,
    2**10,
    2**12,
    2**14,
    2**16,
    2**18,
    2**20,
    906679,
    500
    ]

test_lens= [2**8, 2**10]
number = 2

def test_fft(mode, test_lens):
    for N in test_lens:
        random_data = 1j*np.random.rand(N) + np.random.rand(N)
        wrapped_fft_wrap = wrapper(fft_wrap, random_data, mode = mode)
        #ts_cold = timeit.repeat(wrapped_fft_wrap, number=1)
        ts = timeit.repeat(wrapped_fft_wrap, number=number)
        print('%s, %i, %f, %f, %f'%(mode, N, min(ts)/number, np.mean(ts)/number, max(ts)/number), file=f)
        print("FFT:", N, 'Mode:', mode, min(ts)/number)

f = open('FFT_logs.csv','w')

print('## FFT Timing Test ##', file=f)
print('Mode, N, Min_Time, Mean_Time, Cold_Run', file=f)

print("Running FFT Test - This may take a while")

test_fft('scipy', test_lens) 
test_fft('pyfftw', test_lens) 

f.close()
