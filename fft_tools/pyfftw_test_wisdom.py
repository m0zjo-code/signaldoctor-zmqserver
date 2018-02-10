import pyfftw, pickle
import numpy as np 


with open('fftw_wisdom.wiz', "rb") as f:
    dump = pickle.load(f)
    # Now you can use the dump object as the original one  
    pyfftw.import_wisdom(dump)
    
N = 20
    
for n in range(1, N+1):
    n = 2**n
    a = np.ones(n, dtype=np.complex64)
    a.real = np.random.rand(n)
    a.imag = np.random.rand(n)

    print("Vector of Len: %i generated! :)" % (len(a)))
    fft_a = pyfftw.interfaces.numpy_fft.fft(a)
    print("Generated FFT")
