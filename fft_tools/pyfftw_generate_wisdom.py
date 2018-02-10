import pyfftw, pickle
import numpy as np 

def generate_wisdom(N):
    for n in range(1, N+1):
        n = 2**n
        a = np.ones(n, dtype=np.complex64)
        a.real = np.random.rand(n)
        a.imag = np.random.rand(n)

        print("Vector of Len: %i generated! :)" % (len(a)))
        fft_a = pyfftw.interfaces.numpy_fft.fft(a)
        print("Generated FFT of len: %i" %(len(a)))

    with open("fftw_wisdom.wiz", "wb") as f:
        pickle.dump(pyfftw.export_wisdom(), f, pickle.HIGHEST_PROTOCOL)

    print("Exported Wisdom file")
    
def test_generated_wisdom(N):
    with open("fftw_wisdom.wiz", "rb") as f:
        dump = pickle.load(f)
        # Now you can use the dump object as the original one  
        pyfftw.import_wisdom(dump)
        
    for n in range(1, N+1):
        n = 2**n
        a = np.ones(n, dtype=np.complex64)
        a.real = np.random.rand(n)
        a.imag = np.random.rand(n)
        fft_a = pyfftw.interfaces.numpy_fft.fft(a)
        print("FFT of Len: %i generated!" % (len(a)))
