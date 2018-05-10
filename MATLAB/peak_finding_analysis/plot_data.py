import numpy as np
import matplotlib.pyplot as plt

filename = 'search_psd.npz'

npzload = np.load(filename)
buffer_abs = npzload['buffer_abs']
buffer_abs = 10*np.log(buffer_abs+0.00001)
_, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(buffer_abs)
ax.set_xlabel('Search Vector', fontsize=14)
ax.set_ylabel('10log(Amplitude/WHz^-1)', fontsize=14)
ax.set_xlim(3000, 4000)
ax.set_ylim(-70, 35)
plt.show()
