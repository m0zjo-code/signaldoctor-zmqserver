import wave
import numpy as np

BUFFER_SIZE = 2048

fp = wave.open('/mnt/datastore/FYP/IQ_Data/RFData1/wbfm/SDRSharp_20170915_130441Z_95500000Hz_IQ.wav')

output = wave.open('output.wav', 'wb')
output.setparams(fp.getparams())

frames_to_read = BUFFER_SIZE / (fp.getsampwidth() + fp.getnchannels())

while True:
    frames = fp.readframes(int(frames_to_read))
    print(frames)
    if not frames:
        break
    output.writeframes(frames)
