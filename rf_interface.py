import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy #use numpy for buffers
import zmq, sys

pubport_global = 5550

pubcontext = zmq.Context()
pubsocket = pubcontext.socket(zmq.PUB)
pubsocket.set_hwm(100)
pubsocket.bind('tcp://127.0.0.1:%i' % pubport_global)

#enumerate devices
results = SoapySDR.Device.enumerate()
for result in results: print(result)

#create device instance
#args can be user defined or from the enumeration result
args = dict(driver="remote", remote="tcp://10.42.0.144:5555")
sdr = SoapySDR.Device(args)

print(SOAPY_SDR_RX)

#query device info
print(sdr.listAntennas(SOAPY_SDR_RX, 0))
print(sdr.listGains(SOAPY_SDR_RX, 0))
print(sdr.getGainRange(SOAPY_SDR_RX, 0))
freqs = sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
for freqRange in freqs: print(freqRange)

#apply settings
sdr.setSampleRate(SOAPY_SDR_RX, 0, 1e6)
sdr.setFrequency(SOAPY_SDR_RX, 0, 100.3e6)
sdr.setGain(SOAPY_SDR_RX, 0, 'LNA', 0)
sdr.setGain(SOAPY_SDR_RX, 0, 'AMP', 27)
sdr.setGain(SOAPY_SDR_RX, 0, 'VGA', 44)


#setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream) #start streaming

#create a re-usable buffer for rx samples
buff = numpy.array([0]*714, numpy.complex64)

out_buff_len = 2**20
out_buff = numpy.array([0]*out_buff_len, numpy.complex64)


##receive some samples
i = 0
j = 0
while True:
    sr = sdr.readStream(rxStream, [buff], len(buff))
    if out_buff_len-i >= 714:
        out_buff[i:i+714] = buff
    else:
        delta = out_buff_len-i
        out_buff[i:i+delta] = buff[0:delta]
        print("Send Pyobj", j)
        j=j+1
        pubsocket.send_pyobj(out_buff)
        i = 0
        #sys.exit(0)
    i = i + 714

#shutdown the stream
sdr.deactivateStream(rxStream) #stop streaming
sdr.closeStream(rxStream)
