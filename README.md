
# Signal Doctor

This is the development page for Jonathan Rawlinson's Final Year Project (FYP). 

For info on training CNNs please look here ---->>> [https://github.com/m0zjo-code/SIGNAL_CNN_TRAIN_KERAS](https://github.com/m0zjo-code/SIGNAL_CNN_TRAIN_KERAS)

For info on classifying meteor scatter please head over to here ----->>>> [https://github.com/m0zjo-code/meteordoctor](https://github.com/m0zjo-code/meteordoctor})

#### Introduction
The Radio Frequency (RF) spectrum is under high demand all over the world. With the advancement of Long Term Evolution (LTE, 4G cellular radio) and 5G on the way, the RF spectrum is going to be even harder to secure for new services due to the extra bandwidth that will be required. In addition to the bandwidth demand, new services also need to work around existing services such as Global System for Mobile Communications (GSM, 2G) or WiFi.

To enable future cognitive communications systems to work effectively and efficiently (i.e. avoiding transmitting over existing systems), they will need to know what type of systems are in operation around them in real time. There are a large number of modulation schemes and protocols that exist, all with different configurations and parameters (even different implementation of the same specification can have operational differences due to different manufacturers, tolerances, etc). This presents a non-trivial challenge to the new communications system.

It is proposed that this problem can be solved utilising neural networks and the power of modern open-source software toolkits and frameworks.

#### Sample IQ Data
Some sample IQ data is available to test the system here:
- [20M Recording - 96 kHz Wav](https://goo.gl/PK8S3v)

These recordings are taken from "SDR RADIO" V3 servers -->> [http://www.sdrspace.com/Version-3](http://www.sdrspace.com/Version-3)

# Real Time Classifier - zmqclient
This functionality is made of a number of components 
#### System diagram:
```
    +   +
    |   |
    +-+-+      +---------------------+           +--------------+
      |        |                     |    IQ     |              |
      |        |     RF Receiver     |  Stream   |   Spectrum   |
      +--------+                     +----------->              +---+
 Wonderwand    |  Hackrf/RTLSDR/etc  | [ZMQ PUB] |   Processor  |   |
 Widebander    |                     |           |              |   | IQ Packets
 HF Antenna    +---------------------+           +--------------+   | [ZMQ PUB]
                                                                    |
                                                         +----------v--------+
                                                         |                   |
                                                         | Localhost/LAN/WAN |
+----------+                                             |                   |
|          |                             Signal Class    +--^--+--^--+--+----+
|          |  +-----------------------+  [ZMQ PUB]          |  |  |  |  |
|   NNET   |  |                       +---------------------+  |  |  |  |
|          +--+  Spectrum Classifier  |                        |  |  |  |
|  Models  |  |                       <------------------------+  |  |  |
|          |  +-----------------------+  IQ Packets [ZMQ SUB]     |  |  |
|          |                                                      |  |  |
|          |                                                      |  |  |
+----------+                                                      |  |  |
                         +---------------+  HTTP Web Connection   |  |  |
                         |               +------------------------+  |  |
                         | GUI Webserver |                           |  |
                         |               <---------------------------+  |
                         +---------------+ Signal Class [ZMQ SUB]       |
                                                                        |
                                                                        |
                                                                        |
                                     +--------------+                   |
                                     |              |      HTTP         |
                                     | GUI Terminal <-------------------+
                                     |              |
                                     +--------------+
```
#### Spectrum Processor
This module takes wideband IQ data from GNURadio over the network and provides the following functions:
- Fills a numpy buffer with IQ data
- Computes the PSD
- Finds power peaks and computes the bandwidth
- Extracts the signal
- The signal is published across the network for classification

Processing pre-recorded IQ data from a .wav file:
```sh
python3 spectrum_processor.py -i /your/file.wav
```

Processing live IQ data from network:  
```sh
python3 spectrum_processor.py -i 127.0.0.1:3343 
```

A config file "spectrum_processor.cfg" (default name) controls the main operation parameters.

#### Spectrum Classifier
This module takes IQ packets taken from the Spectrum Processor and runs them through the classification network/s.
The following functions are implemented:
- Feature calculation
- The data is run through the neural network/s
- The predicted signal class, bandwith, location and other misc details are transmitted over the network

```sh 
python3 spectrum_classifier.py -i 127.0.0.1:3243
```

A config file "spectrum_classifier.cfg" (default name) controls the main operation parameters.


# Setup

The following python libraries are required:
- numpy
- scipy
- detect_peaks
- keras
- tensorflow
- h5py
- matplotlib
- pyfftw (if faster FFT computation is required)
- pyzmq

```sh 
sudo -H pip3 install numpy scipy keras tensorflow h5py matplotlib pyfftw pyzmq
```

SoapySDR is also required (along with support for the radio you want to use).

Install details for your platform can be found here ---->>> [https://github.com/pothosware/SoapySDR/wiki](https://github.com/pothosware/SoapySDR/wiki)
