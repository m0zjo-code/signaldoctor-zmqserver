# Signal Doctor

![SD Logo](https://raw.githubusercontent.com/m0zjo-code/signaldoctor-zmqserver/master/docs/sdlogo.png?token=ALDzqCvSIUXJpBcQLG5wDF6RKRzOOui0ks5a5N2EwA%3D%3D)

This is the development page for Jonathan Rawlinson's Final Year Project (FYP). 

#### Introduction
The Radio Frequency (RF) spectrum is under high demand all over the world. With the advancement of Long Term Evolution (LTE, 4G cellular radio) and 5G on the way, the RF spectrum is going to be even harder to secure for new services due to the extra bandwidth that will be required. In addition to the bandwidth demand, new services also need to work around existing services such as Global System for Mobile Communications (GSM, 2G) or WiFi.

To enable future cognitive communications systems to work effectively and efficiently (i.e. avoiding transmitting over existing systems), they will need to know what type of systems are in operation around them in real time. There are a large number of modulation schemes and protocols that exist, all with different configurations and parameters (even different implementation of the same specification can have operational differences due to different manufacturers, tolerances, etc). This presents a non-trivial challenge to the new communications system.

It is proposed that this problem can be solved utilising neural networks and the power of modern open-source software toolkits and frameworks.

#### Sample IQ Data
Some sample IQ data is availible to test the system here:
- [20M Recording - 96 kHz Wav](https://nodejs.org/)
- [40M Recording - 96 kHz Wav](https://nodejs.org/)
- [80M Recording - 96 kHz Wav](https://nodejs.org/)

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
python3 spectrum_processor.py -c /the/config.cfg -i /your/file.wav
```

Processing live IQ data from GNURadio:  %%%TBC%%%
```sh
python3 spectrum_processor.py -c /the/config.cfg -i 127.0.0.1:3343 
```

A config file "spectrum_processor.cfg" (default name) controls the main operation parameters.

#### Spectrum Classifier %%%TBC%%%
This module takes IQ packets taken from the Spectrum Processor and runs them through the classification network/s.
The following functions are implemented:
- Feature calculation
- The data is run through the neiral network/s
- The predicted signal class, bandwith, location and other misc details are transmitted over the network

```sh 
python3 spectrum_classifier.py -c /the/config.cfg -i 127.0.0.1:3243
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

GNURadio is also required (along with support for the radio you want to use).

```sh 
sudo apt install gnuradio
```

