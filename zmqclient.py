import sys
import zmq
import numpy as np
port = "5555"

import signaldoctorlib as sdl


if __name__ == "__main__":
    
    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print("Collecting updates from IQ server...")
    socket.connect ("tcp://127.0.0.1:5555")

    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    while True:
        string = socket.recv()
        buffer_data = np.fromstring(string, dtype = 'complex64')
        sdl.process_buffer(buffer_data)
    