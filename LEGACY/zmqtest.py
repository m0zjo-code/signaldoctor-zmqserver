import zmq
import numpy as np
# Socket to talk to server
port = 5556
context = zmq.Context()
socket = context.socket(zmq.SUB)
print("Collecting updates from IQ server...")
socket.connect ("tcp://127.0.0.1:%i" % port)

socket.setsockopt_string(zmq.SUBSCRIBE, "")

i = 1
while True:
    array = socket.recv_pyobj()
    print(i)
    i = i + 1
    print(type(array))
