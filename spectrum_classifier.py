import zmq
import numpy as np
import signaldoctorlib as sdl
import signaldoctorlib_class as sdlc

plot_features = False


model1_name = 'nnet_models/psdmodel'
index1_name = 'nnet_models/psd_data_index'
model1, index1 = sdlc.get_spec_model(model1_name, index1_name)

model2_name = 'nnet_models/specmodel'
index2_name = 'nnet_models/spec_data_index'
model2, index2 = sdlc.get_spec_model(model2_name, index2_name)



# Socket to talk to IQ server
port_rx = 5556
context_rx = zmq.Context()
socket_rx = context_rx.socket(zmq.SUB)
print("Collecting updates from IQ server...")
socket_rx.connect ("tcp://127.0.0.1:%i" % port_rx)
socket_rx.setsockopt_string(zmq.SUBSCRIBE, "")

# Socket to talk to GUI server/s
port_tx = 5557
context_tx = zmq.Context()
socket_tx = context_tx.socket(zmq.PUB)
socket_tx.bind('tcp://127.0.0.1:%i' % port_tx)


i = 0
while True:
    input_packet = socket_rx.recv_pyobj()
    print("## Packet No.",i)
    i = i + 1
    feature_dict = sdl.generate_features(input_packet['local_fs'], input_packet['iq_data'], plot_features=plot_features)
    class_output1 = sdlc.classify_buffer1d(feature_dict, model1)
    print("PSD Prediction  -->> %s"%index1[class_output1])
    class_output2 = sdlc.classify_buffer2d(feature_dict, model2, spec_size = 256)
    print("Spec Prediction -->> %s"%index1[class_output1])
    
    output_dict = {}
    output_dict['pred1'] = class_output1
    output_dict['pred2'] = class_output2
    output_dict['metadata'] = input_packet['metadata']
    output_dict['spectrogram'] = input_packet['magnitude']
    pubsocket.send_pyobj(output_dict)
    

