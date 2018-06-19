"""
Jonathan Rawlinson - 2018
Imperial College EEE Department
Spectrum Classifier (from ZMQ PUB) for RF Signal Classification Project
"""
# Import std python modules
import argparse
import time, datetime
import configparser

# Import 3rd party modules
import zmq
import numpy as np

# Import my modules
import signaldoctorlib as sdl
import signaldoctorlib_class as sdlc #Seperate lib so that the processor doesn't need keras

plot_features = False


network_prefix = "/home/jonathan/prototype_networks/"

network_definition_location = [
    network_prefix+"CecSpec_Adadelta_4_1_1527900104",
    network_prefix+"FFTSpec_Adadelta_4_1_1527917966",
    network_prefix+"MagSpec_Adadelta_4_1_1527866143",
    network_prefix+"MaxPSD_Adamax_1_2_1527865830",
    network_prefix+"MeanPSD_Adamax_1_2_1527865687",
    network_prefix+"MinPSD_Adamax_1_2_1527865966",
    network_prefix+"VarPSD_Adamax_1_2_1527866038"
    ]

class_dec = network_prefix+"MagSpecTrainingData"

network_list = []
for name in network_definition_location:
    network_list.append(sdlc.get_spec_model(modelname=name, indexname=class_dec))

print(network_list[0][1])

def norm_data(X):
    if np.max(X) == 0:
        return np.zeros(X.shape, dtype=np.float32)
    return (X-np.min(X))/(np.max(X)-np.min(X))

feature_gen_modes = [
    "corrcoef",
    "fft_spectrum",
    "magnitude",
    "max_spectrum",
    "psd",
    "min_spectrum",
    "variencespectrum"
    ]

num_classes = 6

def main(args, config):
    
    # If default settings
    if args.d:
        # Extract classifier sections in ini file
        classifier_sections = [s for s in config.sections() if "CLASSIFIER" in s]
        
        #TODO - implement dynamic classifier loading
        
        # Load models
        #model1_name = 'nnet_models/psdmodel'
        #index1_name = 'nnet_models/psd_data_index'
        #model1, index1 = sdlc.get_spec_model(model1_name, index1_name)

        #model2_name = 'nnet_models/specmodel'
        #index2_name = 'nnet_models/spec_data_index'
        #model2, index2 = sdlc.get_spec_model(model2_name, index2_name)

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

        packetno = 0
        while True:
            # Get classification packet
            input_packet = socket_rx.recv_pyobj()
            print("## Packet No.",packetno)
            packetno = packetno + 1 # Packet no for debugging
            
            # Generate features
            feature_dict = sdl.generate_features(input_packet['local_fs'], input_packet['iq_data'], plot_features=plot_features, config=config)
            
            score_output = []
            
            for i in range(0, len(feature_gen_modes)):
                data_list = np.asarray(feature_dict['%s'%feature_gen_modes[i]])
            
                #run Classification
                data_list = data_list.astype('float32')
                data_list = norm_data(data_list)
                data_list = np.expand_dims(data_list, axis=-1)
                data_list = np.array([data_list])
                #scores = network_list[i][0].evaluate(data_list, Y_test_SNR, verbose=1)  # Evaluate the trained model on the test set!
                #print("Mode:", modes[i], "SNR:", snr)
                #print("%s: %.2f%%" % (network_list[i][0].metrics_names[1], scores[1]*100))
                network_prediction = network_list[i][0].predict(data_list, verbose=1)
                score_output.append(network_prediction)
                
            score_output = np.asarray(score_output)
            
            # Get the max value argument across the signal sample axis
            
            
            score_output = np.sum(score_output, axis = 0)
            
            print("Score:", score_output)
            aggregated_output = np.argmax(score_output, axis=1)
            idx = aggregated_output[0]
            
            print("Idx:", network_list[0][1][idx])
            
            #### If we want more classifiers - we add more here ###
            ## Classify from network 1
            #class_output1 = sdlc.classify_buffer1d(feature_dict, model1)
            #print("PSD Prediction  -->> %s"%index1[class_output1])
            
            ## Classify from network 2 
            #class_output2 = sdlc.classify_buffer2d(feature_dict, model2, spec_size = 256)
            #print("Spec Prediction -->> %s"%index1[class_output1])
            
            #TODO Reenable
            
            # Setup output packet
            feature_dict['pred1'] = idx
            feature_dict['pred2'] = "N/A"
            feature_dict['metadata'] = input_packet['metadata']
            feature_dict['offset'] = input_packet['offset']
            ts = time.time()
            feature_dict['timestamp'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f')
            print(feature_dict['metadata'])
            
            # Send dict
            socket_tx.send_pyobj(feature_dict)
    
    print("No arguments supplied - exiting. For help please run with -h flag")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='### Classification Server for Signal Doctor - Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")

    parser.add_argument('-d', help='RX from spectrum processor with default settings (in->127.0.0.1:5556, out->127.0.0.1:5557)', action="store_true")
    
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read('sdl_config.ini')
    main(args, config)
