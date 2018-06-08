from keras.models import model_from_json

print("Loaded data file")

json_file = open('/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/prototype_networks/MeanPSD_Adamax_1_2_1527865687.nn', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/media/jonathan/ea2eea90-b89c-4e24-b854-05970b317ba4/prototype_networks/MeanPSD_Adamax_1_2_1527865687.h5")
print("Loaded model from disk")

print(loaded_model.summary())


