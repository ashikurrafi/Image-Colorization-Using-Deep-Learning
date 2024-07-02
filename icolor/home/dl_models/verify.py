
import os
from keras.models import model_from_json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_json_path = os.path.join(base_dir, 'home', 'dl_models', 'model.json')
model_weights_path = os.path.join(base_dir, 'home', 'dl_models', 'model.h5')

json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weights_path)
loaded_model.summary()


print(loaded_model_json)