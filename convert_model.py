# tensorflow is not supported on raspberry pi 
# tensorflow-list is being used

# need to convert the model from tensorflow version to tflite version

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("weights/traffic.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("weights/traffic.tflite", "wb") as f:
    f.write(tflite_model)