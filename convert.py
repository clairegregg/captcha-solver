# Convert an existing tensorflow model to a LiteRT model
import tensorflow as tf
import keras
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='Name of pretrained tensorflow model stored in h5 format (eg if you have test.json and test.h5, set this to "test")', type=str)
  args = parser.parse_args()
  if args.model is None:
    print("Please provide model name")
    exit(1)
  
  # Load in pretrained model.
  json_file = open(args.model+".json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = keras.models.model_from_json(loaded_model_json)
  model.load_weights(args.model+'.h5')
  model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                  metrics=['accuracy'])
  
  # Convert to LiteRT (previously known as TFLite) model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model.
  with open('finalmodel.tflite', 'wb') as f:
    f.write(tflite_model)