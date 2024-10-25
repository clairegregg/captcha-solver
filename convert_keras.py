import argparse

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--model', help='Name of pretrained tensorflow model stored in keras format (eg if you have test.keras, set this to "test")', type=str)
   args = parser.parse_args()

   if args.model is None:
    print("Please provide model name")
    exit(1)

    model = keras.models.load_model(args.model+'.keras')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    filename = args.model+'.tflite'

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
  main()