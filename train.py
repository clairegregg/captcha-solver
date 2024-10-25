# Initial code provided by Ciaran McGoldrick
# Modified to train on 1 character captchas with preprocessing performed on them
#!/usr/bin/env python3

import tensorflow.keras as keras
import tensorflow as tf
import argparse
import random
import numpy
import cv2
import os
from pathlib import Path
import warnings
import preprocess_testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Build a Keras model given some parameters

def create_model(captcha_num_symbols, input_shape, model_depth=3, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(
                32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    # A single dense layer for one-character output
    x = keras.layers.Dense(captcha_num_symbols, activation='softmax', name='output')(x)
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model


# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size


class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_symbols, captcha_width, captcha_height, min_two_char, min_three_char, min_four_char):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        self.min_two_char = min_two_char
        self.min_three_char = min_three_char
        self.min_four_char = min_four_char

        file_list = [str(file) for file in Path(self.directory_name).rglob('*') if file.is_file()] # Retrieve files recursively
        self.files = dict(
            zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height,
                        self.captcha_width, 1), dtype=numpy.float32)
        y = numpy.zeros((self.batch_size, len(self.captcha_symbols)),
                         dtype=numpy.uint8)

        for i in range(self.batch_size):
            if (len(self.files) == 0):
                break
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is read in as 8-bit RGB
            raw_data = cv2.imread(random_image_file)
            chars = preprocess_testing.preprocess(raw_data, two_char_min=self.min_two_char, three_char_min=self.min_three_char, four_char_min=self.min_four_char)
            if len(chars) != 1: # If 0 chars are identified, segmentation failed 
                break
            preprocessed_data = chars[0]
            processed_data = numpy.array(preprocessed_data) / 255
            processed_data = numpy.expand_dims(processed_data, axis=-1)
            X[i] = processed_data

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.
            random_image_label = random_image_label.split('_')[0]
            random_image_label = random_image_label.replace("~", "\\") # Replace backslashes in file names
            random_image_label = random_image_label[-1] # Retrieve the one character file name

            # One hot encode the character
            y[i, :] = 0
            y[i, self.captcha_symbols.find(random_image_label)] = 1

        return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument(
        '--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument(
        '--train-dataset', help='Where to look for the training image dataset', type=str)
    parser.add_argument(
        '--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name',
                        help='Where to save the trained model', type=str)
    parser.add_argument(
        '--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument(
        '--epochs', help='How many training epochs to run', type=int)
    parser.add_argument(
        '--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument(
        '--min-two-char', help='Mininum width where chracter is deemed as being two overlapping characters', type=int, default=40) 
    parser.add_argument(
        '--min-three-char', help='Mininum width where chracter is deemed as being three overlapping characters', type=int, default=60)
    parser.add_argument(
        '--min-four-char', help='Mininum width where chracter is deemed as being four overlapping characters', type=int, default=80)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "No GPU available!"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    with tf.device('/device:GPU:0'):
    # with tf.device('/device:CPU:0'):
        # with tf.device('/device:XLA_CPU:0'):
        model = create_model(len(captcha_symbols), (args.height, args.width, 1))

        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        # training_data = ImageSequence(
        #     args.train_dataset, args.batch_size, captcha_symbols, args.width, args.height)
        # validation_data = ImageSequence(
        #     args.validate_dataset, args.batch_size, captcha_symbols, args.width, args.height)
        
        training_data = ImageSequence(
            args.train_dataset, args.batch_size, captcha_symbols, args.width, args.height,
            args.min_two_char, args.min_three_char, args.min_four_char
        )
        validation_data = ImageSequence(
            args.validate_dataset, args.batch_size, captcha_symbols, args.width, args.height,
            args.min_two_char, args.min_three_char, args.min_four_char
        )


        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.keras', save_best_only=False)]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(training_data,
          validation_data=validation_data,
          epochs=args.epochs,
          callbacks=callbacks)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' +
                  args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')


if __name__ == '__main__':
    main()
