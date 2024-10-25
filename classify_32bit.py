#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import argparse
import tflite_runtime.interpreter as tflite
import preprocess_testing

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    return characters[y[0]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--username', help="TCD username to put in the output file", type=str)
    parser.add_argument('--model', help="Path to saved TFLite model")
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--min-two-char', help='Mininum width where chracter is deemed as being two overlapping characters', type=int, default=40) 
    parser.add_argument('--min-three-char', help='Mininum width where chracter is deemed as being three overlapping characters', type=int, default=60)
    parser.add_argument('--min-four-char', help='Mininum width where chracter is deemed as being four overlapping characters', type=int, default=80)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.username is None:
        print("Please specify the the TCD username to put in the output file")
        exit(1)

    if args.model is None:
        print("Please specify the path to the saved model")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    if args.verbose:
        print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    interpreter = tflite.Interpreter(args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output = []

    for x in os.listdir(args.captcha_dir):
        # load image and preprocess it
        raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
        rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
        preprocessed_characters = preprocess_testing.preprocess(rgb_data, args.min_two_char, args.min_three_char, args.min_four_char)
        prediction = []

        for char in preprocessed_characters:
            image = numpy.array(char, dtype=numpy.float32) / 255.0
            image = numpy.expand_dims(image, axis=-1)

            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            char_pred = interpreter.get_tensor(output_details[0]['index'])
            pred = decode(captcha_symbols, char_pred)
            prediction.append(pred)

        prediction = ''.join(prediction)
        output.append(x + "," + prediction + "\n")

        if args.verbose:
            print('Classified ' + x)

    output.sort()

    with open(args.output, 'w') as output_file:
        output_file.write(args.username + " \n")
        for line in output:
            output_file.write(line)

if __name__ == '__main__':
    main()
