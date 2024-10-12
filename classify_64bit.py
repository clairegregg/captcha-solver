#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import argparse
from ai_edge_litert.interpreter import Interpreter

TFLITE_FILE_PATH = 'model.tflite'

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    return characters[y[0]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--verbose', default=False, type=bool)
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

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    if args.verbose:
        print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    interpreter = Interpreter(TFLITE_FILE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output = []

    for x in os.listdir(args.captcha_dir):
        # load image and preprocess it
        raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
        rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
        image = numpy.array(rgb_data, dtype=numpy.float32) / 255.0
        (c, h, w) = image.shape
        image = image.reshape([-1, c, h, w])

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        prediction=[None]*5
        for ind, detail in enumerate(output_details):
            char_pred = interpreter.get_tensor(output_details[ind]['index'])
            char_placement = int(detail["name"][-1]) # name takes the form StatefulPartitionedCall:0 (up to 4)
            prediction[char_placement] = decode(captcha_symbols, char_pred)
        prediction = ''.join(prediction)
        output.append(x + "," + prediction + "\n")

        if args.verbose:
            print('Classified ' + x)

    output.sort()

    with open(args.output, 'w') as output_file:
        output_file.write("cgregg \n")
        for line in output:
            output_file.write(line)

if __name__ == '__main__':
    main()
