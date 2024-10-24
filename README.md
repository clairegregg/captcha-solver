# Scalable Computing - Captcha Solver
Completed for CS7NS1, Scalable Computing by Claire Gregg and Patrick Haughey.

## Instructions to Run Model

### 1. Generate One-Character Test Captchas 

**Required:** Symbol set to generate captchas with, font to generate captchas with. 

Example command:
```
python3 generate.py --width=100 --height=100 --length=1 --count=10 --output-dir=claire-files/validate/TheJjester/ --symbols=symbols.txt --font="claire-files/fonts/The Jjester.otf"
```

NB: do not modify width, height, or length. Repeat this for all fonts used in your captchas, building a training and validation dataset (stored in different folders).

### 2. Train the Model

**Required:** Symbol set, training dataset, validation dataset

Example command:
```
python3 train.py --width=100 --height=100 --batch-size=32 --train-dataset=claire-files/training --validate-dataset=claire-files/validate --epochs=10 --symbols=symbols.txt --output-model-name=claire-files/model
```

NB: do not modify width or height.

### 3. Convert the Model

This model is built in Tensorflow on a 64-bit device. However, inference must be performed on a 32-bit device. For this, we use LiteRT (previously known as TFLite). A script is provided to convert the model to LiteRT.

**Requires:** A trained Tensorflow model stored as .json (for structure) and .h5 (for weights).

Example command:
```
python3 convert.py --model=claire-files/model
```

### 4. Perform Inferences

**Requires:** directory filled with captchas to crack, symbol set captchas were built with, a username to put at the start of the output file, and a model stored in file format .tflite.

Example command (for 64 bit devices - requires ai-edge-litert to be pip installed):
```
python3 classify_64bit.py --captcha-dir=claire-files/small-captchas --output=claire-files/output.csv --symbols=symbols.txt --username=cgregg --model=claire-files/model.tflite --verbose=true
```

Example command (for 32 bit devices - requires tflite_runtime to be pip installed):
```
python3 classify_32bit.py --captcha-dir=claire-files/small-captchas --output=claire-files/output.csv --symbols=symbols.txt --username=cgregg --model=claire-files/model.tflite --verbose=true
```

