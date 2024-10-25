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

## Instructions for training and running font classification model

### 1. Generate captchas
I recommend generating 50,000 captchas for each font (100,000 total). Each captcha should have a width of 192 and a height of 96, and captchas should be generated using a random number of characters from 1 to 6 (to most accurately represent the data provided by Ciaran). **Note: ensure you include the '--vary-char-size' and do not modify width, height or length.**
```
python3 generate.py --width=192 --height=96 --font=patrick-files/fonts/WildCrazy.ttf --output-dir=patrick-files/training-data/WildCrazy --symbols=symbols.txt --count=50000 --length=6 --vary-char-size
```
**Run the above command twice (once for each font) and ensure you name the output directory as the font name **

### 2. Train model
I have created a Convolutional Neural Network with preselected hyperparameters (please note these parameters can be changed by being passed as arguments in the command line, but they have my default values). You must pass in a directory that contains two subdirectories in which each of these directories contains examples of each font. The names of these subdirectories should be the respective font names. Note that if you use the generate method in part 1, this should be done automatically. 

```
python3 train_font.py --data_dir=patrick-files/training-data/ 
```
**Note: You can specify the hyperparameters if you would like, but I have set default values that I used for my model. The optional commands are: **

'--batch_size', type=int, default=64, help='Batch size for training'
'--img_height', type=int, default=96, help='Height of the input images'
'--img_width', type=int, default=192, help='Width of the input images'
'--num_classes', type=int, default=2, help='Number of classes in the dataset'
'--epochs', type=int, default=50, help='Number of epochs for training'


### 3. Classify captchas by font 
Once you have trained your model, you can call the classify_captchas_font.py class and pass in the directory of your captchas. This will generate a new directory that will contain two sub-directories (one for each of your captchas categorised by font). You must also include the class name for each font; this will be the name of each font. I have also put a print statement in the training model so that you can identify the order of the fonts/ their respective names. These names should just be the names of the respective directories containing the font training data.  

**Note you may optionally include the '--model_path' parameter, which allows you to specify the path to the model you want to use. This is not a required parameter as the default output file from the train_font.py execution will be used. 

```
python3 classify_captchas_font.py --input_dir=patrick-files/captchas --output_dir=patrick-files/captcha-categorized-by-font --class-name-1=WildCrazy --class-name-2=cRAZYsTYLE
```




## Instructions for identifying the ideal values for identifying overlapping characters
One issue with segmentation is that if characters are touching, they will be categorised as one character. The preprocessing code handles this by defining a width value that identifies a given segmented character as actually containing 2, 3 or 4 overlapping characters (we can then attempt to split these characters up). To identify the ideal values, you can run the code/commands below for each font (as the value changes per font).

**Repeat the steps below once for each font**

### 1. Generate captchas (optional)
**(this step and step 2 are optional; they give you the average character length, which can be used to inform your starting point when testing different values)**

Generate 1 character-long captcha (I used 2000 for my testing, and the average code in part 2 below is hard coded for 2000 captchas). 
```
python3 generate.py --width=192 --height=96 --font=patrick-files/fonts/WildCrazy.ttf --output-dir=patrick-files/training-data/WildCrazy/1-char-captchas --symbols=symbols.txt --count=2000 --length=1
```

### 2. Calculate average character length (optional)
**(this is optional and only required if you carry out the above step as well)**
Once you have your 2000 captchas, you can call the code below, passing in the directory containing the 2000 1-character captchas to the code below. The average character length will be displayed to the console
```
python3 get_char_average_size.py --captcha-dir=/location-of-1-char-captchas
```
**All steps below this are mandatory**

### 3. Generate captchas (these captchas will be used to test the accuracy of the segmentation)
You will now generate 2000 captchas of varying character amounts from 2 - 6 (the arguments in the below command are set to do this already)

Ensure that the count is set to 2000, length is 6 and --vary-char-size is present. Also, I recommend not changing the height or width specified below as this matches Kieran's captchas.
```
python3 generate.py --width=192 --height=96 --font=patrick-files/fonts/WildCrazy.ttf --output-dir=dir-for-output-captchas --symbols=symbols.txt --count=2000 --length=6 --vary-char-size
```

### 4. Clean the 2000 captchas generated above
We now clean the captchas once so that we can reuse these captchas without having to clean them each time (this allows us to run tests pretty quickly).
```
python3 generate_2000_cleaned_images.py --captcha-dir=location-of-captchas-generated-above --output=location-for-cleaned-files
```

### 5. Run Tests
Now that we have our 2000 cleaned files, we can run some tests to check a wide range of values and see how many captchas have been segmented with the correct number of characters. We can then pick the best-performing values for each. 

You can use the average char size as a good starting point for each value (i.e. set the value for 2 characters overlapping at values around average character length * 2, for three: average char length *3, etc.

The below code is useful for testing a wide range of values; you can vary the ranges for each as you see fit; see line 226 and below for the ranges. You can specify the range of each min value you would like to test along with the step size. The model will test the combination of all these ranges and return the top 10 best-performing values. You can then rerun with different values if you would like until you get a score you are happy with. 

The results will be printed to the console.

```
python3 wider_range_test.py --captcha-dir=location-of-cleaned-captchas
```




If you want to choose a smaller, more specific range of values, you can use the code below and hard code an array of values (see lines 231 - 262 for examples). Each index in each of the arrays will be tested so all the values in ind 0 of each array will be set as the min value for each character (again you must hard code these). I would recommend running this when you have a good idea of the range of values you want to test and want to check specific/more granular values. 

```
python3 best_segment_range.py --captcha-dir=location-of-cleaned-captchas
```











