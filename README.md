# Scalable Computing - Captcha Solver
Completed for CS7NS1, Scalable Computing by Claire Gregg and Patrick Haughey.

## How to use this Repo

First, this assumes you have the latest version of tensorflow and ai_edge_litert installed on your workstation, and the latest version of tflite-runtime installed on your Raspberry Pi.

### 1. Retrieve Captchas

(Assuming you are part of CS7NS1 - scalable computing)

This step should be completed on your Raspberry Pi.

Captchas can be retrieved from the website using `retrieve_captchas.py`, which requires:
- The path to the file where the list of files to download are stored.
- Your TCD shortname (eg cgregg).
- Directory to store captchas in.

Example command
```
python3 retrieve_captchas.py --list-path=user-files/captchas --shortname=user --output=user-files/file-list.txt
```

### 2. Generate Datasets

Once you have your font and symbol set identified, you should start generating training/validation datasets. You need to generate 4 datasets/

#### 2.1. Data to find average character size

Repeat this for each font. Generate 2000 1 character-long captchas.
```
python3 generate.py --width=192 --height=96 --font=user-files/fonts/FontName.ttf --output-dir=user-files/training-data/FontName/1-char-captchas --symbols=symbols.txt --count=2000 --length=1
```

#### 2.2. Data to find where intersecting characters should be split

Repeat this for each font. You will now generate 2000 captchas of varying character amounts from 2 - 6 (the arguments in the below command are set to do this already)

Ensure that the count is set to 2000, length is 6 and --vary-char-size is present. Also, I recommend not changing the height or width specified below as this matches Kieran's captchas.
```
python3 generate.py --width=192 --height=96 --font=user-files/fonts/FontName.ttf --output-dir=training-data/FontName/varying-char-captchas --symbols=symbols.txt --count=2000 --length=6 --vary-char-size
```

#### 2.3 Data to train font classification

I recommend generating 50,000 captchas for each font (100,000 total). Each captcha should have a width of 192 and a height of 96, and captchas should be generated using a random number of characters from 1 to 6 (to most accurately represent the data provided by Ciaran). **Note: ensure you include the '--vary-char-size' and do not modify width, height or length.**
```
python3 generate.py --width=192 --height=96 --font=user-files/fonts/FontName.ttf --output-dir=user-files/font-training-data/FontName --symbols=symbols.txt --count=50000 --length=6 --vary-char-size
```
**Run the above command twice (once for each font) and ensure you name the output directory as the font name**

#### 2.4 Data to train character classifier

Repeat this for each font. This should generate 10000 training images of size 100x100 containing one character each.

```
python3 generate.py --width=100 --height=100 --length=1 --count=10000 --output-dir=user-files/training-data/FontName/char-training --symbols=symbols.txt --font="user-files/fonts/FontName.ttf"
```

The character classifier also requires a validation dataset, which we recommend to be 1000 files large:

```
python3 generate.py --width=100 --height=100 --length=1 --count=1000 --output-dir=user-files/validation-data/FontName --symbols=symbols.txt --font="user-files/fonts/FontName.ttf"
```

### 3. Identify how each font's overlapping characters should be split

#### 3.1. Identify the font's average character size

Repeat this for each font. This can be done with the following command, assuming you have generated datasets correctly. The average character length will be displayed to the console

```
python3 Character-Segmentation-testing/get_char_average_size.py --captcha-dir=user-files/training-data/FontName/1-char-captchas
```

#### 3.2. Clean the 2000 varying length captchas 

Repeat this for each font. We now clean the captchas once so that we can reuse these captchas without having to clean them each time (this allows us to run tests pretty quickly).
```
python3 Character-Segmentation-testing/generate_2000_cleaned_images.py --captcha-dir=user-files/training-data/FontName/varying-char-captchas --output=user-files/training-data/FontName/clean-varying-char-captchas
```

#### 3.3. Run Tests

Repeat this for each font. Now that we have our 2000 cleaned files, we can run some tests to check a wide range of values and see how many captchas have been segmented with the correct number of characters. We can then pick the best-performing values for each. 

You can use the average char size as a good starting point for each value (i.e. set the value for 2 characters overlapping at values around average character length * 2, for three: average char length *3, etc.

The below code is useful for testing a wide range of values; you can vary the ranges for each as you see fit; see line 226 and below for the ranges. You can specify the range of each min value you would like to test along with the step size. The model will test the combination of all these ranges and return the top 10 best-performing values. You can then rerun with different values if you would like until you get a score you are happy with. 

The results will be printed to the console.

```
python3 Character-Segmentation-testing/wider_range_test.py --captcha-dir=user-files/training-data/FontName/clean-varying-char-captchas
```

If you want to choose a smaller, more specific range of values, you can use the code below and hard code an array of values (see lines 231 - 262 for examples). Each index in each of the arrays will be tested so all the values in ind 0 of each array will be set as the min value for each character (again you must hard code these). I would recommend running this when you have a good idea of the range of values you want to test and want to check specific/more granular values. 

```
python3 Character-Segmentation-testing/best_segment_range.py --captcha-dir=user-files/training-data/FontName/clean-varying-char-captchas
```

### 4. Train the font identification model
This trains for both fonts simultaneously. I have created a Convolutional Neural Network with preselected hyperparameters (please note these parameters can be changed by being passed as arguments in the command line, but they have my default values). You must pass in a directory that contains two subdirectories in which each of these directories contains examples of each font. The names of these subdirectories should be the respective font names. Note that if you use follow the instructions above, this should be done automatically. 
```
python3 train_font.py --data_dir=user-files/font-training-data --output=user-files/font-identification-model
```
**Note: You can specify the hyperparameters if you would like, but I have set default values that I used for my model. The optional commands are: **

'--batch_size', type=int, default=64, help='Batch size for training'
'--img_height', type=int, default=96, help='Height of the input images'
'--img_width', type=int, default=192, help='Width of the input images'
'--num_classes', type=int, default=2, help='Number of classes in the dataset'
'--epochs', type=int, default=50, help='Number of epochs for training'

### 5. Train the character identification model
Repeat for each font. Feel free to mess around with parameters here, except for width and height.

```
python3 train.py --width=100 --height=100 --batch-size=32 --train-dataset=user-files/training-data/FontName/char-training --validate-dataset=user-files/validation-data/FontName --epochs=100 --symbols=symbols.txt --output-model-name=user-files/font-name-model
```

### 6. Convert models to LiteRT

The font identification model, and both character identification models should have been saved as .keras models. To convert these to tflite, run the following set of commands

```
python3 convert_keras.py --model=user-files/font-identification-model
```
```
python3 convert_keras.py --model=user-files/font-name-1-model
```
```
python3 convert_keras.py --model=user-files/font-name-2-model
```

### 7. Inference

This step can be run on either your 64-bit workstation or (preferably) on your 32-bit Pi. If using your workstation, replace any mentions of 32bit in file names with 64, and everything should work the same.

#### 7.1. Classify captchas based on fonts

Once you have trained your model, you can call the classify_font_32bit.py class and pass in the directory of your captchas. This will generate a new directory that will contain two sub-directories (one for each of your captchas categorised by font). You must also include the class name for each font; this will be the name of each font. I have also put a print statement in the training model so that you can identify the order of the fonts/ their respective names. These names should just be the names of the respective directories containing the font training data.  

```
python3 classify_font_32bit.py --input_dir=user-files/captchas --output_dir=user-files/captcha-categorised --class-name-1=FontName1 --class-name-2=FontName2 --model=user-files/font-identification-model.tflite
```

#### 7.2. Solve the Captchas

Repeat this for each font. This will generate an output file with each captcha filename followed by its solution.
```
python3 classify_32bit.py --captcha-dir=user-files/captchas-categorised/FontName --output=user-files/FontNameOutput.csv --symbols=symbols.txt --username=user --model=user-files/font-name-model.tflite --verbose=true
```

### 8. Complete your solution

The last step required is to combine the two solutions provided by the seperate font's character classifiers.

```
python3 format_submission.py FontName1Out.csv FontName2Out.csv output.csv
```
