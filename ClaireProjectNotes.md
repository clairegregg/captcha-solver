# Claire's Project Notes
This will be used to track my approach to this project, and what work I'm doing on what devices (etc.)

## Retriving the files

I retrieved the initial file list from the Pi using
```
wget --wait=30 "https://cs7ns1.scss.tcd.ie/?shortname=cgregg"
```

This included a wait to ensure the device was not overloaded if retries were necessary.

With this file, I created a python script to retrieve all of the captchas, using python library `urllib3` which was already installed on the pi. This script loops through all of the files described in the file list, and requests them (with a randomised 1-5s delay between requests to prevent throttling).

## Removing noise

At this point, I was blocked from continuing on the pi as I lost access, so I did this step on my laptop. 

After a lot of difficulty, I found https://medium.com/analytics-vidhya/solving-noisy-text-captchas-126734c3c717 which I decided to follow instead. 

This code makes use of the scipy.ndimage, which is not supported on a 32-bit architecture, which this project should be deployed to. However, opencv (which does work on a 32 bit architecture) does not support rectangular inputs to perform a median filter, which is used to remove noise from captchas. I used ChatGPT to generate code to perform a median filter with a rectangular filter shape, and modified the generated code to use BORDER_REFLECT for padding instead of BORDER_CONSTANT.

## Splitting the captcha to help identify characters

Followed help from https://stackoverflow.com/questions/42502176/opencv-extract-letters-from-string-using-python - just for a basic rectangle splitting approach. This should be improved in future, but for now it's passable.

## Generating training data

Added code to generate training data - generates a 1 letter captcha, uses existing preprocessing, and writes it to folder (depending on font used).

For now the training for all fonts will be completed together, but the data is being separated based on font in case we move on in future to have font specific models.

Files generated with 
```
py generate-clean.py --font-dir=claire-files/fonts --output-dir=claire-files/training-data --symbols=symbols.txt --count=10
```

Also added a way to handle \ in images - filenames cannot contain. I replace them with ~ in image names, and will deal with this translation when training the ML model.

Also added handling for if preprocessing doesnt find any characters.

## Basic ML model

Making use of existing code, modifying to use our preprocessing and training data, and to only accept captchas of length 1.

```
py train.py --width=100 --height=100 --batch-size=32 --train-dataset=claire-files/training-files --validate-dataset=claire-files/validation-files --epochs=10 --symbols=symbols.txt --output-model-name=claire-files/model
```

## Converting Tensorflow model to TFLite

Using existing convert.py, written for the previous assignment

```
py convert.py --model=claire-files/model
```

## Classifying

Classification done with existing code, modifying to identify each character individually. This version of the code was run on my laptop for testing, but the equivalent change has been made for the raspberry pi
```
python3 classify_64bit.py --captcha-dir=claire-files/small-captchas --output=claire-files/output.csv --symbols=symbols.txt --username=cgregg --model=claire-files/model.tflite --verbose=true
```

While doing this, I noticed the model is training on "rgb" data, which is not actually being used (it's trained on preprocessed data), so this should be changed.

I fixed this in the next change, where I moved the preprocessing into the train step. Now, the model is being trained as though the image is greyscale (only 1 channel) - there could be more work here to change the model to be trained based on a binary (b/w) image.

## Splitting captchas by font

Done on the raspbery pi
python3 classify_font_32bit.py --model_path=claire-files/font_classifier.tflite --input_dir=claire-files/captchas --output_dir=claire-files/captchas-categorised --class-name-1=DreamingofLilian --class-name-2=TheJjester

## Finding appropriate symbol widths

### DOL
```
python3 generate.py --width=192 --height=96 --font="claire-files/fonts/Dreaming of Lilian.ttf" --output-dir=claire-files/width-based-data/DreamingOfLilian --symbols=symbols.txt --count=2000 --length=1
```

Then
```
python3 get_char_average_size.py --captcha-dir=../claire-files/width-based-data/DreamingOfLilian
```

Giving 46.918 as average

Next

```
python3 generate.py --width=192 --height=96 --font="claire-files/fonts/Dreaming of Lilian.ttf" --output-dir=claire-files/width-varying-data/DreamingOfLilian --symbols=symbols.txt --count=2000 --length=6 --vary-char-size
```

```
python3 generate_2000_cleaned_images.py --captcha-dir=../claire-files/width-varying-data/DreamingOfLilian --output=../claire-files/clean/DreamingOfLilian
```

And finally, setting the ranges to 2chars (70, 110), 3 chars (115, 155), and 4 chars (160, 200), with steps of 10 for all.
```
python3 wider_range_test.py --captcha-dir=../claire-files/clean/DreamingOfLilian
```

Returned
```
1. Accuracy: 42.70%, two_char_min: 70, three_char_min: 135, four_char_min: 190
2. Accuracy: 42.65%, two_char_min: 70, three_char_min: 125, four_char_min: 190
3. Accuracy: 42.60%, two_char_min: 70, three_char_min: 135, four_char_min: 180
4. Accuracy: 42.55%, two_char_min: 70, three_char_min: 125, four_char_min: 180
5. Accuracy: 42.40%, two_char_min: 70, three_char_min: 115, four_char_min: 190
6. Accuracy: 42.35%, two_char_min: 70, three_char_min: 135, four_char_min: 170
7. Accuracy: 42.35%, two_char_min: 70, three_char_min: 145, four_char_min: 190
8. Accuracy: 42.30%, two_char_min: 70, three_char_min: 115, four_char_min: 180
9. Accuracy: 42.30%, two_char_min: 70, three_char_min: 125, four_char_min: 170
10. Accuracy: 42.30%, two_char_min: 70, three_char_min: 135, four_char_min: 160
```

Then did (50, 70), (115, 135), (160,180)
```
1. Accuracy: 42.70%, two_char_min: 60, three_char_min: 125, four_char_min: 170
2. Accuracy: 42.65%, two_char_min: 60, three_char_min: 125, four_char_min: 160
3. Accuracy: 42.45%, two_char_min: 60, three_char_min: 115, four_char_min: 170
4. Accuracy: 42.40%, two_char_min: 60, three_char_min: 115, four_char_min: 160
5. Accuracy: 41.80%, two_char_min: 50, three_char_min: 125, four_char_min: 170
6. Accuracy: 41.75%, two_char_min: 50, three_char_min: 125, four_char_min: 160
7. Accuracy: 41.65%, two_char_min: 50, three_char_min: 115, four_char_min: 170
8. Accuracy: 41.60%, two_char_min: 50, three_char_min: 115, four_char_min: 160
```

Finally, with steps of 1 (57,63), (122,128), (167,173) giving final best values of 57, 123, and 168.


### JJ

```
python3 generate.py --width=192 --height=96 --font="claire-files/fonts/The Jjester.otf" --output-dir=claire-files/width-based-data/TheJjester --symbols=symbols.txt --count=2000 --length=1
```

Getting 20.263

```
python3 get_char_average_size.py --captcha-dir=../claire-files/width-based-data/TheJjester
```

```
python3 generate.py --width=192 --height=96 --font="claire-files/fonts/The Jjester.otf" --output-dir=claire-files/width-varying-data/TheJjester --symbols=symbols.txt --count=2000 --length=6 --vary-char-size
```

```
python3 generate_2000_cleaned_images.py --captcha-dir=../claire-files/width-varying-data/TheJjester --output=../claire-files/clean/TheJjester
```

And finally, setting the regions to 2 chars (30,50), 3 chars (50,70), 4 chars (70,90) with steps of 5 for all.
```
python3 wider_range_test.py --captcha-dir=../claire-files/clean/TheJjester
```

Returned
```
Top 10 combinations:
1. Accuracy: 48.75%, two_char_min: 30, three_char_min: 50, four_char_min: 70
2. Accuracy: 48.75%, two_char_min: 30, three_char_min: 50, four_char_min: 75
3. Accuracy: 48.25%, two_char_min: 30, three_char_min: 50, four_char_min: 80
4. Accuracy: 47.90%, two_char_min: 30, three_char_min: 50, four_char_min: 85
5. Accuracy: 47.70%, two_char_min: 30, three_char_min: 55, four_char_min: 70
6. Accuracy: 47.60%, two_char_min: 30, three_char_min: 55, four_char_min: 75
7. Accuracy: 47.10%, two_char_min: 30, three_char_min: 55, four_char_min: 80
8. Accuracy: 46.75%, two_char_min: 30, three_char_min: 55, four_char_min: 85
9. Accuracy: 46.60%, two_char_min: 30, three_char_min: 60, four_char_min: 70
10. Accuracy: 46.50%, two_char_min: 30, three_char_min: 60, four_char_min: 75
```
Then (20,25), (45,50), (65,70) giving final best values of 22, 48, 67.

## Training
```
python3 train.py --width=100 --height=100 --batch-size=64 --train-dataset=claire-files/training/DreamingofLilian --validate-dataset=claire-files/validate/DreamingofLilian --epochs=100 --symbols=symbols.txt --output-model-name=claire-files/dol --min-two-char=57 --min-three-char=123 --min-four-char=168
```

```
python3 train.py --width=100 --height=100 --batch-size=128 --train-dataset=claire-files/training/TheJjester --validate-dataset=claire-files/validate/TheJjester --epochs=100 --symbols=symbols.txt --output-model-name=claire-files/jj --min-two-char=22 --min-three-char=48 --min-four-char=67
```

## Classifying

```
python3 classify_64bit.py --captcha-dir=claire-files/captchas-categorised/DreamingofLilian --output=dol-out.csv --symbols=symbols.txt --username=cgregg --model=claire-files/dol.tflite --verbose=true --min-two-char=57 --min-three-char=123 --min-four-char=168
```

```
python3 classify_64bit.py --captcha-dir=claire-files/captchas-categorised/TheJjester --output=jj-out.csv --symbols=symbols.txt --username=cgregg --model=claire-files/jj.tflite --verbose=true --min-two-char=22 --min-three-char=48 --min-four-char=67
```