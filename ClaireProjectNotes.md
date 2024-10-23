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
