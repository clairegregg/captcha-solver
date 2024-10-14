# Claire's Project Notes
This will be used to track my approach to this project, and what work I'm doing on what devices (etc.)

## Retriving the files

I retrieved the initial file list from the Pi using
```
wget --wait=30 "https://cs7ns1.scss.tcd.ie/?shortname=cgregg"
```

This included a wait to ensure the device was not overloaded if retries were necessary.

With this file, I created a python script to retrieve all of the captchas, using python library `urllib3` which was already installed on the pi. This script loops through all of the files described in the file list, and requests them (with a randomised 1-5s delay between requests to prevent throttling).

## Splitting the captcha to help identify characters

At this point, I was blocked from continuing on the pi as I lost access, so I did this step on my laptop. Followed help from https://stackoverflow.com/questions/42502176/opencv-extract-letters-from-string-using-python