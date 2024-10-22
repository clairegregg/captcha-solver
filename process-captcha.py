import argparse
import os
import cv2
import numpy as np
import scipy.ndimage

def remove_circles(img):
    hough_circle_locations = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1 = 50, param2 = 5, minRadius = 0, maxRadius = 2)
    if hough_circle_locations is not None:
        circles = hough_circle_locations[0]
        for circle in circles:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            img = cv2.circle(img, center = (x, y), radius = r, color = (255), thickness = 2)
    return img

def remove_noise(img_path, display):
    img = cv2.imread(img_path)

    # 1. Shift image colour - to greyscale, then binary, then inverted
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    img = ~img # White letters, black background
    if display:
        cv2.imshow("Binary", img) 
        cv2.waitKey()

    # 2. Initial noise removal - erosion, then scipy median filtering to remove lines and circles
    img = cv2.erode(img, np.ones((2,2), np.uint8), iterations=1)
    img = ~img # Black letters, white background
    img = scipy.ndimage.median_filter(img, (5,1)) # Remove lateral lines
    img = scipy.ndimage.median_filter(img, (1,3)) # Remove circles
    img = cv2.erode(img, np.ones((2,2), np.uint8), iterations=1) # Dilate image (inverted) to original level
    img = scipy.ndimage.median_filter(img, (3, 3)) # Remove any weak noise
    if display:
        cv2.imshow("Initial cleanup", img) 
        cv2.waitKey()

    # 3. Remove circles from image
    img = remove_circles(img)
    if display:
        cv2.imshow("Circles removed", img) 
        cv2.waitKey()

    # 4. Final cleanup
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # Erosion for cleanup
    img = scipy.ndimage.median_filter(img, (5, 1)) # Remove any extra noise
    img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations = 2) # Dilate image to make it look like the original
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # Erode for final cleanup
    if display:
        cv2.imshow("Final", img)
        cv2.waitKey()

    return img
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas', type=str)
    parser.add_argument('--output', help='File where the split captchas should be stored', type=str)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to split")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output files")
        exit(1)

    x = os.listdir(args.captcha_dir)[0]
    for x in os.listdir(args.captcha_dir):
    # if True:
        img = remove_noise(os.path.join(args.captcha_dir, x), False)
        cv2.imshow("Cleaned up", img)
        cv2.waitKey()


if __name__ == "__main__":
    main()