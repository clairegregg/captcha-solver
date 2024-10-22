import argparse
import os
import cv2
import numpy as np

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
    #for x in os.listdir(args.captcha_dir):
    if True:
        img = cv2.imread(os.path.join(args.captcha_dir, x))

        # 1. Upscale the image
        (h, w) = img.shape[:2]
        upscaled_points = (w*5, h*5)
        upscaled = cv2.resize(img, upscaled_points, interpolation=cv2.INTER_LINEAR)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) 

        # 3. Dilate then erode the images to remove noise
        kernel = np.ones((13, 13), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        cv2.imshow("Dilated", dilated) 
        cv2.waitKey()
        eroded = cv2.erode(dilated, kernel, iterations=1)
        cv2.imshow("Eroded", eroded) 
        cv2.waitKey()
        
        # 4. Use mser region detection to find detected characters
        (h, w) = upscaled.shape[:2]
        image_size = h*w
        mser = cv2.MSER_create()
        mser.setMaxArea(int(image_size/10))
        mser.setMinArea(int(image_size/100))
        _, rects = mser.detectRegions(eroded)

        for (x, y, w, h) in rects:
            cv2.rectangle(upscaled, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        
        cv2.imshow("Bounded", upscaled) 
        cv2.waitKey()
        # cv2.imwrite(os.path.join(
        #         args.captcha_dir, 'test'  + '.png'), cv2.GaussianBlur(src=edges, ksize=(5, 5), sigmaX=0.5))


if __name__ == "__main__":
    main()