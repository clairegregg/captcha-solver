import argparse
import os
import cv2
import time

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
        
        (h, w) = img.shape[:2]
        image_size = h*w
        mser = cv2.MSER_create()
        mser.setMaxArea(int(image_size/2))
        mser.setMinArea(10)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
        _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        regions, rects = mser.detectRegions(bw)

        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 255), thickness=1)
        
        cv2.imshow("Bounded", img) 
        cv2.waitKey()


if __name__ == "__main__":
    main()