import argparse
import os
import cv2
import numpy as np

def remove_circles(img):
    hough_circle_locations = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=5, minRadius=0, maxRadius=2)
    if hough_circle_locations is not None:
        circles = hough_circle_locations[0]
        for circle in circles:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            img = cv2.circle(img, center=(x, y), radius=r, color=(255), thickness=2)
    return img

def median_blur_rectangular(image, k_height, k_width):
    # Pad the image to handle borders
    padded_image = cv2.copyMakeBorder(image, k_height//2, k_height//2, k_width//2, k_width//2, cv2.BORDER_REFLECT, value=0)
    
    # Create an empty output image
    output = np.zeros_like(image)
    
    # Iterate over each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract the neighborhood (window)
            window = padded_image[y:y + k_height, x:x + k_width]
            
            # Compute the median (for binary, it's just majority vote in the window)
            output[y, x] = np.median(window)
    return output

def remove_noise(orig_img, display):
    # 1. Shift image colour - to greyscale, then binary, then inverted
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img = median_blur_rectangular(img, 5, 1)
    img = median_blur_rectangular(img, 1, 3)
    img = remove_circles(img)  
    
    # Find all connected components (with statistics)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)

    # Define a minimum size threshold for components (in pixels)
    min_size = 100

    # Create output image with white background
    output_image = np.full(img.shape, 255, dtype=np.uint8)

    # Set large components to black
    for i in range(1, num_labels):  # Skip label 0 for the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_image[labels == i] = 0  # Set large components to black

    # Display images if needed
    if display:
        cv2.imshow('Original Image', orig_img)
        cv2.imshow('Processed Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_image


def preprocess(img):
    img = remove_noise(img, False)
    chars = segment(img, visualize=False)
    return chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas', type=str)
    parser.add_argument('--output', help='Directory where the segmented captchas should be stored', type=str)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to split")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output files")
        exit(1)

    # Create directories if they don't exist
    os.makedirs(args.output, exist_ok=True)
    # cleaned_dir = 'patrick-files/cRAZYsTYLE/cleaned-files'
    # cleaned_dir = 'patrick-files/WildCrazy/cleaned-files'
    # os.makedirs(cleaned_dir, exist_ok=True)

    # Iterate over all images in the captcha directory
    # for filename in os.listdir(args.captcha_dir):
    for idx, filename in enumerate(os.listdir(args.captcha_dir)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(args.captcha_dir, filename)
            img = cv2.imread(img_path)

            # Clean the image
            clean = remove_noise(img, False)

            # Save the cleaned image in 'patrick-files/cleaned-files'
            # clean_filename = f"{os.path.splitext(filename)[0]}.png"
            cleaned_dir = args.output
            clean_output_path = os.path.join(cleaned_dir, filename)
            cv2.imwrite(clean_output_path, clean)
        if idx % 100 == 0:
            print(f'progess: {idx}/2000')


if __name__ == "__main__":
    main()
