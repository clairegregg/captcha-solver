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


def resize_and_center_image(img, target_size=(100, 100)):
    # Get current size of the character image
    h, w = img.shape[:2]
    
    # Create a new blank image with the target size and fill it with white (background)
    new_img = np.ones(target_size, dtype=np.uint8) * 255  # Assuming background is white
    
    # Compute the offset to center the image
    x_offset = (target_size[1] - w) // 2
    y_offset = (target_size[0] - h) // 2
    
    # Place the character image in the center of the new image
    new_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    
    return new_img

# Please note that if you set visuailize to True then you need to pass in a unique index into this function (otherwise files will get overwritten)
def segment(cleaned, two_char_min , three_char_min, four_char_min, index = 0, visualize=False, visualization_dir='visualizations'):
    os.makedirs(visualization_dir, exist_ok=True)
    
    _, thresh = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=5)
    
    # Subtract the sure foreground from the sure background to get the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), markers)
    
    # Generate character segments using contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store character images with their x-coordinates
    char_images_with_positions = []
    
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Remove characters that are likely just noise
        if w > 6 and h > 6:
            char_image = cleaned[y:y+h, x:x+w]
            if w > two_char_min and w <= three_char_min:
                # This is likely an image that has two overlapping characters
                column_sums = np.sum(char_image == 0, axis=0)
                padding_value = 0.3
                margin_length = int(char_image.shape[1] * padding_value)
                
                new_interval_start = margin_length
                new_interval_end = char_image.shape[1] - margin_length
                
                # Ensure indices are valid
                if new_interval_start >= new_interval_end:
                    new_interval_start = 0
                    new_interval_end = char_image.shape[1]
                
                # Find the optimal split column
                sub_column_sums = column_sums[new_interval_start:new_interval_end]
                divider_offset = np.argmin(sub_column_sums) + new_interval_start
                
                # Visualization
                if visualize:
                    vis_image = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)
                    h_img = char_image.shape[0]
                    # Draw margin lines (in red)
                    cv2.line(vis_image, (new_interval_start, 0), (new_interval_start, h_img), (0, 0, 255), 1)
                    cv2.line(vis_image, (new_interval_end, 0), (new_interval_end, h_img), (0, 0, 255), 1)
                    # Draw split line (in green)
                    cv2.line(vis_image, (divider_offset, 0), (divider_offset, h_img), (0, 255, 0), 1)
                    # Save the visualization image
                    vis_filename = f'visualization_{index}_{idx}.png'
                    vis_output_path = os.path.join(visualization_dir, vis_filename)
                    cv2.imwrite(vis_output_path, vis_image)
                
                # Split the image at the divider offset
                left_char = char_image[:, :divider_offset]
                right_char = char_image[:, divider_offset:]
                
                # Adjust x-coordinates for split characters
                left_x = x
                right_x = x + divider_offset
                
                # Append the resized and centered images along with their x-coordinates
                char_images_with_positions.append((left_x, resize_and_center_image(left_char)))
                char_images_with_positions.append((right_x, resize_and_center_image(right_char)))
            elif w > three_char_min and w <= four_char_min:
                # This is likely an image with three overlapping characters
                column_sums = np.sum(char_image == 0, axis=0)
                # padding_value = 0.2
                padding_value = 0.15
                w_img = char_image.shape[1]
                h_img = char_image.shape[0]
                
                # First divider
            # first_split_pos = int(w_img * 0.25)
                first_split_pos = int(w_img / 3)
                first_search_start = max(0, int(first_split_pos - w_img * padding_value))
                first_search_end = min(w_img, int(first_split_pos + w_img * padding_value))
                first_sub_column_sums = column_sums[first_search_start:first_search_end]
                first_divider_offset = np.argmin(first_sub_column_sums) + first_search_start

                # Second divider
            # second_split_pos = int(w_img * 0.75)
                second_split_pos = int(2 * w_img / 3)
                second_search_start = max(0, int(second_split_pos - w_img * padding_value))
                second_search_end = min(w_img, int(second_split_pos + w_img * padding_value))
                second_sub_column_sums = column_sums[second_search_start:second_search_end]
                second_divider_offset = np.argmin(second_sub_column_sums) + second_search_start

                # Visualization
                if visualize:
                    vis_image = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)
                    # Draw estimated split positions (yellow lines)
                    cv2.line(vis_image, (first_split_pos, 0), (first_split_pos, h_img), (0, 255, 255), 1)
                    cv2.line(vis_image, (second_split_pos, 0), (second_split_pos, h_img), (0, 255, 255), 1)
                    # Draw search areas (blue lines)
                    cv2.line(vis_image, (first_search_start, 0), (first_search_start, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (first_search_end, 0), (first_search_end, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (second_search_start, 0), (second_search_start, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (second_search_end, 0), (second_search_end, h_img), (255, 0, 0), 1)
                    # Draw split lines (green lines)
                    cv2.line(vis_image, (first_divider_offset, 0), (first_divider_offset, h_img), (0, 255, 0), 1)
                    cv2.line(vis_image, (second_divider_offset, 0), (second_divider_offset, h_img), (0, 255, 0), 1)
                    # Save the visualization image
                    vis_filename = f'visualization_{index}_{idx}.png'
                    vis_output_path = os.path.join(visualization_dir, vis_filename)
                    cv2.imwrite(vis_output_path, vis_image)
                
                # Split the image at the divider offsets
                first_char = char_image[:, :first_divider_offset]
                second_char = char_image[:, first_divider_offset:second_divider_offset]
                third_char = char_image[:, second_divider_offset:]
                
                # Adjust x-coordinates for split characters
                first_x = x
                second_x = x + first_divider_offset
                third_x = x + second_divider_offset
                
                # Append the resized and centered images along with their x-coordinates
                char_images_with_positions.append((first_x, resize_and_center_image(first_char)))
                char_images_with_positions.append((second_x, resize_and_center_image(second_char)))
                char_images_with_positions.append((third_x, resize_and_center_image(third_char)))
            
            elif w > four_char_min:
                # This is likely an image with four overlapping characters
                column_sums = np.sum(char_image == 0, axis=0)
                padding_value = 0.1
                w_img = char_image.shape[1]
                h_img = char_image.shape[0]
                
                # First divider
                first_split_pos = int(w_img / 4)
                first_search_start = max(0, int(first_split_pos - w_img * padding_value))
                first_search_end = min(w_img, int(first_split_pos + w_img * padding_value))
                first_sub_column_sums = column_sums[first_search_start:first_search_end]
                first_divider_offset = np.argmin(first_sub_column_sums) + first_search_start

                # Second divider
                second_split_pos = int(w_img / 2)
                second_search_start = max(0, int(second_split_pos - w_img * padding_value))
                second_search_end = min(w_img, int(second_split_pos + w_img * padding_value))
                second_sub_column_sums = column_sums[second_search_start:second_search_end]
                second_divider_offset = np.argmin(second_sub_column_sums) + second_search_start

                # Third divider
                third_split_pos = int(3 * w_img / 4)
                third_search_start = max(0, int(third_split_pos - w_img * padding_value))
                third_search_end = min(w_img, int(third_split_pos + w_img * padding_value))
                third_sub_column_sums = column_sums[third_search_start:third_search_end]
                third_divider_offset = np.argmin(third_sub_column_sums) + third_search_start

                # Visualization
                if visualize:
                    vis_image = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)
                    # Draw estimated split positions (yellow lines)
                    cv2.line(vis_image, (first_split_pos, 0), (first_split_pos, h_img), (0, 255, 255), 1)
                    cv2.line(vis_image, (second_split_pos, 0), (second_split_pos, h_img), (0, 255, 255), 1)
                    cv2.line(vis_image, (third_split_pos, 0), (third_split_pos, h_img), (0, 255, 255), 1)
                    # Draw search areas (blue lines)
                    cv2.line(vis_image, (first_search_start, 0), (first_search_start, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (first_search_end, 0), (first_search_end, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (second_search_start, 0), (second_search_start, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (second_search_end, 0), (second_search_end, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (third_search_start, 0), (third_search_start, h_img), (255, 0, 0), 1)
                    cv2.line(vis_image, (third_search_end, 0), (third_search_end, h_img), (255, 0, 0), 1)
                    # Draw split lines (green lines)
                    cv2.line(vis_image, (first_divider_offset, 0), (first_divider_offset, h_img), (0, 255, 0), 1)
                    cv2.line(vis_image, (second_divider_offset, 0), (second_divider_offset, h_img), (0, 255, 0), 1)
                    cv2.line(vis_image, (third_divider_offset, 0), (third_divider_offset, h_img), (0, 255, 0), 1)
                    # Save the visualization image
                    vis_filename = f'visualization_{index}_{idx}.png'
                    vis_output_path = os.path.join(visualization_dir, vis_filename)
                    cv2.imwrite(vis_output_path, vis_image)
                
                # Split the image at the divider offsets
                first_char = char_image[:, :first_divider_offset]
                second_char = char_image[:, first_divider_offset:second_divider_offset]
                third_char = char_image[:, second_divider_offset:third_divider_offset]
                fourth_char = char_image[:, third_divider_offset:]
                
                # Adjust x-coordinates for split characters
                first_x = x
                second_x = x + first_divider_offset
                third_x = x + second_divider_offset
                fourth_x = x + third_divider_offset
                
                # Append the resized and centered images along with their x-coordinates
                char_images_with_positions.append((first_x, resize_and_center_image(first_char)))
                char_images_with_positions.append((second_x, resize_and_center_image(second_char)))
                char_images_with_positions.append((third_x, resize_and_center_image(third_char)))
                char_images_with_positions.append((fourth_x, resize_and_center_image(fourth_char)))
            else:
                # For single character images
                char_images_with_positions.append((x, resize_and_center_image(char_image)))
    
    # Sort the character images based on their x-coordinate to maintain order
    char_images_with_positions.sort(key=lambda tup: tup[0])
    char_images = [img for x_pos, img in char_images_with_positions]
    
    return char_images

def preprocess(img, two_char_min=40, three_char_min=60, four_char_min=80):
    img = remove_noise(img, False)
    chars = segment(img, visualize=False, two_char_min=two_char_min, three_char_min=three_char_min, four_char_min=four_char_min)
    return chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas', type=str)
    parser.add_argument('--output', help='Directory where the segmented captchas should be stored', type=str)
    # These are parameters that should be changed based on the font. They allow you to specify the boundaries of when we classify an image as overlapping for a given number of characters
    # I have set the default values to good general ranges based on my testing 
    parser.add_argument('--min-two-char', help='Mininum width where chracter is deemed as being two overlapping characters', type=int, default=40) 
    parser.add_argument('--min-three-char', help='Mininum width where chracter is deemed as being three overlapping characters', type=int, default=60)
    parser.add_argument('--min-four-char', help='Mininum width where chracter is deemed as being four overlapping characters', type=int, default=80)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to split")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output files")
        exit(1)

    # Create directories if they don't exist
    os.makedirs(args.output, exist_ok=True)
    cleaned_dir = 'patrick-files/cleaned-files'
    os.makedirs(cleaned_dir, exist_ok=True)

    # Iterate over all images in the captcha directory
    # for filename in os.listdir(args.captcha_dir):
    for idx, filename in enumerate(os.listdir(args.captcha_dir)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(args.captcha_dir, filename)
            img = cv2.imread(img_path)

            # Clean the image
            clean = remove_noise(img, False)

            # Save the cleaned image in 'patrick-files/cleaned-files'
            # clean_filename = f"{os.path.splitext(filename)[0]}_cleaned.png"
            # clean_output_path = os.path.join(cleaned_dir, clean_filename)
            # cv2.imwrite(clean_output_path, clean)

            # Segment the cleaned image into individual characters
            chars = segment(clean, index=idx, visualize=False, two_char_min= args.min_two_char, three_char_min= args.min_three_char, four_char_min= args.min_four_char)
        
            for i, char in enumerate(chars):
                char_filename = f"{os.path.splitext(filename)[0]}_char_{i}.png"
                char_output_path = os.path.join(args.output, char_filename)
                cv2.imwrite(char_output_path, char)
        
        # below can be uncommented if you want to iterate through the results one at a time for testing purposes (can be deleted if necessary)
        # input('Press "y" to continue to the next iteration: ')


if __name__ == "__main__":
    main()
