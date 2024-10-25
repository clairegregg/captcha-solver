import argparse
import os
import cv2
import numpy as np
from itertools import product

# Please note that if you set visualize to True then you need to pass in a unique index into this function (otherwise files will get overwritten)
def segment(
    cleaned, two_char_min, three_char_min, four_char_min, index=0, visualize=False, visualization_dir='visualizations'
):
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
                char_images_with_positions.append((left_x, left_char))
                char_images_with_positions.append((right_x, right_char))
            elif w > three_char_min and w <= four_char_min:
                # This is likely an image with three overlapping characters
                column_sums = np.sum(char_image == 0, axis=0)
                padding_value = 0.15
                w_img = char_image.shape[1]
                h_img = char_image.shape[0]
                
                # First divider
                first_split_pos = int(w_img / 3)
                first_search_start = max(0, int(first_split_pos - w_img * padding_value))
                first_search_end = min(w_img, int(first_split_pos + w_img * padding_value))
                first_sub_column_sums = column_sums[first_search_start:first_search_end]
                first_divider_offset = np.argmin(first_sub_column_sums) + first_search_start

                # Second divider
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
                char_images_with_positions.append((first_x, first_char))
                char_images_with_positions.append((second_x, second_char))
                char_images_with_positions.append((third_x, third_char))
            
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
                char_images_with_positions.append((first_x, first_char))
                char_images_with_positions.append((second_x, second_char))
                char_images_with_positions.append((third_x, third_char))
                char_images_with_positions.append((fourth_x, fourth_char))
            else:
                # For single character images
                char_images_with_positions.append((x, char_image))
    
    # Sort the character images based on their x-coordinate to maintain order
    char_images_with_positions.sort(key=lambda tup: tup[0])
    char_images = [img for x_pos, img in char_images_with_positions]
    
    return char_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas', type=str)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to split")
        exit(1)

    # Define the refined ranges based on your top-performing combinations
    two_char_min_values = range(35, 50)       # From 35 to 40 inclusive
    three_char_min_values = range(50, 75)     # From 55 to 65 inclusive
    four_char_min_values = range(86, 105, 5)   # From 75 to 90 inclusive, step by 5 to reduce combinations

    combinations = []

    for tcm in two_char_min_values:
        for thcm in three_char_min_values:
            if thcm > tcm:
                for fcm in four_char_min_values:
                    if fcm > thcm:
                        combinations.append((tcm, thcm, fcm))

    correctly_segmented_per_combination = []

    total_combinations = len(combinations)
    total_captchas = len(os.listdir(args.captcha_dir))
    processed_captchas = 0

    for idx, (two_char_min, three_char_min, four_char_min) in enumerate(combinations):
        print(f'\nTesting combination {idx + 1}/{total_combinations}: two_char_min={two_char_min}, three_char_min={three_char_min}, four_char_min={four_char_min}')
        correctly_segmented = 0
        processed_captchas = 0

        for file_idx, filename in enumerate(os.listdir(args.captcha_dir)):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # Get the filename without the extension
                file_base = filename.split('.')[0]
            
                # Check if there is a duplicate identifier (e.g., _2) and remove it for character count purposes
                if "_" in file_base:
                    file_base = "_".join(file_base.split("_")[:-1])

                # Get the number of characters by measuring the length of the remaining part of the filename
                num_chars_in_filename = len(file_base)
                
                img_path = os.path.join(args.captcha_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # return the size of the segmented character
                segmented_characters = segment(img, index=file_idx, visualize=False, two_char_min=two_char_min, three_char_min=three_char_min, four_char_min=four_char_min)
                number_of_segmented_characters = len(segmented_characters)

                if number_of_segmented_characters == num_chars_in_filename:
                    correctly_segmented += 1

                processed_captchas += 1

                if file_idx % 500 == 0 and file_idx != 0:
                    print(f'Processed {file_idx} captchas')
        
        accuracy = (correctly_segmented / processed_captchas) * 100
        correctly_segmented_per_combination.append((accuracy, (two_char_min, three_char_min, four_char_min)))
        print(f'Combination {idx + 1}/{total_combinations} complete! Correctly segmented: {correctly_segmented}/{processed_captchas} ({accuracy:.2f}%)')

    # Sort the results by accuracy in descending order
    correctly_segmented_per_combination.sort(reverse=True, key=lambda x: x[0])

    print("\nTop 10 combinations:")
    for i, (accuracy, (two_char_min, three_char_min, four_char_min)) in enumerate(correctly_segmented_per_combination[:10]):
        print(f'{i + 1}. Accuracy: {accuracy:.2f}%, two_char_min: {two_char_min}, three_char_min: {three_char_min}, four_char_min: {four_char_min}')

if __name__ == "__main__":
    main()
