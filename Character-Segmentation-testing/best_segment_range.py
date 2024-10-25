import argparse
import os
import cv2
import numpy as np

# Please note that if you set visuailize to True then you need to pass in a unique index into this function (otherwise files will get overwritten)
def segment(
    cleaned, two_char_min , three_char_min, four_char_min, index = 0, visualize=False, visualization_dir='visualizations'
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
    
    # print(f'two_char_min: {two_char_min}, three_char_min: {three_char_min}, four_char_min: {four_char_min}')
    
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


    # WildCrazy - old
    # two_char_min = [42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]
    # three_char_min = [68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88]
    # four_char_min = [94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114]
    
    # WildCrazy - new
    # two_char_min = [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
    # three_char_min = [52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72]
    # four_char_min = [78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
    
    # # WildCrazy - final
    # two_char_min = [40, 41, 42]
    # three_char_min = [66, 67, 68]
    # four_char_min = [92, 93, 94]
    
        # WildCrazy - additional test
    # two_char_min = [40, 40, 40, 40, 40, 40, 40, 40]
    # three_char_min = [66, 66, 66, 66, 66, 66, 66, 66]
    # four_char_min = [92, 92, 92, 92, 92, 92, 92, 92, 101]
   
    # Crazystyle - old
    # two_char_min = [36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
    # three_char_min = [59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79]
    # four_char_min = [82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102]
    
    # two_char_min = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37]
    # three_char_min = [41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60]
    # four_char_min = [64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 83]
    
    two_char_min = [36, 36, 36]
    three_char_min = [64, 59 , 49]
    four_char_min = [90, 82, 90]
    
    
    correctly_segmented_per_batch = []
    
    for i in range(len(two_char_min)):
        # print(f'testing value range: {i}')
        # print(f'values are {two_char_min[i]}, {three_char_min[i]}, {four_char_min[i]}')
        correctly_segmented = 0
        for idx, filename in enumerate(os.listdir(args.captcha_dir)):
            # if idx == 8:
            #     break
            if filename.endswith(".png") or filename.endswith(".jpg"):
                
                # Get the filename without the extension
                file_base = filename.split('.')[0]
            
                # Check if there is a duplicate identifier (e.g., _2) and remove it for character count purposes
                if "_" in file_base:
                    file_base = "_".join(file_base.split("_")[:-1])

                # Get the number of characters by measuring the length of the remaining part of the filename
                num_chars_in_filename = len(file_base)
                
                img_path = os.path.join(args.captcha_dir, filename)
                # img = cv2.imread(img_path) #take in clean image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # return the size of the segmented character
                segmented_characters = segment(img, index=idx, visualize=False, two_char_min=two_char_min[i], three_char_min=three_char_min[i], four_char_min=four_char_min[i])
                number_of_segmented_characters = len(segmented_characters)
                
                # print(f'number_of_segmented_characters : {number_of_segmented_characters}')
                # print(f'num_chars_in_filename - {filename}: {num_chars_in_filename}')

                if number_of_segmented_characters == num_chars_in_filename:
                    correctly_segmented += 1
                #     # print('Correctly Segmented')
                # else:
                #     # print('Incorrectly Segmented')

                if idx % 100 == 0:
                    print(f'progress {idx}/2,000')
                    
        correctly_segmented_per_batch.append(correctly_segmented)
        print(f'Batch {i + 1}/11 complete!')
        # below can be uncommented if you want to iterate through the results one at a time for testing purposes (can be deleted if necessary)
        # input('Press "y" to continue to the next iteration: ')
    
    print(f'Correctly Segmented Per batch : {correctly_segmented_per_batch}')
    # Print the index of the highest value and the highest value itself
    max_value = max(correctly_segmented_per_batch)
    max_index = correctly_segmented_per_batch.index(max_value)
    print(f'Highest value: {max_value}, at index: {max_index}')


if __name__ == "__main__":
    main()
