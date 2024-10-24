import re
import sys

def combine_files(file1, file2, output_file):
    # Helper function to extract sort key based on filename
    def sort_key(line):
        # Extract the filename (part before the comma)
        return re.search(r'([^,]+)', line).group(1)

    # Open files 1 and 2
    with open(file1, 'r') as f1:
        lines1 = f1.readlines()
    with open(file2, 'r') as f2:
        lines2 = f2.readlines()

    # Extract the name from the top of the first file and then take all captcha classifications from both files
    student_name = lines1[0].strip()
    combined_lines = lines1[1:] + lines2[1:]

    # Sort all the captchas
    sorted_lines = sorted(combined_lines, key=sort_key)

    # Write the combined and sorted lines to the output file
    with open(output_file, 'w') as out:
        # Write the student name as the first line
        out.write(student_name + '\n')
        # Write the sorted CAPTCHA answers
        for line in sorted_lines:
            out.write(line)

if __name__ == "__main__":
    # Check if the right number of arguments is passed
    if len(sys.argv) != 4:
        print("Usage: python script.py <file1> <file2> <output_file>")
        sys.exit(1)

    # Get the filenames from the command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Call the combine_files function with the provided arguments
    combine_files(file1, file2, output_file)
