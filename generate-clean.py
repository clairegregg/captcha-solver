import argparse
import os
import generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font-dir', help="Directory where fonts are stored", type=str)
    parser.add_argument('--output-dir', help="Where to store clean generated captchas", type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--count', help="How many captchas to be generated with each font", type=int)
    args = parser.parse_args()

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.font_dir is None:
        print("Please specify the directory containing fonts")
        exit(1)
    
    fonts = os.listdir(args.font_dir)
    print(fonts)

    # Generate count captchas using each font
    for font in fonts:
        font_simple = os.path.splitext(font.replace(" ", ""))[0]
        generate.generate(100,100,1,args.count,os.path.join(args.output_dir, font_simple), args.symbols, os.path.join(args.font_dir, font), True)

if __name__ == "__main__":
    main()