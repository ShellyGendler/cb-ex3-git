import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Function to create and save digit images
def create_and_save_digit_images(digit, fonts, output_dir, image_size=(28, 28)):
    for font_path in fonts:
        try:
            font = ImageFont.truetype(font_path, 24)
        except OSError:
            print(f"Cannot open font file: {font_path}")
            continue
        
        # Create the image
        image = Image.new('L', image_size, color=255)
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox(((0, 0)), str(digit), font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((image_size[0] - width) / 2, (image_size[1] - height) / 2), str(digit), fill=0, font=font)
        
        # Save the image
        image.save(os.path.join(output_dir, f'digit_{digit}_font_{os.path.basename(font_path)}.png'))

# Create all digit images with given fonts and save them to a directory
def pre_create_digit_images(digits, fonts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for digit in digits:
        for font_path in fonts:
            create_and_save_digit_images(digit, [font_path], output_dir)

if __name__ == "__main__":
    digits = list(range(10))
    fonts = [
        'C:/Windows/Fonts/Gadugi.ttf',
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/Calibri.ttf',
        'C:/Windows/Fonts/Times.ttf',
        'C:/Windows/Fonts/Verdana.ttf',
        'C:/Windows/Fonts/Corbel.ttf',
        'C:/Windows/Fonts/Cour.ttf',
        'C:/Windows/Fonts/Comic.ttf',
        'C:/Windows/Fonts/Tahoma.ttf',
        'C:/Windows/Fonts/Georgia.ttf'
    ]
    
    # Directory to save pre-created images
    output_dir = r'C:\Users\Admin\PycharmProjects\cb-ex3\cb-ex3-git\pre_created_images'
    
    # Pre-create digit images with fonts and save them
    pre_create_digit_images(digits, fonts, output_dir)
