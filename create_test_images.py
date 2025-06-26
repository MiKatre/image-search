#!/usr/bin/env python3
"""
Create simple test images for testing the image search CLI.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(filename, color, text, size=(200, 200)):
    """Create a simple colored image with text."""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text with contrasting color
    text_color = 'white' if sum(color) < 384 else 'black'
    draw.text((x, y), text, fill=text_color, font=font)
    
    img.save(f"test_images/{filename}")
    print(f"Created {filename}")

# Create test images
os.makedirs("test_images", exist_ok=True)

# Create different colored squares with descriptive text
create_test_image("red_car.jpg", (255, 0, 0), "Red Car")
create_test_image("blue_sky.jpg", (135, 206, 235), "Blue Sky")
create_test_image("green_tree.jpg", (34, 139, 34), "Green Tree")
create_test_image("yellow_sun.jpg", (255, 255, 0), "Yellow Sun")
create_test_image("purple_flower.jpg", (128, 0, 128), "Purple Flower")
create_test_image("orange_sunset.jpg", (255, 165, 0), "Orange Sunset")

print("✓ Created 6 test images in test_images/ directory")