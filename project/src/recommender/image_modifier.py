from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def process_image(filename, input_folder, output_folder):
    # Open the image file
    with Image.open(os.path.join(input_folder, filename)) as img:
        # Ensure the image is of size 1000x400 before processing
        if img.size == (1000, 400):
            # Crop the image: Remove 16 pixels from the top and 8 pixels from the right
            # Parameters for crop: (left, upper, right, lower)
            cropped_img = img.crop((0, 16, 992, 400))

            # Save the cropped image to the output folder
            cropped_img.save(os.path.join(output_folder, filename))


def batch_resize_images(input_folder, output_folder, target_size=(992, 384)):
    """
    Resize and crop images in a folder to a specified target size.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PNG files in the input folder
    image_files = [f for f in os.listdir(
        input_folder) if f.lower().endswith('.png')]

    # Setup ThreadPoolExecutor to use multiple threads for processing images
    with ThreadPoolExecutor() as executor:
        # Use a tqdm progress bar with executor.map
        results = list(tqdm(executor.map(process_image, image_files,
                                         [input_folder]*len(image_files),
                                         [output_folder]*len(image_files)),
                            total=len(image_files), desc="Processing Images"))


# Example usage
batch_resize_images('../../data/mel_specs_resized_partitions/parts31to46/images',
                    '../../data/mel_specs_resized_partitions/parts31to46_resized/images')
