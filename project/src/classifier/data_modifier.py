import os
from PIL import Image

# Define the source and destination directories
src_dir = '/home/charlie/school/code/csci5707/project/data/songs/images_cropped_augment'
dst_dir = '/home/charlie/school/code/csci5707/project/data/songs/images_cropped_segments'

# List of genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Loop through each genre
for genre in genres:
    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.join(dst_dir, genre), exist_ok=True)
    
    # Loop through each image in the genre directory
    for img_name in os.listdir(os.path.join(src_dir, genre)):
        # Load the image
        img_path = os.path.join(src_dir, genre, img_name)
        img = Image.open(img_path)
        
        # Break the image into 10 segments
        for i in range(10):
            # Define the coordinates for cropping
            left = i * 33
            upper = 0
            right = (i + 1) * 33
            lower = 217
            
            # Crop the segment
            segment = img.crop((left, upper, right, lower))
            
            # Define the new filename
            segment_name = img_name.rsplit('.', 1)[0] + f".{i+1}.png"  # changed to .png
            
            # Save the segment
            segment_path = os.path.join(dst_dir, genre, segment_name)
            segment.save(segment_path, "PNG")  # specify the format as PNG

print("Images segmented and saved successfully!")
