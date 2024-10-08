import os
import numpy as np
import cv2 as cv

# Function to read in the sample images
def read_images(path: str) -> list:
    data = []

    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)

        if os.path.isfile(file):
            img = cv.imread(file)
            data.append(np.array(img, dtype = np.uint8))

    return data

# # Function to read in the sample images
# def read_images(path: str) -> list:
#     data = []

#     for file_name in os.listdir(path):
#         file = os.path.join(path, file_name)

#         if os.path.isfile(file):
#             img = cv.imread(file)

#             if img is not None:  # Check if the image was successfully loaded
#                 data.append(np.array(img, dtype=np.uint8))
#             else:
#                 print(f"Warning: {file_name} could not be read as an image.")

#     return data

data = read_images('downscaled_my_samples')