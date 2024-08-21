import numpy as np

# Function to save the images as separate NumPy arrays
def save_images(data: list):
    for i in range(len(data)):
        np.save(f'data_arrays/{i}', data[i])