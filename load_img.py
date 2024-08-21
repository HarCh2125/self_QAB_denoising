import numpy as np
import os

# Function to load the image arrays
def load_images(path: str) -> list:
    data = []
    for filename in os.listdir(path):
        file = os.path.join(path, filename)

        if os.path.isfile(file):
            data.append(np.load(file))

    return data