import cv2 as cv
import numpy as np

# Function for preprocessing the dataset
def preprocess(data: list, k_size: int = 5, sigma: float = 6.000, pad: int = 50) -> list:
    output = []

    # Step 0: Corrupt with noise
    # Prepare AWGN and add to the image
    for img in data:
        mean = 0
        stddev = 25
        noise = np.random.normal(mean, stddev, img.shape)
        noisy_image = img + noise

        # Clip the values to make sure the pixel values stay between 0 and 255
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        output.append(noisy_image)

    data = output
    output = []

    # STEP 1: Convert each image to grayscale
    for img in data:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        output.append(np.array(img, dtype = np.uint8))

    # STEP 2: Zero-pad the images
    data = output
    output = []

    for img in data:
        img = np.pad(img, pad, mode = 'constant', constant_values = 0)
        output.append(img)

    # STEP 3: Apply a Gaussian filter
    data = output
    output = []

    for img in data:
        img = cv.GaussianBlur(img, (k_size, k_size), sigma)
        cv.imshow('image', img)
        output.append(img)
        cv.waitKey(0) 

    # STEP 4: Vectorise the image
    data = output
    output = []

    for img in data:
        img = np.reshape(img, img.shape[0] * img.shape[1])
        output.append(img)

    # Return the dataset now
    return output

# preprocess([cv.imread('my_samples/boat_new.jpg')])