import numpy as np
import matplotlib as plt
import cv2 as cv
import os
from hyperparameters import *
from read_data import *
from pre_process import *
from build_Hamiltonian import *
import random
from math import sqrt
from compute_eigenvecs import *
from save_img import *
from load_img import *

# Set the hyperparameters
hbar, mass, s, rho, sigma = set_hyperparams()

# Read in the sample images and preprocess it (assuming no noise is already present)
dir_path = 'downscaled_my_samples'
data = read_images(path = dir_path)
data = preprocess(data, k_size = 5, sigma = sigma, pad = 50)

# Save the images as NumPy arrays
save_images(data)

# Load the image arrays
path = 'data_arrays'
data = load_images(path)

# # Build the Hamiltonian list, containing Hamiltonians for each image
# Hamiltonians = build_Hamiltonian(data, hbar, mass)

# # Compute the eigenvectors and eigenvalues of the Hamiltonaians
# eigenvals, eigenvecs = compute_eigenpairs(Hamiltonians, num_eigen = 3000)