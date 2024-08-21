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

# Set the hyperparameters
hbar, mass, s, rho, sigma = set_hyperparams()

# Read in the sample images and preprocess it (assuming no noise is already present)
dir_path = 'downscaled_my_samples'
data = read_images(path = dir_path)
data = preprocess(data, k_size = 5, sigma = sigma, pad = 50)

# Build the Hamiltonian list, containing Hamiltonians for each image
Hamiltonians = build_Hamiltonian(data, hbar, mass)