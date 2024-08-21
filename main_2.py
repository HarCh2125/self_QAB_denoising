from build_Hamiltonian import *
from compute_eigenvecs import *
from load_img import *
import numpy as np



# Build the Hamiltonian list, containing Hamiltonians for each image
Hamiltonians = build_Hamiltonian(data, hbar, mass)

# Compute the eigenvectors and eigenvalues of the Hamiltonaians
eigenvals, eigenvecs = compute_eigenpairs(Hamiltonians, num_eigen = 3000)