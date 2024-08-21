import numpy as np
from scipy.sparse.linalg import eigsh

# Function to compute the eigenvectors and eigenvalues of the Hamiltonians
def compute_eigenpairs(Hamiltonians: list, num_eigen: int = 6) -> list:
    eigenpairs = []

    for H in Hamiltonians:
        # Convert to CSR format for efficient eigenvalue computation
        H_csr = H.tocsr()

        # Compute the num_eigen smallest eigenvalues and corresponding eigenvectors
        eigvals, eigvecs = eigsh(H_csr, k = num_eigen, which = 'SM')

        eigenpairs.append((eigvals, eigvecs))

    return eigenpairs