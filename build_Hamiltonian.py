import numpy as np
import cv2 as cv
import scipy.sparse as sp

# Function to build Hamiltonian
def build_Hamiltonian(data: list, hbar: float, mass: float) -> list:
    Hamiltonians = []

    # Define a constant for the ratio \frac{\hbar^2}{2m}
    qdc = (hbar**2)/(2*mass)

    for img in data:
        N = img.shape[0]
        dim = N * N
        H = sp.lil_matrix((dim, dim))  # Create a sparse matrix in LIL format

        # Diagonal construction similar to MATLAB
        main_diag = img.flatten() + 4 * qdc
        off_diag_1 = -qdc * np.ones(dim-1)
        off_diag_N = -qdc * np.ones(dim-N)

        H.setdiag(main_diag)  # Main diagonal
        H.setdiag(off_diag_1, -1)  # Just below main diagonal
        H.setdiag(off_diag_1, 1)  # Just above main diagonal
        H.setdiag(off_diag_N, -N)  # Nth diagonal below main diagonal
        H.setdiag(off_diag_N, N)  # Nth diagonal above main diagonal

        # Boundary condition adjustments
        for bloc in range(N):
            H[bloc*N, bloc*N] -= qdc
            H[(bloc+1)*N-1, (bloc+1)*N-1] -= qdc

        for ligne in range(N):
            H[ligne, ligne] -= qdc
            H[-(ligne+1), -(ligne+1)] -= qdc

        for inter in range(1, N):
            H[inter*N, inter*N-1] = 0
            H[inter*N-1, inter*N] = 0

        Hamiltonians.append(H)

    return Hamiltonians