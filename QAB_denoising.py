import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.linalg import eigh
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def calc_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    if img1.shape != img2.shape:
        raise ValueError('Inputs must be of the same size.')
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = np.max([img1.max(), img2.max()])
    return 10 * np.log10(max_val ** 2 / mse)

def f_ondes2D(image, poids):
    """Calculate the eigenvectors and eigenvalues of the Hamiltonian of the image."""
    N, M = image.shape  # assume N = M
    dim = N ** 2
    im_vect = image.flatten()

    # Create space for data
    psi_tmp = np.zeros((dim, dim))
    E = np.zeros(dim)
    
    # Hamiltonian matrix H construction
    terme_hsm = np.ones(dim) * poids
    H = np.diag(im_vect) + np.diag(terme_hsm) * 4 \
        - np.diag(terme_hsm[:-1], -1) - np.diag(terme_hsm[:-1], 1) \
        - np.diag(terme_hsm[:-N], -N) - np.diag(terme_hsm[:-N], N)
    
    # Boundary conditions
    for bloc in range(N):
        H[N * bloc, N * bloc] -= poids
        H[N * (bloc + 1) - 1, N * (bloc + 1) - 1] -= poids
    
    for ligne in range(N):
        H[ligne, ligne] -= poids
        H[dim - ligne - 1, dim - ligne - 1] -= poids
    
    for inter in range(1, N):
        H[N * inter, N * inter - 1] = 0
        H[N * inter - 1, N * inter] = 0
    
    # Eigenvalue decomposition
    print('Calculating eigenvectors...')
    eig_vals, eig_vecs = eigh(H)
    print('Eigenvectors calculated.')
    
    valPmat = eig_vals
    vpM = valPmat.max()
    
    for g in range(dim):
        E_min_idx = np.argmin(valPmat)
        psi_tmp[:, g] = eig_vecs[:, E_min_idx]
        E[g] = valPmat[E_min_idx]
        valPmat[E_min_idx] = vpM + 1
    
    psi = psi_tmp.reshape(N, N, dim)
    psi_cols = psi_tmp
    
    return psi, psi_cols, E

def heavi(x, eps):
    """Heaviside step function with smoothing."""
    N = x.size
    fonct = np.zeros(N)
    
    gauche = np.ones(N)
    milieu = 0.5 * (1 - x / eps - 1 / np.pi * np.sin(np.pi * x / eps))
    
    H = fonct + gauche * (x <= -eps) + milieu * (x > -eps) * (x < eps)
    return H

def image_denoising_QAB(I, J, Ms, pds, sg):
    """Image denoising using Quantum Adaptive Basis (QAB)."""
    M, M1 = I.shape
    N = 64
    NN = N ** 2
    P = 2 * M // N
    
    I_old = I.copy()
    J_old = J.copy()
    
    saut = 12
    Vs = np.linspace(7, 11, Ms)
    Vs = 2 ** Vs
    
    print('Start the search of wave function')
    J_new = np.zeros((M, M))
    cmpt = np.zeros((M, M))
    
    for i in range(P - 1):
        for j in range(P - 1):
            J_part = J[i * N // 2:(i + 1) * N // 2, j * N // 2:(j + 1) * N // 2]
            I_part = I[i * N // 2:(i + 1) * N // 2, j * N // 2:(j + 1) * N // 2]
            
            J_part_max = J_part.max()
            I_part /= J_part_max
            J_part /= J_part_max
            
            pI_part = np.sum(I_part ** 2) / NN
            
            x, y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
            z = (1 / (np.sqrt(2 * np.pi) * sg)) ** 2 * np.exp(-(x ** 2 + y ** 2) / (2 * sg))
            
            gaussF = fft2(fftshift(z))
            JF = fft2(J_part)
            n_J = np.real(ifft2(gaussF * JF))
            
            J_col = J_part.flatten()
            psi, psi_col, E = f_ondes2D(n_J, pds)
            alp = np.linalg.lstsq(psi_col, J_col, rcond=None)[0]
            
            for k in range(Ms):
                v = Vs[k]
                max_atteint = 0
                RSB_s = 0
                l = 1
                
                while l <= NN and max_atteint < 20:
                    x = np.arange(1, NN + 1) - l + 2
                    taux = heavi(x, v)
                    
                    n_I = np.zeros((N, N))
                    for t in range(NN):
                        n_I += psi[:, :, t] * taux[t] * alp[t]
                    
                    n_B = n_I - I_part
                    pnB = np.sum(n_B ** 2) / NN
                    RSB_n = 10 * np.log10(pI_part / pnB)
                    
                    max_atteint = (max_atteint + 1) * (RSB_s > RSB_n)
                    RSB_s = max(RSB_s, RSB_n)
                
                Jns = n_I * J_part_max
                
                J_new[i * N // 2:(i + 1) * N // 2, j * N // 2:(j + 1) * N // 2] += Jns
                cmpt[i * N // 2:(i + 1) * N // 2, j * N // 2:(j + 1) * N // 2] += 1
    
    J_new /= cmpt
    return J_new

# Main script
if __name__ == "__main__":
    # Load your sample image here
    I = np.load('sample_image.npy')  # Assuming image is saved as numpy array
    M, N = I.shape
    SNR = 15
    pI = np.sum(I ** 2) / I.size
    B = np.random.randn(N, N) * np.sqrt(np.abs(I))
    pB_tmp = np.sum(B ** 2) / (N ** 2)
    B = B / np.sqrt(pB_tmp) * np.sqrt(pI * 10 ** (-SNR / 10))
    J = B + I

    Ms = 20
    pds = 3
    sg = 7.5

    I_result = image_denoising_QAB(I, J, Ms, pds, sg)
    
    # Calculate PSNR, SNR, SSIM
    pnB = np.mean((I_result - I) ** 2)
    SNR_end = 10 * np.log10(pI / pnB)
    PSNR_end = calc_psnr(I, I_result)
    SSIM_end = ssim(I_result, I, data_range=I.max() - I.min())

    print(f'OUTPUT: SNR = {SNR_end:.2f}, PSNR = {PSNR_end:.2f}, SSIM = {SSIM_end:.2f}')
    
    # Plot figures
    font_size = 12
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Clean image')
    plt.subplot(1, 3, 2)
    plt.imshow(J, cmap='gray')
    plt.title(f'Noisy image (SNR = {SNR:.2f} dB)')
    plt.subplot(1, 3, 3)
    plt.imshow(I_result, cmap='gray')
    plt.title('Denoised image')
    plt.show()
