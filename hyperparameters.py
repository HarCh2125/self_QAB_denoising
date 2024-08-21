# Function to set the hyperparameters
def set_hyperparams():
    h_bar = float(input("Enter the value of \\hbar (the reduced Planck's constant): "))
    mass = float(input("Enter the mass: "))
    s = float(input("Enter the value of s: "))
    rho = float(input("Enter the value of \\rho: "))
    sigma = float(input("Enter the value of \\sigma: "))

    return h_bar, mass, s, rho, sigma