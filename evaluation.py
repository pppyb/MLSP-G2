import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import FastICA, NMF, PCA, IncrementalPCA
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

# Define the sampling frequency
fs = 4000


# Define a function to calculate the reconstruction error
def calculate_reconstruction_error(original_signal, components, H):
    """
    Calculate the signal reconstruction error (E_r).

    Args:
    - original_signal: numpy array of the original HDEMG signal.
    - components: numpy array of the components (W matrix in NMF, PCA scores).
    - H: numpy array of the components' basis (H matrix in NMF, PCA loadings).

    Returns:
    - E_r: The reconstruction error.
    """
    reconstructed_signal = np.dot(components, H)
    discrepancies = (original_signal - reconstructed_signal) ** 2
    E_r = np.mean(discrepancies)
    return E_r


# Define a function to calculate the maximum correlation coefficient
def calculate_max_correlation(components):
    max_corr = 0
    for i in range(components.shape[0]):
        for j in range(i + 1, components.shape[0]):
            corr, _ = pearsonr(components[i], components[j])
            max_corr = max(max_corr, abs(corr))
    return max_corr


# Define a function to calculate mutual information
def calculate_mutual_information(components):
    mi = 0
    for i in range(components.shape[0]):
        for j in range(i + 1, components.shape[0]):
            mi = max(mi, mutual_info_score(components[i], components[j]))
    return mi


# Define a function to calculate the physiological plausibility
def calculate_physiological_plausibility(components):
    # Placeholder for actual physiological plausibility calculation
    # This could be based on known frequency bands, spike shapes, etc.
    return np.mean(np.std(components, axis=1))  # use standard deviation as a proxy


# Define a function to calculate the stability across different conditions
def calculate_stability(components_list):
    # Placeholder for actual stability calculation
    # This could compare the components across different conditions
    return np.mean([calculate_max_correlation(components) for components in components_list])

# Load ICA data
# Calculate evaluation metrics for ICA
E_r_ica = calculate_reconstruction_error(out_mat_steady, S_ica, A_ica)
max_corr_ica = calculate_max_correlation(S_ica)
mi_ica = calculate_mutual_information(S_ica)
physio_ica = calculate_physiological_plausibility(S_ica)
stability_ica = calculate_stability([S_ica])

# Print the evaluation metrics for ICA
print(f"ICA Reconstruction Error: {E_r_ica}")
print(f"ICA Maximum Correlation: {max_corr_ica}")
print(f"ICA Mutual Information: {mi_ica}")
print(f"ICA Physiological Plausibility: {physio_ica}")
print(f"ICA Stability: {stability_ica}")

# Repeat similar calculations for NMF, PCA, IPCA
# Define the number of components you want to extract
n_comps = [2, 4, 8, 16, 32]

# Initialize lists to store the components for each method
H_list_steady_nmf = []
H_list_incr_1_nmf = []
H_list_incr_2_nmf = []

H_list_steady_pca = []
H_list_incr_1_pca = []
H_list_incr_2_pca = []

H_list_steady_ipca = []
H_list_incr_1_ipca = []
H_list_incr_2_ipca = []

# Define the output directory for saving the components
output_directory = "./MLSP-G2/decomposed_data/"

# Apply NMF, PCA, and IPCA for each number of components
for each_n in n_comps:
    # NMF
    model_nmf = NMF(n_components=each_n, init='nndsvda', max_iter=1000)
    W_nmf = model_nmf.fit_transform(out_mat_steady)
    H_nmf = model_nmf.components_
    H_list_steady_nmf.append(H_nmf.T)
    # Repeat for out_mat_incr_1 and out_mat_incr_2

    # PCA
    model_pca = PCA(n_components=each_n)
    model_pca.fit(out_mat_steady)
    H_pca = model_pca.components_
    H_list_steady_pca.append(H_pca.T)
    # Repeat for out_mat_incr_1 and out_mat_incr_2

    # IPCA
    ipca = IncrementalPCA(n_components=each_n, batch_size=5)
    for i in range(0, len(out_mat_steady), ipca.batch_size):
        batch_data = out_mat_steady[i:i + ipca.batch_size]
        ipca.partial_fit(batch_data)
    H_ipca = ipca.components_
    H_list_steady_ipca.append(H_ipca.T)
    # Repeat for out_mat_incr_1 and out_mat_incr_2


# Plot and save the components for each method
for i, each_n in enumerate(n_comps):
    # Plot NMF components
    plt.figure(figsize=(20, 10))
    plt.plot(H_list_steady_nmf[i])
    plt.title(f"NMF Components with n_components={each_n}")
    plt.savefig(output_directory + f'steady_mat_nmf_compo_{each_n}.png')
    plt.show()

    # Plot PCA components
    plt.figure(figsize=(20, 10))
    plt.plot(H_list_steady_pca[i])
    plt.title(f"PCA Components with n_components={each_n}")
    plt.savefig(output_directory + f'steady_mat_pca_compo_{each_n}.png')
    plt.show()

    # Plot IPCA components
    plt.figure(figsize=(20, 10))
    plt.plot(H_list_steady_ipca[i])
    plt.title(f"IPCA Components with n_components={each_n}")
    plt.savefig(output_directory + f'steady_mat_ipca_compo_{each_n}.png')
    plt.show()