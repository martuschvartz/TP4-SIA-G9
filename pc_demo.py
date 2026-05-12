import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def pca_function():
    # 1. load and standardize data
    europe_data = pd.read_csv('europe.csv')
    country_names = europe_data['Country']
    numeric_data = europe_data.drop(columns=['Country'])    # remove country since its not numeric

    scaled_array = StandardScaler().fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_data.columns)

    # 2. build correlation matrix
    correlation_matrix = scaled_df.corr()

    # 3. calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    print(f"All eigenvalues: {eigenvalues}")
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # keep only the first two eigenvalues/eigenvectors for PC1 and PC2
    pc1_pc2_eigenvalues = eigenvalues[sorted_indices][:2]
    pc1_pc2_eigenvectors = eigenvectors[:, sorted_indices][:, :2]

    # 4. print PC1 loadings (weights of each variable in Y1)
    print("=== PC1 Loadings ===")
    for var, weight in zip(numeric_data.columns, pc1_pc2_eigenvectors[:, 0]):
        print(f"  {var}: {weight:.4f}")
    print(f"\n  Eigenvalue PC1: {pc1_pc2_eigenvalues[0]:.4f}")
    print(f"  Variance explained by PC1: {pc1_pc2_eigenvalues[0] / sum(eigenvalues) * 100:.1f}%")

    # 5. project data onto PC1 and PC2
    projected = scaled_array @ pc1_pc2_eigenvectors

    # --- GRAPH 1: before PCA ---
    plt.figure(figsize=(10, 7))

    # get the two variables with highest absolute weight in PC1 -> those will affect the most PC1
    pc1_loadings = np.abs(pc1_pc2_eigenvectors[:, 0])
    top2_indices = np.argsort(pc1_loadings)[::-1][:2]
    top2_vars = numeric_data.columns[top2_indices]

    # then use them in the before plot
    plt.scatter(scaled_array[:, top2_indices[0]], scaled_array[:, top2_indices[1]])
    plt.xlabel(top2_vars[0])
    plt.ylabel(top2_vars[1])
    plt.title('Before PCA (original variables)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig('before_pca.png')

    # --- GRAPH 2: PC1 vs PC2 scatter ---
    plt.figure(figsize=(10, 7))
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA - European Countries')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig('pca_scatter.png')

    # --- GRAPH 3: bar chart PC1 per country ---
    plt.figure(figsize=(12, 6))
    plt.bar(country_names, projected[:, 0])
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('PC1')
    plt.title('PC1 per Country')
    plt.savefig('pc1_bar.png')

if __name__ == '__main__':
    pca_function()