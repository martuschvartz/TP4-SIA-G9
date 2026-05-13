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
    sorted_indices = np.argsort(eigenvalues)[::-1]
    all_sorted_eigenvalues = eigenvalues[sorted_indices]
    all_sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # keep only the first two eigenvectors for PC1 and PC2
    pc1_pc2_eigenvectors = all_sorted_eigenvectors[:, :2]

    # 4. print eigenvalue table
    print("=== Eigenvalue Table ===")
    print(f"{'PC':<6} {'Eigenvalue':<12} {'% Variance':<12}")
    for i, val in enumerate(all_sorted_eigenvalues):
        pct = val / sum(eigenvalues) * 100
        print(f"{'PC' + str(i + 1):<6} {val:<12.4f} {pct:<12.1f}")
    print("-" * 45)

    # 5. print loadings table
    print("\n=== Eigenvector Weights Table ===")
    header = f"{'Variable':<15}" + "".join([f"{'PC'+str(i+1):<10}" for i in range(len(eigenvalues))])
    print(header)
    print("-" * (15 + 10 * len(eigenvalues)))
    for j, var in enumerate(numeric_data.columns):
        row = f"{var:<15}" + "".join([f"{all_sorted_eigenvectors[j, i]:<10.4f}" for i in range(len(eigenvalues))])
        print(row)

    # 6. project data onto PC1 and PC2
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
    plt.title(f'Before PCA (top 2 PC1 variables: {top2_vars[0]} and {top2_vars[1]})')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('before_pca.png')

    # --- GRAPH 2: PC1 vs PC2 scatter ---
    plt.figure(figsize=(10, 7))
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('After PCA - European Countries (PC1 vs PC2)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('pca_scatter.png')

    # --- GRAPH 3: bar chart PC1 per country ---
    plt.figure(figsize=(12, 6))
    plt.bar(country_names, projected[:, 0])
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('PC1')
    plt.title('PC1 per Country')
    plt.tight_layout()
    plt.savefig('pc1_bar.png')

    # --- GRAPH 4: scree plot (eigenvalues) ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), all_sorted_eigenvalues, 'bo-')
    plt.axhline(y=1, color='red', linestyle='--', label='Kaiser criterion (eigenvalue = 1)')
    plt.xlabel('PC')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.xticks(range(1, len(eigenvalues) + 1), [f'PC{i}' for i in range(1, len(eigenvalues) + 1)])
    plt.legend()
    plt.tight_layout()
    plt.savefig('scree_plot.png', bbox_inches='tight')

if __name__ == '__main__':
    pca_function()