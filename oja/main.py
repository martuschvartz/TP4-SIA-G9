import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

from oja.SimplePerceptron import SimplePerceptron


def main():
    # 1. load and standardize data
    europe_data = pd.read_csv('../europe.csv')
    country_names = europe_data['Country']
    numeric_data = europe_data.drop(columns=['Country'])  # remove country since its not numeric

    input_len = len(numeric_data.columns)
    # for now, use input len as grid size. In this case, the grid will have 49 neurons


    # for now, we will standardize the input
    scaled_array = StandardScaler().fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_data.columns)

    perceptron = SimplePerceptron(input_len)

    for epoch in range(5200):
        for row_idx, X in enumerate(scaled_array):  # X is a row (vector)
            pred = perceptron.predict(X)
            perceptron.update_weights(X)

       # print(f"Epoch {epoch+1}: Weights: {perceptron.weights}, Learning Rate: {perceptron.learning_rate:.4f}")
    # print("Final weights:")
    # for name, w in zip(numeric_data.columns, perceptron.weights):
    #     print(f"{name}: {w:.6f}")

    #print("\n\n\n--------------------------\n\n\n")


    compare_with_pc_demo(perceptron, numeric_data)



def compare_with_pc_demo(perceptron : SimplePerceptron, numeric_data : pd.DataFrame):

    # PC1 values from your pc_demo output, in the SAME order as numeric_data.columns
    pc1 = np.array([0.1249, -0.5005, 0.4065, -0.4829, 0.1881, -0.4757, 0.2717])

    w = perceptron.weights.copy()

    # Normalize both vectors
    pc1_norm = pc1 / np.linalg.norm(pc1)
    w_norm = w / np.linalg.norm(w)

    # Compare cosine similarity (1.0 or -1.0 means perfect alignment)
    cos_sim = np.dot(pc1_norm, w_norm)
    print(f"Cosine similarity with PC1: {cos_sim:.4f}")

    # If negative, flip Oja weights for direct comparison
    if cos_sim < 0:
        w_norm = -w_norm
        cos_sim = -cos_sim
        print("Flipped Oja weights for sign alignment.")

    # Print side-by-side
    print("Variable | Oja weight | PC1")
    for name, w_i, pc1_i in zip(numeric_data.columns, w_norm, pc1_norm):
        print(f"{name:12} {w_i:9.4f}   {pc1_i:9.4f}")












if __name__ == '__main__':
    main()