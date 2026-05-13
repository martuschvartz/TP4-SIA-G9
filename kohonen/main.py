import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import KohonenMap

def main():


    # 1. load and standardize data
    europe_data = pd.read_csv('../europe.csv')
    country_names = europe_data['Country']
    numeric_data = europe_data.drop(columns=['Country'])    # remove country since its not numeric

    input_len = len(numeric_data.columns)
    # for now, use input len as grid size. In this case, the grid will have 49 neurons
    grid_size = input_len

    # for now, we will standardize the input
    scaled_array = StandardScaler().fit_transform(numeric_data)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_data.columns)

    kohonen_map = KohonenMap.KohonenMap(input_len,grid_size,dataset=scaled_array)

    # np.random.seed = 42


    for epoch in range(1000):
        indices = np.random.permutation(len(scaled_array))
        for idx in indices:
            X = scaled_array[idx]
            winner_cords = kohonen_map.find_winner_cords(X)
            kohonen_map.update_weights(X, winner_cords)
            kohonen_map.update_radius()
            kohonen_map.update_learning_rate(epoch)
        # Count hits per cell

        # Compute BMU hit map AFTER training
    bmu_coords = []
    for X in scaled_array:
        winner_cords = kohonen_map.find_winner_cords(X)
        bmu_coords.append(winner_cords.get_cords())

    hits = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in bmu_coords:
        hits[x, y] += 1

    plt.imshow(hits, cmap="viridis")
    plt.colorbar(label="hits")
    plt.title("BMU hit map")
    plt.savefig("bmu_hit_map.png", dpi=150, bbox_inches="tight")
    plt.close()





    # for i in range(1000 * input_len):
    #     X = scaled_array[np.random.randint(len(scaled_array))]
    #     winner_cords = kohonen_map.find_winner_cords(X)
    #     kohonen_map.update_weights(X, winner_cords)
    #
    #     if i % 20 == 0:
    #         print(f" Updating radius and learning rate")
    #         kohonen_map.update_radius()
    #         kohonen_map.update_learning_rate()
    #         print(f" New Radius: {kohonen_map.radius}")
    #         print(f" New Learning rate: {kohonen_map.learning_rate}")





















if __name__ == '__main__':
    main()