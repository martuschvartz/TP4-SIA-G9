import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from kohonen.KohonenMap import KohonenMap

def main():
    epochs = 1000

    # 1. Load and standardize data
    europe_data = pd.read_csv('../europe.csv')
    # country_names = europe_data['Country']
    numeric_data = europe_data.drop(columns=['Country'])    # remove country since its not numeric

    #
    input_len = len(numeric_data.columns)
    k = int(np.sqrt(5 * np.sqrt(numeric_data.shape[0])))
    # for 28 countries: sqrt(5 * sqrt(28)) ~ sqrt(26.5) ~ 5
    # a 5x5 = 25 neuron grid fits much better for 28 datapoints -> 7x7 was way too large

    scaled_array = StandardScaler().fit_transform(numeric_data)
    # scaled_df = pd.DataFrame(scaled_array, columns=numeric_data.columns)

    kohonen_map = KohonenMap(
        n_inputs = input_len,
        init_radius = input_len,
        grid_size = k,
        dataset = scaled_array,
        radius_decay = epochs // 2,
        lr_decay = epochs * 10,
        init_learning_rate = 0.5,
    )

    # np.random.seed = 42

    # 2. Train
    print(f"start training... ")
    for epoch in range(epochs):
        indices = np.random.permutation(len(scaled_array))
        for idx in indices:
            X = scaled_array[idx]
            winner_cords = kohonen_map.find_winner_cords(X)
            kohonen_map.update_weights(X, winner_cords)

        # both called once per epoch
        kohonen_map.update_radius(epoch)
        kohonen_map.update_learning_rate(epoch)

        # uncomment to see weight map!

        # if epoch % 25 == 0:
        #     print(f"stored training for epoch {epoch} where ")
        #     gdp_idx = list(numeric_data.columns).index("GDP")
        #     weight_map(
        #         kohonen_map,
        #         variable_idx=gdp_idx,
        #         variable_name="GDP",
        #         save_path=f"output/weight_map_GDP_epoch_{epoch}.png"
        #     )


    # 3. Visualize

    # 3.1 SOM (heat map) of 1D
    bmu_hit_map(kohonen_map, scaled_array)


def bmu_hit_map(
        kohonen_map: KohonenMap,
        scaled_array: np.ndarray,
        save_path: str = "output/bmu_hit_map.png"):
    """
    For each country, find its winning neuron and count how many countries
    land on each neuron. Plots a heatmap of those counts.
    """
    grid_size = kohonen_map.grid.grid_size
    hits = np.zeros((grid_size, grid_size), dtype=int)

    for X in scaled_array:
        winner = kohonen_map.find_winner_cords(X)
        hits[winner.x, winner.y] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(hits, cmap="viridis", annot=True, fmt="d", cbar=True, ax=ax)
    ax.set_title("BMU hit map - countries per neuron")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



    # bmu_coords = []
    # for X in scaled_array:
    #     winner_cords = kohonen_map.find_winner_cords(X)
    #     bmu_coords.append(winner_cords.get_coordinates())
    #
    # hits = np.zeros((k, k), dtype=int)
    # for x, y in bmu_coords:
    #     hits[x, y] += 1
    #
    # plt.imshow(hits, cmap="viridis")
    # plt.colorbar(label="hits")
    # plt.title("BMU hit map")
    # plt.savefig("bmu_hit_map.png", dpi=150, bbox_inches="tight")
    # plt.close()




def weight_map(
        kohonen_map: KohonenMap,
        variable_idx: int,
        variable_name: str,
        save_path: str = "output/weight_map.png"):
    """
    For each neuron, read its learned weight for one feature.
    Plots a heatmap of those weights across the grid.

    Note: each neuron becomes a prototype that summarizes the countries mapped to it -> shows "avg" value for the cluster
    """
    grid_size = kohonen_map.grid.grid_size
    n_inputs = kohonen_map.grid.get_neuron(0, 0).n_inputs

    if not 0 <= variable_idx < n_inputs:
        raise ValueError(f"variable_idx {variable_idx} out of range, must be 0–{n_inputs - 1}")

    weights = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            weights[i, j] = kohonen_map.grid.get_neuron(i, j).weights[variable_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(weights, cmap="viridis", annot=False, cbar=True, ax=ax)
    ax.set_title(f"Weight map — {variable_name}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




















if __name__ == '__main__':
    main()