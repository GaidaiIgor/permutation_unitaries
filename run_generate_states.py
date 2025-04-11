import itertools
import os
import pickle
from itertools import product

import numpy as np
import numpy.linalg as linalg
import numpy.random as rnd
import pandas as pd
from numpy import ndarray

from src.utilities.quantum import get_average_neighbors


def get_all_cluster_states(cluster: ndarray) -> list[str]:
    """ Returns a list of all amplitudes from a given cluster. """
    num_spanned_dims = np.sum(cluster == -1)
    state_labels = []
    for i in range(2 ** num_spanned_dims):
        i_bin = [int(c) for c in format(i, f'0{num_spanned_dims}b')]
        state_label = cluster.copy()
        state_label[state_label == -1] = i_bin
        state_labels.append(''.join([str(val) for val in state_label]))
    return state_labels


def generate_state(all_clusters: list[ndarray]) -> dict[str, complex]:
    """ Fills selected clusters with random normalized amplitudes. """
    all_state_labels = list(itertools.chain.from_iterable([get_all_cluster_states(cluster) for cluster in all_clusters]))
    all_amplitudes = rnd.uniform(0, 1, len(all_state_labels)) * np.exp(-1j * rnd.uniform(0, 2 * np.pi, len(all_state_labels)))
    all_amplitudes /= linalg.norm(all_amplitudes)
    state = {label: amplitude for label, amplitude in zip(all_state_labels, all_amplitudes)}
    return state


def generate_cluster_states():
    """ An entry point. Generates random sparse clustered states. Writes the results to specified folder. """
    num_states = 100
    num_qubits_dense_all = [7]
    num_qubits_all = [12]
    num_clusters_all = [2, 4, 8, 16, 32, 64, 128]

    generator = np.random.default_rng()
    iterable = list(product(num_qubits_all, num_qubits_dense_all, num_clusters_all))
    for num_qubits, num_qubits_dense, num_clusters in iterable:
        print(f'Qubits: {num_qubits}, Dense: {num_qubits_dense}, Clusters: {num_clusters}')
        out_path = f'data/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/states.pkl'
        states = []
        for i in range(num_states):
            num_cluster_dims = int(np.log2(2 ** num_qubits_dense / num_clusters))
            cluster_dims = generator.choice(num_qubits, num_cluster_dims, replace=False, shuffle=False)
            clusters = []
            for j in range(num_clusters):
                while True:
                    next_cluster = generator.choice(2, num_qubits)
                    next_cluster[cluster_dims] = -1
                    if not any([np.all(cluster == next_cluster) for cluster in clusters]):
                        break
                clusters.append(next_cluster)
            states.append(generate_state(clusters))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


def calculate_average_degree():
    """ An entry point. Iterates through the generated states and calculates the average number of non-zero amplitude neighbors (clustering). Writes the results to a csv file. """
    num_qubits = [10]
    num_qubits_dense = [5]
    num_clusters = [2, 4, 8, 16, 32]
    out_col_name = 'average_degree'
    data_folder_parent = 'data'

    iterable = list(product(num_qubits, num_qubits_dense, num_clusters))
    for item in iterable:
        print(f'Current iterable: {item}')
        data_folder = f'{data_folder_parent}/qubits_{item[0]}/dense_{item[1]}/clusters_{item[2]}'
        states_file_path = os.path.join(data_folder, 'states.pkl')
        with open(states_file_path, 'rb') as f:
            state_list = pickle.load(f)

        results = []
        for state in state_list:
            bases_array = np.array([[int(val) for val in basis] for basis in state])
            average_degree = get_average_neighbors(bases_array)
            results.append(average_degree)

        cx_counts_file_path = os.path.join(data_folder, 'cx_counts.csv')
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        df.to_csv(cx_counts_file_path, index=False)
        print(f'Avg deg: {np.mean(df[out_col_name])}\n')


if __name__ == "__main__":
    generate_cluster_states()
    # calculate_average_degree()
