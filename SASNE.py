import numpy as np
from sklearn.manifold import TSNE
from construct_graph import construct_graph
from graph_distance import get_symbiharmonic_coords
import time


def SASNE_biharmonic(distance_matrix):
    """
    Compute graph + biharmonic coordinates on FULL dataset.
    """

    start_time = time.time()

    print("Constructing graph...")
    W = construct_graph(distance_matrix)

    print("--- %s seconds elapsed ---"
          % round(time.time() - start_time, 5))

    start_time = time.time()

    print("Computing graph distance...")
    Z, eigenval = get_symbiharmonic_coords(W)

    print("--- %s seconds elapsed ---"
          % round(time.time() - start_time, 5))

    return Z, eigenval


def SASNE_from_Z(Z, eigenval, n_components=3):
    """
    Run only the t-SNE stage.
    """

    init_Y = (
        #1e-4 * Z[:, [1, 2, 3]] * np.sqrt(eigenval[1])
        Z[:, [1, 2, 3]]
        #1e-4 * Z[:, [1, 2, 3]]
        #Z[:, [1, 2, 3]] * np.sqrt(eigenval[1])
    )

    perplexity = 0.9 * len(Z)
    print(f"perplexity={perplexity}")

    embedding = TSNE(
        n_components=n_components,
        init=init_Y,
        perplexity=perplexity
    ).fit_transform(Z)

    return embedding
