### Modified SASNE Implementation

## Introduction

This repository contains a modified implementation of Shape-aware Stochastic Neighbor Embedding (SASNE) developed for experiments involving precomputed SAGD matrix and the visualization of neural network checkpoints.

The original SASNE algorithm was proposed by Tobias Wangberg et al. and is described in:

Shape-aware Stochastic Neighbor Embedding, BMC Bioinformatics (2022).

The official implementation is available at:

https://github.com/tobiaswangberg/SASNE

## Disclaimer

This is not the original SASNE implementation.

It is an adaptation of the original code developed for this project. While the core methodology remains the same, several modifications have been introduced to accommodate our experimental setup.

## Modifications

Compared to the official implementation, the following changes have been made:

* The input is a precomputed distance matrix instead of a feature matrix where rows correspond to observations and columns correspond to features.
* The algorithm supports an arbitrary embedding dimension through

SASNE(distance_matrix, n_components=3)

instead of being restricted to a two-dimensional embedding.

* The initialization uses the first n_components nontrivial graph eigenvectors rather than being fixed to two eigenvectors.
* The graph construction has been modified to work directly with the supplied distance matrix instead of computing pairwise Euclidean distances from raw observations.
* Minor corrections and robustness improvements have been made to the recursive graph construction routines.

## Requirements

The implementation requires:

* Python 3.8 or newer

and the following Python packages:

* numpy
* scipy
* scikit-learn
* plotly

Input

The function expects a precomputed distance matrix

distance_matrix.shape == (N, N)

where

* N is the number of observations,
* the matrix must be square,
* all distances must be non-negative.

Example:

import numpy as np
D = np.load("wasserstein_matrix_cpu.npy")

Usage

import numpy as np
from SASNE import SASNE
D = np.load("wasserstein_matrix_cpu.npy")
embedding, Z = SASNE(
    D,
    n_components=3
)

where

* embedding is the low-dimensional embedding,
* Z contains the symbiharmonic coordinates computed by SASNE.

Output

For

embedding, Z = SASNE(D, n_components=3)

the returned embedding has shape

(N, 3)

Similarly,

embedding, Z = SASNE(D, n_components=2)

returns an embedding of shape

(N, 2)

# Notes

This implementation was developed specifically for experiments using precomputed Wasserstein distance matrices and higher-dimensional embeddings. Consequently, its interface differs from that of the official SASNE implementation, which expects a data matrix with observations as rows and features as columns.

For the original implementation and complete methodological details, please refer to the official SASNE repository and the associated publication.
