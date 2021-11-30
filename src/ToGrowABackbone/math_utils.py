import numpy as np
from numpy.random import choice


def weights2np_probs(weights: list) -> np.ndarray:
    probabilities = np.array(weights)
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def choose_index_by_prob(probs: np.ndarray):
    choices = np.arange(len(probs))
    chosen = choice(choices, p=probs)
    return chosen