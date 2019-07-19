import numpy as np

from Source.clustering import KMeansClustering
from Source.driver import Driver
from Source.hidden_markov_model import HMM

from config import *


def get_input():
    while True:
        new_transaction = input('Please add your new transaction : ')
        if int(new_transaction) == TERMINATE:
            break
        new_transaction = k.predict(int(new_transaction))
        new_observation = np.append(observations[1:], [new_transaction])

        if h.detect_fraud(observations, new_observation, THRESHOLD):
            print('Fraud')
        else:
            print('Normal')


if __name__ == '__main__':
    d = Driver('./Data/train_data.txt')

    h = HMM(n_states=STATES, n_possible_observations=CLUSTERS)
    k = KMeansClustering()

    observations = k.run(d.get_data()[0:192])
    h.train_model(observations=list(observations), steps=STEPS)

    get_input()
