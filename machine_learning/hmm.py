# -*- coding: utf-8 -*-
# @Time    : 16/3/2023 6:44 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: hmm.py

"""
Copyright (C) 2022 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the terms of the JXW license, 
which unfortunately won't be written for another century.

You should have received a copy of the JXW license with this file. If not, 
please write to: josephchenhk@gmail.com
"""
import numpy as np

# Define the HMM parameters
states = hidden_states = ['H1', 'H2']
obs = observations = ['O1', 'O2', 'O3']
P0 = start_prob = {'H1': 0.6, 'H2': 0.4}
PT = transition_prob = {
    'H1': {'H1': 0.7, 'H2': 0.3},
    'H2': {'H1': 0.4, 'H2': 0.6}
}
PE = emission_prob = {
    'H1': {'O1': 0.1, 'O2': 0.4, 'O3': 0.5},
    'H2': {'O1': 0.6, 'O2': 0.3, 'O3': 0.1}
}

# Define the sequence of observations
obs_seq = ['O1', 'O1', 'O2', 'O3']

# Implement the Viterbi algorithm (to find the most likely sequence of hidden states)
def viterbi(obs_seq, hidden_states, start_prob, transition_prob, emission_prob):
    T = len(obs_seq)
    N = len(hidden_states)

    # Initialize the probability matrix and the backpointer matrix
    prob = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=int)

    # Set the initial probabilities
    for s in range(N):
        # prob[0][s] = start_prob[hidden_states[s]] * emission_prob[hidden_states[s]][obs_seq[0]]
        prob[0][s] = P0[states[s]] * PE[states[s]][obs_seq[0]]

    # Calculate the probabilities for each time step
    for t in range(1, T):
        for s in range(N):
            max_prob = 0
            max_state = 0
            for s_prev in range(N):
                curr_prob = prob[t-1][s_prev] * PT[states[s_prev]][states[s]] \
                            * PE[states[s]][obs_seq[t]]
                if curr_prob > max_prob:
                    max_prob = curr_prob
                    max_state = s_prev
            prob[t][s] = max_prob
            backpointer[t][s] = max_state

    # Find the path with the highest probability
    max_prob = max(prob[T-1])
    max_state = np.argmax(prob[T-1])
    path = [hidden_states[max_state]]
    for t in range(T-1, 0, -1):
        max_state = backpointer[t][max_state]
        path.insert(0, hidden_states[max_state])

    return path

# Find the most likely sequence of hidden states
path = viterbi(obs_seq, hidden_states, start_prob, transition_prob, emission_prob)
print("Most likely sequence of hidden states:", path)