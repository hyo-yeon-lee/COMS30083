from venv import create

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Load rewards data and convert to integer symbols if necessary
rewards_data = np.loadtxt('Data/rewards.txt', dtype=int).reshape(-1, 1)

# Define the grid size and reward states
n_states = 9  # 3x3 grid
n_rewards = 3  # rewards {0, 1, 2}

# Define transition matrix based on grid rules
def create_transition_matrix():
    transmat = np.zeros((n_states, n_states))
    for i in range(3):
        for j in range(3):
            state = i * 3 + j
            neighbors = []
            if i > 0: neighbors.append((i-1) * 3 + j)  # up
            if i < 2: neighbors.append((i+1) * 3 + j)  # down
            if j > 0: neighbors.append(i * 3 + (j-1))  # left
            if j < 2: neighbors.append(i * 3 + (j+1))  # right
            prob = 1 / len(neighbors)
            for n in neighbors:
                transmat[state, n] = prob
    return transmat

true_transmat = create_transition_matrix()

# Visualization function for the transition matrix
def visualize_transition_matrix(transmat, title="Transition Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()

# Code Task 14: EM Algorithm without Known Transition Probabilities
def train_hmm_without_transitions(rewards_data):
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=100)  # random init
    model.n_features = n_rewards
    model.fit(rewards_data)
    # print("------------------------------------WITHOUT------------------------------------")
    #
    # print("Learned Emission Matrix (Task 14):")
    # print("Initial probabilities: ", model.startprob_)
    # print("Initial prior probabilities: ", model.startprob_prior)
    # # print("Transmat probability: " , model.transmat_)
    # print("emission probability: " , model.emissionprob_)
    # print("emission prior probability: " , model.emissionprob_prior)
    #
    # print(model.emissionprob_)
    # visualize_transition_matrix(model.transmat_, "True Transition Matrix Not Probided")
    return model, model.emissionprob_

# Code Task 15: EM Algorithm with Known Transition Probabilities
def train_hmm_with_transitions(rewards_data, true_transmat):
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, tol=1e-4)
    model.n_features = n_rewards
    model.transmat_ = true_transmat  # set known transitions
    model.params = "se"  # only estimate startprob and emissionprob
    model.init_params = "se"
    model.fit(rewards_data)
    # print("Learned Emission Matrix (Task 15):")
    # print(model.emissionprob_)
    # visualize_transition_matrix(model.transmat_, "True Transition Matrix Provided")
    # print("------------------------------------WITH------------------------------------")
    # print("Learned Emission Matrix (Task 14):")
    # print("Initial probabilities: ", model.startprob_)
    # print("Initial prior probabilities: ", model.startprob_prior)
    # print("Transmat probability: " , model.transmat_)
    # print("emission probability: " , model.emissionprob_)
    # print("emission prior probability: " , model.emissionprob_prior)
    return model, model.emissionprob_


def plot_transition_comparison(true_mat, learned_mat, title1="Learned Transition Matrix", title2="Given True Transition Matrix"):
    plt.figure(figsize=(14, 6))

    # Plot the true transition matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(true_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title1)
    plt.xlabel("To State")
    plt.ylabel("From State")

    # Plot the learned transition matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(learned_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title2)
    plt.xlabel("To State")
    plt.ylabel("From State")

    plt.tight_layout()
    plt.show()


def plot_emission_difference(emission1, emission2, title="Difference in Emission Matrices (Absolute)"):
    """
    Plots a heatmap of the absolute differences between two emission matrices.

    Parameters:
        emission1 (np.array): First emission matrix.
        emission2 (np.array): Second emission matrix.
        title (str): Title for the heatmap.
    """
    # Calculate the absolute difference
    difference_matrix = np.abs(emission1 - emission2)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(difference_matrix, annot=True, fmt=".2f", cmap="Reds", cbar=True)
    plt.title(title)
    plt.xlabel("Rewards")
    plt.ylabel("States")
    plt.show()

# Example usage in your code


# Run tasks and visualize
# Run tasks and visualize
without_model, without_prob = train_hmm_without_transitions(rewards_data)
with_model, with_prob = train_hmm_with_transitions(rewards_data, true_transmat)

# Extract the learned transition matrix from the model
learned_transmat = without_model.transmat_
given_transmat = with_model.transmat_

# Compare true and learned transition matrices
plot_transition_comparison(learned_transmat, given_transmat)

# Compare emission probabilities
plot_emission_difference(without_prob, with_prob)

# Print the MSE for the emission probabilities
print("MSE: ", mean_squared_error(with_prob, without_prob))




