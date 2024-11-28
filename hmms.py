import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

rewards_data = np.loadtxt('rewards.txt', dtype=int).reshape(-1, 1)
n_states = 9
n_rewards = 3


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


def visualize_transition_matrix(transmat, title="Transition Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()

# Code Task 14
def train_hmm_without_transitions(rewards_data):
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=50, tol=1e-4)  # random init
    model.n_features = n_rewards
    model.fit(rewards_data)
    return model, model.emissionprob_


# Code Task 15
def train_hmm_with_transitions(rewards_data, true_transmat):
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=50, tol=1e-4)
    model.n_features = n_rewards
    model.transmat_ = true_transmat
    model.params = "se"
    model.init_params = "se"
    model.fit(rewards_data)
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


def plot_transition_difference(matrix1, matrix2, title="Difference in Transition Matrices (Absolute)"):
    difference_matrix = np.abs(matrix1 - matrix2)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(difference_matrix, annot=True, fmt=".2f", cmap="Reds", cbar=True)
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()


def plot_emission_difference(emission1, emission2, title="Difference in Emission Matrices (Absolute)"):

    difference_matrix = np.abs(emission1 - emission2)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(difference_matrix, annot=True, fmt=".2f", cmap="Reds", cbar=True)
    plt.title(title)
    plt.xlabel("Rewards")
    plt.ylabel("States")
    plt.show()


def plot_emission_matrices_with_coordinates(emission1, emission2,
                                            title1="Emission Matrix (Without Transitions)",
                                            title2="Emission Matrix (With Transitions)"):
    # Create the (row, column) coordinates for states
    state_labels = [f"({i},{j})" for i in range(3) for j in range(3)]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap for emission1
    sns.heatmap(emission1, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=[0, 1, 2], yticklabels=state_labels, ax=axs[0])
    axs[0].set_title(title1)
    axs[0].set_xlabel("Rewards")
    axs[0].set_ylabel("States")

    # Heatmap for emission2
    sns.heatmap(emission2, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=[0, 1, 2], yticklabels=state_labels, ax=axs[1])
    axs[1].set_title(title2)
    axs[1].set_xlabel("Rewards")
    axs[1].set_ylabel("States")

    plt.tight_layout()
    plt.show()



without_model, without_prob = train_hmm_without_transitions(rewards_data)
with_model, with_prob = train_hmm_with_transitions(rewards_data, true_transmat)

learned_transmat = without_model.transmat_
given_transmat = with_model.transmat_

plot_emission_matrices_with_coordinates(without_prob, with_prob)
plot_transition_comparison(learned_transmat, given_transmat)





