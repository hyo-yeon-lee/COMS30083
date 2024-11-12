# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler


# Load training data
def load_data(filename):
    data = np.loadtxt(filename)
    x_data = data[:, 0].reshape(-1, 1)  # x values
    y_data = data[:, 1].reshape(-1, 1)  # y values
    return x_data, y_data


# Code Task 10: Train a linear regression model
def train_linear_regression(x_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    return linear_model


# Code Task 11: Defining nn architecture
def train_neural_network(X, y, epochs=2000, learning_rate=0.01):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", label="Data Points")
    plt.plot(X, y_pred, color="red", label="Neural Network Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network Regression")
    plt.legend()
    plt.show()

    mse = mean_squared_error(y, y_pred)
    print("Neural Network MSE:", mse)
    return model



def test_neural_network_model(model, X, y):
    X_tensor = torch.FloatTensor(X).view(-1, 1)  # Ensure shape compatibility

    model.eval()
    with torch.no_grad():
        nn_predictions = model(X_tensor).numpy().flatten()

    mse_test = mean_squared_error(y, nn_predictions)
    print(f"Test Mean Squared Error: {mse_test}")

    # Sort X for smooth plotting if test data isn't ordered
    sorted_indices = np.argsort(X.flatten())
    X_sorted = X.flatten()[sorted_indices]
    nn_predictions_sorted = nn_predictions[sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Test Data')
    plt.plot(X_sorted, nn_predictions_sorted, color='red', label='Model Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural Network Regression on Test Data')
    plt.legend()
    plt.show()



# Code Task 12: Bayesian regression model using PyMC
def train_bayesian_model(x_train, y_train):
    with pm.Model() as model:
        # Define priors
        slope = pm.Normal('Slope', mu=0, sigma=10)
        intercept = pm.Normal('Intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('Sigma', sigma=1)

        mean = intercept + slope * x_train.flatten()
        y_obs = pm.Normal('Y_obs', mu=mean, sigma=sigma, observed=y_train)

        trace = pm.sample(1000, cores=1)


    with model:
        az.plot_trace(trace)
        plt.show()

    with model:
        summary = az.summary(trace)
        print(summary)

    return trace


def plot_results(x, y, y_pred, title, is_train=True):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data', color='blue', alpha=0.5)
    plt.plot(x, y_pred, label='Model Prediction', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{title} - {'Train' if is_train else 'Test'} Data")
    plt.legend()
    plt.show()


# Code Task 13: Produce predictions and compute MSE for linear and neural network models
def evaluate_models(linear_model, neural_network, x_test, y_test):
    # Linear regression predictions and MSE
    y_pred_linear = linear_model.predict(x_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)

    # Neural network predictions and MSE
    x_test_tensor = torch.FloatTensor(x_test)
    y_pred_nn = neural_network(x_test_tensor).detach().numpy()
    mse_nn = mean_squared_error(y_test, y_pred_nn)

    return mse_linear, mse_nn, y_pred_linear, y_pred_nn


# Main function to execute all tasks
def main():
    # Load the training and test data
    x_train, y_train = load_data("Data/regression_train.txt")
    x_test, y_test = load_data("Data/regression_test.txt")

    # Task 10: Train Linear Regression Model
    linear_model = train_linear_regression(x_train, y_train)

    # Task 11: Train Neural Network Model
    neural_network = train_neural_network(x_train, y_train)

    # Task 12: Bayesian Regression Model (we only need to train it, no prediction required on test set)
    bayesian_trace = train_bayesian_model(x_train, y_train)

    # Task 13: Evaluate models on test set
    mse_linear, mse_nn, y_pred_linear_train, y_pred_nn_train = evaluate_models(linear_model, neural_network, x_train, y_train)
    mse_linear_test, mse_nn_test, y_pred_linear_test, y_pred_nn_test = evaluate_models(linear_model, neural_network, x_test, y_test)

    # Plot for Linear Regression Model
    plot_results(x_train, y_train, y_pred_linear_train, "Linear Regression", is_train=True)
    plot_results(x_test, y_test, y_pred_linear_test, "Linear Regression", is_train=False)

    # Plot for Neural Network Model
    plot_results(x_train, y_train, y_pred_nn_train, "Neural Network", is_train=True)
    plot_results(x_test, y_test, y_pred_nn_test, "Neural Network", is_train=False)

    # Print results
    print(f"Linear Regression MSE: {mse_linear}")
    print(f"Neural Network MSE: {mse_nn}")


# Run main
if __name__ == "__main__":
    main()
