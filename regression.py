import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.GimpGradientFile import linear
from PIL.ImageOps import scale
import arviz as az
import pymc as pm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# from theano.compile import shape

torch.manual_seed(0)
np.random.seed(0)
# Initialise scalers outside any function to ensure consistency
x_scaler = StandardScaler()
y_scaler = StandardScaler()

def load_data(file_name):
    data = np.loadtxt(file_name)
    x_data = data[:, 0].reshape(-1, 1)  # x values
    y_data = data[:, 1].reshape(-1, 1)  # y values

    return x_data, y_data


def scale_data(X_train, y_train, X_test, y_test):
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    return X_train, y_train, X_test, y_test


def descale(X_train, y_train, y_pred):
    X_train_orig = x_scaler.inverse_transform(X_train)
    y_train_orig = y_scaler.inverse_transform(y_train)
    y_train_pred_orig = y_scaler.inverse_transform(y_pred)
    return X_train_orig, y_train_orig, y_train_pred_orig


# Function to add polynomial features
def add_polynomial_features(X, degree=3):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)


# Linear regression model training function with polynomial features
def train_linear_regression(X_train, y_train, degree=3):
    X_train_poly = add_polynomial_features(X_train, degree)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_poly, y_train)

    return model, X_train_poly


def test_lin_reg(model, X_train, y_train, X_test, y_test, degree=3):
    # Transform features to polynomial
    X_train_poly = add_polynomial_features(X_train, degree)
    X_test_poly = add_polynomial_features(X_test, degree)

    y_train_pred_lreg = model.predict(X_train_poly)
    y_test_pred_lreg = model.predict(X_test_poly)

    X_train_orig, y_train_orig, y_train_pred_orig = descale(X_train, y_train, y_train_pred_lreg)
    X_test_orig, y_test_orig, y_test_pred_orig = descale(X_test, y_test, y_test_pred_lreg)

    mse_train_lreg = mean_squared_error(y_train_orig, y_train_pred_orig)
    mse_test_lreg = mean_squared_error(y_test_orig, y_test_pred_orig)
    print(f"Train accuracy: {mse_train_lreg},   Test accuracy: {mse_test_lreg}")

    plot_test_results(X_train_orig, y_train_orig, y_train_pred_orig,
                      X_test_orig, y_test_orig, y_test_pred_orig, "Linear Regression (Polynomial)")



def train_neural_network(X, y, hidden_layer_sizes=[32, 16], epochs=90, learning_rate=0.018, weight_decay=1e-4, clip_value=1.0):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class NeuralNetwork(nn.Module):
        def __init__(self, hidden_layer_sizes):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, hidden_layer_sizes[0])
            self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
            self.fc3 = nn.Linear(hidden_layer_sizes[1], 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = NeuralNetwork(hidden_layer_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    loss_values = []

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)

        loss = criterion(outputs, y_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

        optimizer.step()
        scheduler.step()


        loss_values.append(loss.item())

    return model, loss_values



def train_bayesian_regression(X_train, y_train, y_scaler, num_samples=1000):
    with pm.Model() as model:
        w0 = pm.Normal("w0", mu=0, sigma=10)
        w1 = pm.Normal("w1", mu=0, sigma=10)
        w2 = pm.Normal("w2", mu=0, sigma=10)
        w3 = pm.Normal("w3", mu=0, sigma=10)

        noise = pm.Uniform("noise", lower=0, upper=200)

        y_est = (
            w0
            + w1 * X_train
            + w2 * X_train**2
            + w3 * X_train**3
        )
        likelihood = pm.Normal("likelihood", mu=y_est, sigma=noise, observed=y_train)

        trace = pm.sample(1250, tune=200, return_inferencedata=True)

    summary = az.summary(trace, var_names=["w0", "w1", "w2", "w3", "noise"])
    print(summary)


    return model, trace




def plot_loss_curves(learning_rate, X_train, y_train, epochs=150):
    plt.figure(figsize=(10, 8))
    for learning in learning_rate:
        _, loss_values = train_neural_network(X_train, y_train, [32, 16], epochs, learning)
        plt.plot(loss_values, label=f"Learning Rate: {learning}")
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.title('Training Loss Curves for Different Learning Rate Configurations')
    plt.legend()
    plt.show()


def plot_test_results(X_train_orig, y_train_orig, y_train_pred_orig,
                      X_test_orig, y_test_orig, y_test_pred_orig, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(X_train_orig, y_train_orig, label='Training Data', color='blue', alpha=0.5)
    axs[0].plot(X_train_orig, y_train_pred_orig, label='Model Prediction (Train)', color='red')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title(f"{title} Prediction on Training Set")
    axs[0].legend()

    axs[1].scatter(X_test_orig, y_test_orig, label='Test Data', color='green', alpha=0.5)
    axs[1].plot(X_test_orig, y_test_pred_orig, label='Model Prediction (Test)', color='orange')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title(f"{title} Prediction on Testing Set")
    axs[1].legend()

    fig.suptitle(f"{title} Performance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def test_nn(model, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_pred_nn = model(X_train_tensor).numpy()
        y_test_pred_nn = model(X_test_tensor).numpy()

    X_train_orig, y_train_orig, y_train_pred_nn_orig = descale(X_train, y_train, y_train_pred_nn)
    X_test_orig, y_test_orig, y_test_pred_nn_orig = descale(X_test, y_test, y_test_pred_nn)

    mse_train_nn = mean_squared_error(y_train_orig, y_train_pred_nn_orig)
    mse_test_nn = mean_squared_error(y_test_orig, y_test_pred_nn_orig)
    print(f"Train accuracy: {mse_train_nn}    Test accuracy: {mse_test_nn}")

    plot_test_results(X_train_orig, y_train_orig, y_train_pred_nn_orig,
                                X_test_orig, y_test_orig, y_test_pred_nn_orig, "Neural Network")




def main():
    # Load and scale data
    X_train, y_train = load_data("Data/regression_train.txt")
    X_test, y_test = load_data("Data/regression_test.txt")
    # X_train_orig, y_train_orig = X_train, y_train
    X_train, y_train, X_test, y_test = scale_data(X_train, y_train, X_test, y_test)

    degree = 3  # Degree of the polynomial
    linear_model, X_train_poly = train_linear_regression(X_train, y_train, degree)
    test_lin_reg(linear_model, X_train, y_train, X_test, y_test, degree)
    X_linreg = add_polynomial_features(X_test)
    lin_score = linear_model.score(X_linreg, y_test)
    print(f"Linear score: {lin_score}")


    nn_model, nn_loss = train_neural_network(X_train, y_train, hidden_layer_sizes=[32, 16], epochs=80,
                                             learning_rate=0.035, weight_decay=1e-5)
    test_nn(nn_model, X_train, y_train, X_test, y_test)
    nn_score = nn_model.score(X_test, y_test)
    print(f"NN score: {nn_score}")

    # bayesian_model, bayesian_trace = train_bayesian_regression(X_train_orig, y_train_orig, y_scaler, 1000)




if __name__ == "__main__":
    main()