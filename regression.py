import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler instead of StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Initialize scalers outside any function to ensure consistency
x_scaler = MinMaxScaler()  # Use MinMaxScaler
y_scaler = MinMaxScaler()  # Use MinMaxScaler

def load_data(file_name):
    # Load data from the file
    data = np.loadtxt(file_name)
    x_data = data[:, 0].reshape(-1, 1)  # x values
    y_data = data[:, 1].reshape(-1, 1)  # y values
    return x_data, y_data

def scale_data(X_train, y_train, X_test, y_test):
    # Fit scalers on training data and transform both training and test data consistently
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = x_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    return X_train, y_train, X_test, y_test


# Linear regression model training function
def train_linear_regression(X_train, y_train):
    # X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    # theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    # intercept = theta[0]
    # coefficients = theta[1:]

    model = LinearRegression().fit(X_train, y_train)
    # model.intercept_ = intercept
    # model.coef_ = coefficients

    # print("Theta:", theta)
    return model


# Neural network architecture and training function
def train_neural_network(X, y, hidden_layer_sizes=[8, 8], epochs=40, learning_rate=0.05):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Define the neural network model
    class NeuralNetwork(nn.Module):
        def __init__(self, hidden_layer_sizes):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, hidden_layer_sizes[0])
            self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
            self.fc3 = nn.Linear(hidden_layer_sizes[1],1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            # x = torch.relu(self.fc3(x))
            return self.fc3(x)

    # Initialize model, loss function, and optimizer
    model = NeuralNetwork(hidden_layer_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    loss_values = []



    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)

        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"{epoch}: Initial Weights (before training):")
        print("fc1 weights:\n", model.fc1.weight.detach().numpy())
        print("fc2 weights:\n", model.fc2.weight.detach().numpy())
        print("fc3 weights:\n", model.fc3.weight.detach().numpy())


        if epoch % 10 == 0:  # print every 10th epoch
            print(f"Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {scheduler.get_last_lr()}")



        loss_values.append(loss.item())

    return model, loss_values


# Training function
def train_neural_network_plot(X_train, y_train, hidden_layer_configs, epochs=100, learning_rate=0.01):
    # Initialize a list to store the loss values for each hidden layer configuration
    loss_values_all_configs = []

    # Iterate over different hidden layer configurations
    for hidden_layers in hidden_layer_configs:
        print(f"Training model with hidden layers: {hidden_layers}")

        # Train the neural network with the current hidden layer configuration
        model, loss_values = train_neural_network(X_train, y_train, hidden_layer_sizes=hidden_layers, epochs=epochs,
                                                  learning_rate=learning_rate)

        # Store the loss values for this configuration
        loss_values_all_configs.append((hidden_layers, loss_values))

    # Plot the results for different hidden layer configurations
    plt.figure(figsize=(10, 8))
    for hidden_layers, loss_values in loss_values_all_configs:
        label = f"Training Loss {hidden_layers}"
        plt.plot(range(epochs), loss_values, label=label)

    # Add title and labels to the plot
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Training Loss (MSE)', fontsize=14)
    plt.title('Training Loss Curves for Different Hidden Layer Configurations', fontsize=16)
    plt.legend()
    plt.show()


##

def train_with_learning_rates(X_train, y_train, learning_rates, epochs=500):
    # Store training loss values for each learning rate
    train_loss_list = []

    for lr in learning_rates:
        # Train the model with the current learning rate
        nn_model, loss_values = train_neural_network(X_train, y_train, hidden_layer_sizes=[64, 32],
                                                     epochs=epochs, learning_rate=lr)

        # Calculate the training loss (last value from the loss_values)
        train_loss = loss_values[-1]
        print(f"{lr}: ")
        train_loss_list.append(train_loss)

    return train_loss_list


def train_with_kfold_cv(X, y, learning_rates, epochs=100, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Initialize KFold
    cv_loss_list = []

    for lr in learning_rates:
        fold_val_losses = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train the neural network on the current fold
            nn_model, _ = train_neural_network(X_train, y_train, hidden_layer_sizes=[64, 32],
                                               epochs=epochs, learning_rate=lr)

            # Predictions for the validation set
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_pred_nn = nn_model(X_val_tensor).numpy()

            # Inverse transform predictions and calculate MSE for validation set
            X_val_orig = x_scaler.inverse_transform(X_val)
            y_val_orig = y_scaler.inverse_transform(y_val)
            y_val_pred_orig_nn = y_scaler.inverse_transform(y_val_pred_nn)

            mse_val_nn = mean_squared_error(y_val_orig, y_val_pred_orig_nn)
            fold_val_losses.append(mse_val_nn)

        # Calculate the average cross-validation loss for this learning rate
        cv_loss_list.append(np.mean(fold_val_losses))

    return cv_loss_list


def plot_loss_comparison(learning_rates, train_loss_list, cv_loss_list):
    plt.figure(figsize=(10, 8))

    # Plot training loss
    plt.plot(learning_rates, train_loss_list, label='Training Loss', marker='o', linestyle='-', color='blue')

    # Plot cross-validation loss
    plt.plot(learning_rates, cv_loss_list, label='Cross-validation Loss (CV-5)', marker='o', linestyle='--',
             color='red')

    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Training and Cross-validation Loss vs Learning Rate', fontsize=16)
    plt.legend()
    plt.xscale('log')  # Use log scale to better visualize the trend of different learning rates
    plt.show()

    ##


# Plotting function for multiple hidden layer configurations
def plot_loss_curves(hidden_layer_configs, X_train, y_train, epochs=50, learning_rate=0.01):
    plt.figure(figsize=(10, 8))
    for hidden_layers in hidden_layer_configs:
        _, loss_values = train_neural_network(X_train, y_train, hidden_layers, epochs, learning_rate)
        plt.plot(loss_values, label=f"Hidden Layers: {hidden_layers}")
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.title('Training Loss Curves for Different Hidden Layer Configurations')
    plt.legend()
    plt.show()


# Function to plot neural network results for training and test data
def plot_neural_network_results(X_train_orig, y_train_orig, y_train_pred_orig_nn,
                                X_test_orig, y_test_orig, y_test_pred_orig_nn):
    # Plot training results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_orig, y_train_orig, label='Training Data', color='blue', alpha=0.5)
    plt.plot(X_train_orig, y_train_pred_orig_nn, label='Model Prediction (Train)', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Neural Network Prediction on Training Set")
    plt.legend()
    plt.show()

    # Plot test results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_orig, y_test_orig, label='Test Data', color='green', alpha=0.5)
    plt.plot(X_test_orig, y_test_pred_orig_nn, label='Model Prediction (Test)', color='orange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Neural Network Prediction on Testing Set")
    plt.legend()
    plt.show()


def descale(X_train, y_train, y_pred):
    X_train_orig = x_scaler.inverse_transform(X_train)
    y_train_orig = y_scaler.inverse_transform(y_train)
    y_train_pred_orig = y_scaler.inverse_transform(y_pred)
    return X_train_orig, y_train_orig, y_train_pred_orig


# Main function to execute all tasks
def main():
    # Load and scale data
    X_train, y_train = load_data("Data/regression_train.txt")
    X_test, y_test = load_data("Data/regression_test.txt")
    X_train, y_train, X_test, y_test = scale_data(X_train, y_train, X_test, y_test)

    # hidden_layer_configs = [[4, 4], [8, 8], [16, 8], [16, 16], [32, 16], [64, 32]]  # Different hidden layer configurations
    # train_neural_network_plot(X_train, y_train, hidden_layer_configs, epochs=175, learning_rate=0.01)

    # Train Linear Regression Model
    linear_model = train_linear_regression(X_train, y_train)
    y_train_pred_lreg = linear_model.predict(X_train)
    y_test_pred_lreg = linear_model.predict(X_test)

    # Train Neural Network
    nn_model, _ = train_neural_network(X_train, y_train, hidden_layer_sizes=[64, 32], epochs=280, learning_rate=0.05)
    # Get predictions from the neural network model
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_pred_nn = nn_model(X_train_tensor).numpy()
        y_test_pred_nn = nn_model(X_test_tensor).numpy()

    # Inverse transform predictions and data to original scale
    X_train_orig = x_scaler.inverse_transform(X_train)
    y_train_orig = y_scaler.inverse_transform(y_train)
    y_train_pred_orig_lreg = y_scaler.inverse_transform(y_train_pred_lreg)
    y_train_pred_orig_nn = y_scaler.inverse_transform(y_train_pred_nn)

    X_test_orig = x_scaler.inverse_transform(X_test)
    y_test_orig = y_scaler.inverse_transform(y_test)
    y_test_pred_orig_lreg = y_scaler.inverse_transform(y_test_pred_lreg)
    y_test_pred_orig_nn = y_scaler.inverse_transform(y_test_pred_nn)

    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

    # Get the training loss for each learning rate
    # train_loss_list = train_with_learning_rates(X_train, y_train, learning_rates, epochs=175)

    # Get the cross-validation loss for each learning rate (using 5-fold CV)
    # cv_loss_list = train_with_kfold_cv(X_train, y_train, learning_rates, epochs=500, n_splits=5)

    # Plot the results
    # plot_loss_comparison(learning_rates, train_loss_list, cv_loss_list)

    mse_train_lreg = mean_squared_error(y_train_orig, y_train_pred_orig_lreg)
    mse_test_lreg = mean_squared_error(y_test_orig, y_test_pred_orig_lreg)
    print("Neural Network - Training MSE:", mse_train_lreg)
    print("Neural Network - Testing MSE:", mse_test_lreg)

    # Calculate and print MSE for neural network model
    mse_train_nn = mean_squared_error(y_train_orig, y_train_pred_orig_nn)
    mse_test_nn = mean_squared_error(y_test_orig, y_test_pred_orig_nn)
    print("Neural Network - Training MSE:", mse_train_nn)
    print("Neural Network - Testing MSE:", mse_test_nn)

    # Plot the results (for both Linear Regression and Neural Network)
    plot_neural_network_results(X_train_orig, y_train_orig, y_train_pred_orig_nn,
                                X_test_orig, y_test_orig, y_test_pred_orig_nn)

if __name__ == "__main__":
    main()
