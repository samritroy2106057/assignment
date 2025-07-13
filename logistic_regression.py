import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Data ---
try:
    X_df = pd.read_csv('Predictor Variable (X).csv')
    y_df = pd.read_csv('Response Variable (Y).csv')
    # Select only the first column of X for a single predictor
    X = X_df.iloc[:, 0].values.reshape(-1, 1)
    y = y_df.values
except FileNotFoundError:
    print("Error: Make sure 'Predictor Variable (X).csv' and 'Response Variable (Y).csv' are in the same directory.")
    exit()

# Normalize the predictor variable X
X_normalized = (X - np.mean(X)) / np.std(X)

# Add a column of ones to X for the intercept term (theta_0)
m = len(y)
X_b = np.c_[np.ones((m, 1)), X_normalized]


# --- 2. Gradient Descent Function Definitions ---

def batch_gradient_descent(X_b, y, learning_rate=0.5, n_iterations=1000, tolerance=1e-7):
    m = len(y)
    theta = np.zeros((2, 1))
    cost_history = []
    for iteration in range(n_iterations):
        predictions = X_b.dot(theta)
        error = predictions - y
        gradients = (1/m) * X_b.T.dot(error)
        theta = theta - learning_rate * gradients
        cost = (1/(2*m)) * np.sum(error**2)
        if iteration > 0 and abs(cost_history[-1] - cost) < tolerance:
            break
        cost_history.append(cost)
    return theta, cost_history

def stochastic_gradient_descent(X_b, y, learning_rate=0.05, n_epochs=50):
    m = len(y)
    theta = np.zeros((2, 1))
    cost_history = []
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi, yi = X_b[random_index:random_index+1], y[random_index:random_index+1]
            error = xi.dot(theta) - yi
            gradients = xi.T.dot(error)
            theta = theta - learning_rate * gradients
        total_error = X_b.dot(theta) - y
        cost_history.append((1/(2*m)) * np.sum(total_error**2))
    return theta, cost_history

def minibatch_gradient_descent(X_b, y, learning_rate=0.05, n_epochs=50, batch_size=10):
    m = len(y)
    theta = np.zeros((2, 1))
    cost_history = []
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled, y_shuffled = X_b[shuffled_indices], y[shuffled_indices]
        for i in range(0, m, batch_size):
            xi, yi = X_b_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
            error = xi.dot(theta) - yi
            gradients = (1/batch_size) * xi.T.dot(error)
            theta = theta - learning_rate * gradients
        total_error = X_b.dot(theta) - y
        cost_history.append((1/(2*m)) * np.sum(total_error**2))
    return theta, cost_history


# --- 3. Generate and Show Plots ---

# Plot for Question 3: Cost Function vs. Iteration
print("Generating Plot 1: Cost Function vs. Iteration...")
theta_q1, cost_history_q1 = batch_gradient_descent(X_b, y, learning_rate=0.5)
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history_q1[:50])), cost_history_q1[:50], 'b-', marker='o', markersize=4)
plt.title('Plot 1: Cost Function vs. Iteration (lr=0.5)', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (J)', fontsize=12)
plt.grid(True)
plt.show()

# Plot for Question 4: Dataset with Fitted Line
print("Generating Plot 2: Dataset with Fitted Line...")
y_predict = X_b.dot(theta_q1)
plt.figure(figsize=(10, 6))
plt.plot(X, y, "b.", label="Given Dataset")
plt.plot(X, y_predict, "r-", linewidth=2, label="Fitted Straight Line")
plt.title('Plot 2: Dataset with Fitted Regression Line', fontsize=14)
plt.xlabel('Predictor Variable (X)', fontsize=12)
plt.ylabel('Response Variable (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Plot for Question 5: Effect of Different Learning Rates
print("Generating Plot 3: Comparison of Learning Rates...")
_, cost_lr_005 = batch_gradient_descent(X_b, y, learning_rate=0.005, n_iterations=50)
_, cost_lr_05 = batch_gradient_descent(X_b, y, learning_rate=0.5, n_iterations=50)
_, cost_lr_5 = batch_gradient_descent(X_b, y, learning_rate=5, n_iterations=50)
plt.figure(figsize=(12, 7))
plt.plot(range(len(cost_lr_005)), cost_lr_005, 'r-', label='lr = 0.005')
plt.plot(range(len(cost_lr_05)), cost_lr_05, 'g-', label='lr = 0.5')
plt.plot(range(len(cost_lr_5)), cost_lr_5, 'b-', label='lr = 5')
plt.title('Plot 3: Cost Function Change for Different Learning Rates', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (J)', fontsize=12)
plt.ylim(0, 50)
plt.xlim(0, 50) # Set x-axis limit to 50 for consistent comparison
plt.legend()
plt.grid(True)
plt.show()

# Plot for Question 6: Comparison of Gradient Descent Methods
print("Generating Plot 4: Comparison of GD Methods...")
_, cost_batch = batch_gradient_descent(X_b, y, learning_rate=0.05, n_iterations=50)
_, cost_stochastic = stochastic_gradient_descent(X_b, y, learning_rate=0.05, n_epochs=50)
_, cost_minibatch = minibatch_gradient_descent(X_b, y, learning_rate=0.05, n_epochs=50)
plt.figure(figsize=(12, 7))
plt.plot(range(50), cost_batch, 'r-', label='Batch GD')
plt.plot(range(50), cost_stochastic, 'g-', label='Stochastic GD')
plt.plot(range(50), cost_minibatch, 'b-', label='Mini-Batch GD')
plt.title('Plot 4: Cost Function Comparison: Batch vs. Stochastic vs. Mini-Batch', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cost (J)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

