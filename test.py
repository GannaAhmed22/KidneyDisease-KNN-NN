#
# # -----------------------------
# # Activation & Transfer
# # -----------------------------
# def activate(weights, inputs):
#     activation = weights[-1]  # bias
#     for i in range(len(weights) - 1):
#         activation += weights[i] * inputs[i]
#     return activation
#
#
# def transfer_sigmoid(x, derivate=False):
#     if not derivate:
#         return 1.0 / (1.0 + exp(-x))
#     else:
#         return x * (1.0 - x)
#
#
# def transfer_tanh(x, derivate=False):
#     if not derivate:
#         return tanh(x)
#     else:
#         return 1.0 - tanh(x) ** 2
#
#
# # -----------------------------
# # Forward Propagation
# # -----------------------------
# def forward_propagate(network, row, transfer):
#     inputs = row
#     for layer in network:
#         new_inputs = []
#         for neuron in layer:
#             activation = activate(neuron['weights'], inputs)
#             neuron['output'] = transfer(activation, False)
#             new_inputs.append(neuron['output'])
#         inputs = new_inputs
#     return inputs
#
#
# # -----------------------------
# # Backward Propagation
# # -----------------------------
# def backward_propagate_error(network, expected, transfer):
#     for i in reversed(range(len(network))):
#         layer = network[i]
#         errors = []
#         if i == len(network) - 1:  # Output layer
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 errors.append(expected[j] - neuron['output'])
#         else:  # Hidden layers
#             for j in range(len(layer)):
#                 error = sum(neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1])
#                 errors.append(error)
#         for j in range(len(layer)):
#             neuron = layer[j]
#             neuron['delta'] = errors[j] * transfer(neuron['output'], True)
#
#
# # -----------------------------
# # Update Weights
# # -----------------------------
# def update_weights(network, row, l_rate):
#     for i in range(len(network)):
#         inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)):
#                 neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] += l_rate * neuron['delta']  # bias
#
#
# # -----------------------------
# # Prediction
# # -----------------------------
# def predict(network, row, transfer):
#     outputs = forward_propagate(network, row, transfer)
#     return outputs.index(max(outputs))
#
#
# def one_hot_encoding(n_outputs, row):
#     expected = [0 for _ in range(n_outputs)]
#     expected[int(row[-1])] = 1
#     return expected
#
#
# def accuracy_metric(actual, predicted):
#     correct = sum([1 for i in range(len(actual)) if actual[i] == predicted[i]])
#     return correct / float(len(actual)) * 100.0
#
#
# # -----------------------------
# # Training Loop
# # -----------------------------
# def train_network(network, train, test, l_rate, n_epoch, n_outputs, transfer):
#     history = []
#     for epoch in range(n_epoch):
#         sum_error = 0
#         for row in train:
#             outputs = forward_propagate(network, row, transfer)
#             expected = one_hot_encoding(n_outputs, row)
#             sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(n_outputs)])
#             backward_propagate_error(network, expected, transfer)
#             update_weights(network, row, l_rate)
#         predictions = [predict(network, row, transfer) for row in test]
#         actual = [int(row[-1]) for row in test]
#         acc = accuracy_metric(actual, predictions)
#         print(f">epoch={epoch + 1}, error={sum_error:.3f}, accuracy={acc:.2f}%")
#         history.append(acc)
#     return history
#
#
# # -----------------------------
# # Backpropagation Entrypoint
# # -----------------------------
# def back_propagation(train, test, l_rate, n_epoch, layers, transfer):
#     n_outputs = len(set([int(row[-1]) for row in train]))
#     network = initialize_network_custom(layers)
#     print(f"Network initialized with layers: {layers}")
#     history = train_network(network, train, test, l_rate, n_epoch, n_outputs, transfer)
#
#     # Final test accuracy
#     predictions = [predict(network, row, transfer) for row in test]
#     actual = [int(row[-1]) for row in test]
#     final_acc = accuracy_metric(actual, predictions)
#     print(f"\nFinal Accuracy on Test Set: {final_acc:.2f}%")
#
# # -----------------------------
# # Example Dataset & Run
# # -----------------------------
# def main():
#     layers = [23, 16, 1]
#     data = preprocessing.DataPreprocessor("Kidney_Disease_data_for_Classification_V2.csv",50,75)
#     # Create the neural network
#     nn = SimpleNeuralNetwork(layers,data)
#
#     # Print the initialized weights
#     nn.print_network()
#
# if __name__ == "__main__":
#     main()
# from random import seed, random
# from math import exp, tanh
#


# import random
#
# import preprocessing
#
#
# import random
# import math
#
# import random
# import math
#
# class SimpleNeuralNetwork:
#     def __init__(self, layers, data, learning_rate=0.1, iterations=1000):
#         self.layers = layers  # Network architecture (e.g., [4,5,1] for 4 input, 5 hidden, 1 output)
#         self.network = self._initialize_network()  # construct the network structure
#         self.dataset = data  # Training data as list of (inputs, target) tuples
#         self.learning_rate = learning_rate
#         self.iteration = iterations
#         self.activation = self.sigmoid
#         self.activation_derivative = self.sigmoid_derivative
#     def _initialize_network(self):
#         network = []
#         for i in range(1, len(self.layers)):
#             layer = []
#             for _ in range(self.layers[i]):
#                 weights = [random.random() for _ in range(self.layers[i - 1] + 1)]
#                 layer.append({'weights': weights})
#             network.append(layer)
#         return network
#
#     def sigmoid(self, activation):
#         return round(1 / (1 + math.exp(-activation)))
#
#     def sigmoid_derivative(self, output):
#         return round(output * (1 - output))
#
#     def forward_propagate(self, inputs):
#         for layer in self.network:
#             new_inputs = []
#             for neuron in layer:
#                 activation = neuron['weights'][-1]
#                 for i in range(len(neuron['weights']) - 1):
#                     activation += neuron['weights'][i] * inputs[i]
#
#                 neuron['output'] = self.activation(activation)
#                 new_inputs.append(neuron['output'])
#             inputs = new_inputs
#         return inputs
#
#     def backward_propagate_error(self, expected):
#         for i in reversed(range(len(self.network))):
#             layer = self.network[i]
#             errors = []
#             if i == len(self.network) - 1:
#                 # Output layer
#                 for j in range(len(layer)):
#                     neuron = layer[j]
#                     errors.append(expected[j] - neuron['output'])
#             else:
#                 # Hidden layers
#                 for j in range(len(layer)):
#                     error = 0.0
#                     for neuron in self.network[i + 1]:
#                         error += neuron['weights'][j] * neuron['delta']
#                     errors.append(error)
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 neuron['delta'] = errors[j] * self.sigmoid_derivative(neuron['output'])
#
#     def update_weights(self, row):
#         for i in range(len(self.network)):
#             inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in self.network[i - 1]]
#             for neuron in self.network[i]:
#                 for j in range(len(inputs)):
#                     neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
#                 neuron['weights'][-1] += self.learning_rate * neuron['delta']  # Bias update
#
#     def train(self):
#         for epoch in range(self.epochs):
#             total_error = 0
#             for row in self.dataset:
#                 outputs = self.forward_propagate(row[:-1])
#                 expected = [0 for _ in range(self.layers[-1])]
#                 expected[int(row[-1])] = 1
#                 total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
#                 self.backward_propagate_error(expected)
#                 self.update_weights(row)
#             if epoch % 100 == 0 or epoch == self.epochs - 1:
#                 print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")
#
#     def predict(self, row):
#         outputs = self.forward_propagate(row)
#         return outputs.index(max(outputs))
#
#     def print_network(self):
#         for i, layer in enumerate(self.network):
#             print(f"Layer {i + 1}:")
#             for j, neuron in enumerate(layer):
#                 print(f"  Neuron {j + 1} weights: {neuron['weights']}")

#--------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import random

# Ensure DataPreprocessor is correctly imported
from preprocessing import DataPreprocessor  # Ensure this matches your file structure

#
# class SimpleNeuralNetwork:
#     def __init__(self, layers, data, learning_rate=0.1, epochs=10000, lambda_param=0.01):
#         self.layers = layers
#         self.network = self._initialize_network()
#         self.dataset = data
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.lambda_param = lambda_param  # Regularization strength
#         self.activation = self.sigmoid
#         self.activation_derivative = self.sigmoid_derivative
#         self.best_loss = float('inf')
#         self.patience = 100  # Number of epochs to wait before early stopping
#         self.no_improvement_count = 0
#
#     def _initialize_network(self):
#         network = []
#         for i in range(1, len(self.layers)):
#             layer = []
#             for _ in range(self.layers[i]):
#                 weights = [random.random() for _ in range(self.layers[i - 1] + 1)]  # Including bias
#                 layer.append({'weights': weights})
#             network.append(layer)
#         return network
#
#     def sigmoid(self, activation):
#         return 1 / (1 + math.exp(-activation))
#
#     def sigmoid_derivative(self, output):
#         return output * (1 - output)
#
#     def forward_propagate(self, inputs):
#         for layer in self.network:
#             new_inputs = []
#             for neuron in layer:
#                 activation = neuron['weights'][-1]  # Bias term
#                 for i in range(len(neuron['weights']) - 1):
#                     activation += neuron['weights'][i] * inputs[i]
#                 neuron['output'] = self.activation(activation)
#                 new_inputs.append(neuron['output'])
#             inputs = new_inputs
#         return inputs
#
#     def backward_propagate_error(self, expected):
#         for i in reversed(range(len(self.network))):
#             layer = self.network[i]
#             errors = []
#             if i == len(self.network) - 1:
#                 for j in range(len(layer)):
#                     neuron = layer[j]
#                     errors.append(expected[j] - neuron['output'])
#             else:
#                 for j in range(len(layer)):
#                     error = 0.0
#                     for neuron in self.network[i + 1]:
#                         error += neuron['weights'][j] * neuron['delta']
#                     errors.append(error)
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 neuron['delta'] = errors[j] * self.activation_derivative(neuron['output'])
#
#     def update_weights(self, row):
#         for i in range(len(self.network)):
#             inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in self.network[i - 1]]
#             for neuron in self.network[i]:
#                 for j in range(len(inputs)):
#                     # L2 regularization added here
#                     neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j] - \
#                                             self.learning_rate * self.lambda_param * neuron['weights'][j]
#                 neuron['weights'][-1] += self.learning_rate * neuron['delta']  # Bias update
#
#     def train(self):
#         # Find the number of unique classes in the target column (row[-1])
#         num_classes = len(set(row[-1] for row in self.dataset))
#
#         for epoch in range(self.epochs):
#             total_error = 0
#             for row in self.dataset:
#                 outputs = self.forward_propagate(row[:-1])
#
#                 # Initialize the expected list with zeros for the correct number of classes
#                 expected = [0 for _ in range(num_classes)]
#
#                 # Ensure the target value (row[-1]) is within bounds
#                 target = int(row[-1])  # Convert to int, in case it's a string
#                 if 0 <= target < num_classes:
#                     expected[target] = 1
#                 else:
#                     print(f"Warning: Target value {target} out of range, skipping this row.")
#                     continue  # Skip the current row if the target is out of range
#
#                 total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
#                 self.backward_propagate_error(expected)
#                 self.update_weights(row)
#
#             # Early stopping
#             if total_error < self.best_loss:
#                 self.best_loss = total_error
#                 self.no_improvement_count = 0
#             else:
#                 self.no_improvement_count += 1
#
#             # Early stopping condition
#             if self.no_improvement_count >= self.patience:
#                 print(f"Early stopping at epoch {epoch + 1}, total error: {total_error:.6f}")
#                 break
#
#             if epoch % 100 == 0 or epoch == self.epochs - 1:
#                 print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")
#
#     def predict(self, row):
#         outputs = self.forward_propagate(row)
#         return outputs.index(max(outputs))  # Return the index of the maximum output

#
# def main():
#     # === Preprocess Data ===
#     processor = DataPreprocessor("Kidney_Disease_data_for_Classification_V2.csv", 100, 75)
#     processor.process()  # Call the process() method to preprocess the data
#     train_set, test_set = processor.get_sets()  # Now get the training and testing sets
#
#     # Ensure the number of unique classes in the target column is correct
#     num_classes = len(set(row[-1] for row in train_set))  # Count unique classes in training set
#
#     # === Train Neural Network ===
#     nn = SimpleNeuralNetwork(layers=[24, 16, num_classes], data=train_set, learning_rate=0.1, epochs=1000,
#                              lambda_param=0.01)
#     nn.train()
#
#     # === Evaluate ===
#     correct = 0
#     for row in test_set:
#         prediction = nn.predict(row[:-1])
#         actual = int(row[-1])  # Ensure actual is cast to integer for correct comparison
#         if prediction == actual:
#             correct += 1
#
#     accuracy = correct / len(test_set) * 100
#     print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{len(test_set)})")
#
#
# if __name__ == "__main__":
#     main()
import math
import random

from preprocessing import DataPreprocessor

class SimpleNeuralNetwork:
    def __init__(self, layers, data, learning_rate=0.001, epochs=10000, lambda_param=0.01):
        self.layers = layers
        self.network = self._initialize_network()
        self.dataset = data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param  # Regularization strength
        self.activation = self.sigmoid
        self.activation_derivative = self.sigmoid_derivative
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def _initialize_network(self):
        network = []
        for i in range(1, len(self.layers)):
            layer = []
            for _ in range(self.layers[i]):
                weights = [random.random() for _ in range(self.layers[i - 1] + 1)]  # Including bias
                layer.append({'weights': weights})
            network.append(layer)
        return network

    def sigmoid(self, activation):
        return 1 / (1 + math.exp(-activation))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def forward_propagate(self, inputs):
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = neuron['weights'][-1]  # Bias
                for i in range(len(neuron['weights']) - 1):
                    activation += neuron['weights'][i] * inputs[i]
                neuron['output'] = self.activation(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i == len(self.network) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron['weights'][j] * neuron['delta']
                    errors.append(error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.activation_derivative(neuron['output'])

    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    # L2 Regularization added here
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[
                        j] - self.learning_rate * self.lambda_param * neuron['weights'][j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']  # Bias

    def train(self):
        for epoch in range(self.epochs):
            total_error = 0
            for row in self.dataset:
                outputs = self.forward_propagate(row[:-1])
                expected = [0 for _ in range(self.layers[-1])]  # Make sure layers[-1] matches number of classes
                expected[int(row[-1])] = 1  # Set the correct class to 1
                total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
                self.backward_propagate_error(expected)
                self.update_weights(row)

            # Check for early stopping based on best_loss
            if total_error < self.best_loss:
                self.best_loss = total_error
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count > 1000:
                print("Early stopping...")
                break

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

def main():
    # === Preprocess Data ===
    processor = DataPreprocessor("Kidney_Disease_data_for_Classification_V2.csv", 100, 75)
    processor.process()
    train_set, test_set = processor.get_sets()
    target_column = [row[-1] for row in train_set]
    num_classes = len(set(target_column))
    print(f"Number of unique classes in target: {num_classes}")

    # === Train Neural Network ===
    nn = SimpleNeuralNetwork(layers=[24, 20, 2], data=train_set, learning_rate=0.01, epochs=800,
                             lambda_param=0.001)
    nn.train()

    # === Evaluate ===
    correct = 0
    for row in test_set:
        prediction = nn.predict(row[:-1])
        actual = int(row[-1])  # Ensure actual is cast to integer for correct comparison
        if prediction == actual:
            correct += 1

    accuracy = correct / len(test_set) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{len(test_set)})")

if __name__ == "__main__":
    main()
