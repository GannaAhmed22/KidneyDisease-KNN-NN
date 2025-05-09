# import math
# import random
#
# from preprocessing import DataPreprocessor
#
#
# class SimpleNeuralNetwork:
#     def __init__(self, layers, data, learning_rate=0.001, epochs=10000, lambda_param=0.01):
#         self.layers = layers
#         self.network = self._initialize_network()
#         self.dataset = data
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.lambda_param = lambda_param  # Regularization strength
#         self.activation = self.sigmoid
#         self.activation_derivative = self.sigmoid_derivative
#         self.best_loss = float('inf')
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
#     def relu(self, activation):
#         return max(0, activation)
#
#     def relu_derivative(self, output):
#         return 1 if output > 0 else 0
#     def forward_propagate(self, inputs):
#         for layer in self.network:
#             new_inputs = []
#             for neuron in layer:
#                 activation = neuron['weights'][-1]  # Bias
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
#                     # L2 Regularization added here
#                     neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[
#                         j] - self.learning_rate * self.lambda_param * neuron['weights'][j]
#                 neuron['weights'][-1] += self.learning_rate * neuron['delta']  # Bias
#
#     def train(self):
#         patience = 100  # Number of epochs without improvement before stopping
#         best_loss = float('inf')
#         no_improvement_count = 0
#
#         for epoch in range(self.epochs):
#             total_error = 0
#             for row in self.dataset:
#                 outputs = self.forward_propagate(row[:-1])
#                 expected = [0 for _ in range(self.layers[-1])]
#                 expected[int(row[-1])] = 1  # Set the correct class to 1
#                 total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
#                 self.backward_propagate_error(expected)
#                 self.update_weights(row)
#
#             # Check for improvement
#             if total_error < best_loss:
#                 best_loss = total_error
#                 no_improvement_count = 0
#             else:
#                 no_improvement_count += 1
#
#             # Early stopping check
#             if no_improvement_count >= patience:
#                 print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
#                 break
#
#             if epoch % 100 == 0 or epoch == self.epochs - 1:
#                 print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")
#
#     def predict(self, row):
#         outputs = self.forward_propagate(row)
#         return outputs.index(max(outputs))
#
#
# def main():
#     # === Preprocess Data ===
#     processor = DataPreprocessor("Kidney_Disease_data_for_Classification_V2.csv", 100, 75)
#     processor.process()
#     train_set, test_set = processor.get_sets()
#     target_column = [row[-1] for row in train_set]
#     num_classes = len(set(target_column))
#     print(f"Number of unique classes in target: {num_classes}")
#
#     # === Train Neural Network ===
#     nn = SimpleNeuralNetwork(layers=[24, 20, 2], data=train_set, learning_rate=0.001, epochs=1000,
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
#
import math
import random
import numpy as np
from preprocessing import DataPreprocessor


class SimpleNeuralNetwork:
    def __init__(self, layers, data, learning_rate=0.001, epochs=10000, lambda_param=0.01):
        self.layers = layers
        self.network = self._initialize_network()
        self.dataset = data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param  # Regularization strength
        self.activation = self.relu
        self.activation_derivative = self.relu_derivative
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def _initialize_network(self):
        network = []
        for i in range(1, len(self.layers)):
            layer = []
            for _ in range(self.layers[i]):
                weights = np.random.randn(self.layers[i - 1] + 1) * np.sqrt(2 / self.layers[i - 1])  # Xavier Initialization
                layer.append({'weights': weights})
            network.append(layer)
        return network

    def relu(self, activation):
        return max(0, activation)

    def relu_derivative(self, output):
        return 1 if output > 0 else 0

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
        patience = 100  # Number of epochs without improvement before stopping
        best_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(self.epochs):
            total_error = 0
            for row in self.dataset:
                outputs = self.forward_propagate(row[:-1])
                expected = [0 for _ in range(self.layers[-1])]
                expected[int(row[-1])] = 1  # Set the correct class to 1
                total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
                self.backward_propagate_error(expected)
                self.update_weights(row)

            # Check for improvement
            if total_error < best_loss:
                best_loss = total_error
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping check
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
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

    # === Data Scaling (Normalization) ===
    # Apply scaling methods such as MinMax or Standardization if necessary here.

    target_column = [row[-1] for row in train_set]
    num_classes = len(set(target_column))
    print(f"Number of unique classes in target: {num_classes}")

    # === Train Neural Network ===
    nn = SimpleNeuralNetwork(layers=[24, 20, 2], data=train_set, learning_rate=0.01, epochs=1000,
                             lambda_param=0.01)
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
