import numpy as np
from preprocessing import DataPreprocessor
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, input_values, original_data):
        self.learning_rate = 0.01
        self.epochs = 300
        self.n_inputs = input_neurons
        self.n_hidden = hidden_neurons
        self.n_outputs = output_neurons
        self.dataset = input_values
        self.original_data = original_data
        self.accuracy = None
        self.result = []
        self.weights = []
        self.biases = []
        self.construct_network()

    def sigmoid_func(self, net_value):
        return 1 / (1 + np.exp(-net_value))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def construct_network(self):
        self.weights = [
            np.random.rand(self.n_hidden, self.n_inputs) * 0.1,
            np.random.rand(self.n_outputs, self.n_hidden) * 0.1,
        ]
        self.biases = [
            np.zeros((self.n_hidden, 1)),
            np.zeros((self.n_outputs, 1))
        ]

    def forward_propagation(self, features):
        features = np.array(features, dtype=float).reshape(-1, 1)

        net_hidden = np.dot(self.weights[0], features) + self.biases[0]
        out_hidden = self.sigmoid_func(net_hidden)

        net_output = np.dot(self.weights[1], out_hidden) + self.biases[1]
        out_output = self.sigmoid_func(net_output)

        return net_hidden, out_hidden, net_output, out_output

    def train_model(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for row in self.dataset:
                features_values = np.array(row[:-1], dtype=float).reshape(-1, 1)
                target_value = np.array([[float(row[-1])]])

                _, out_hidden, _, out_output = self.forward_propagation(features_values)

                # Calculate output layer error
                output_error = (target_value - out_output) * out_output * (1 - out_output)
                # Calculate hidden layer error
                hidden_error = np.dot(self.weights[1].T, output_error) * out_hidden * (1 - out_hidden)

                # Update output layer weights and biases
                # (output,1)(1,hidden)
                self.weights[1] += self.learning_rate * np.dot(output_error, out_hidden.T)
                self.biases[1] += self.learning_rate * output_error

                # Update hidden layer weights and biases
                self.weights[0] += self.learning_rate * np.dot(hidden_error, features_values.T)
                self.biases[0] += self.learning_rate * hidden_error

                total_loss += np.mean(np.abs(target_value - out_output))

        print("Training complete.")

    def predict(self, row):
        features = np.array(row, dtype=float).reshape(-1, 1)
        _, _, _, out_output = self.forward_propagation(features)
        prob = out_output.item()
        return 1 if prob >= 0.5 else 0

    def evaluate_model(self, model, test_set):
        correct = 0
        test_size = len(test_set)

        print("\n--- Predictions on Test Set ---")

        for i, row in enumerate(test_set):
            sample_input = row[:-1]
            actual_class = int(row[-1])
            predicted_class = model.predict([sample_input])
            matched_id = None
            sample_input_arr = np.array(sample_input, dtype=float)
            for idx, orig_row in self.original_data.iterrows():
                orig_features = orig_row[1:-1].to_numpy(dtype=float)
                if np.allclose(orig_features, sample_input_arr, atol=1e-6):
                    matched_id = orig_row['id']
                    break

            self.result.append({
                'id': int(matched_id),
                'predicted': predicted_class,
                'actual': actual_class
            })

            input_str = ", ".join(f"{val:.3f}" for val in sample_input)
            pred_str = 'ckd' if predicted_class == 1 else 'notckd'
            actual_str = 'ckd' if actual_class == 1 else 'notckd'

            print(
                f"Sample {i + 1} (ID: {matched_id}): Features = [{input_str}] => Predicted: {pred_str}, Actual: {actual_str}")

            if predicted_class == actual_class:
                correct += 1

        accuracy = correct / test_size if test_size > 0 else 0
        self.accuracy = accuracy
        print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")

        return accuracy, self.result


def main():
    processor = DataPreprocessor("Kidney_Disease_data_for_Classification_V2.csv", 100, 75)
    processor.process()
    train_set, test_set = processor.get_sets()

    input_neurons = train_set.shape[1] - 1
    hidden_neurons = 10
    output_neurons = 1

    nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons, train_set,processor.original_data)
    nn.construct_network()
    nn.train_model()
    nn.evaluate_model(nn,test_set)


if __name__ == "__main__":
    main()
