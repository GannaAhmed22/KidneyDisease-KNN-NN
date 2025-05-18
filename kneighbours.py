import numpy as np
import pandas as pd
from preprocessing import DataPreprocessor
import math
import operator


class KNearestNeighbours:
    def __init__(self, feature_data, target_data, feature_test_data, target_test_data, num_k, original_data):
        self.features = feature_data
        self.target = target_data
        self.ftest_data = feature_test_data
        self.tartest_data = target_test_data
        self.k = num_k
        self.original_data = original_data
        self.accuracy = None
        self.results = None

    def euclidean_distance(self, point_one, point_two):
        return math.sqrt(np.sum((point_one - point_two) ** 2))

    def predict(self, test_point):
        distances = []
        for i in range(len(self.features)):
            dist = self.euclidean_distance(test_point, self.features[i])
            distances.append((self.target[i], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])

        prediction = max(set(neighbors), key=neighbors.count)
        return prediction

    def evaluate_model(self, model, test_set):
        correct = 0
        test_size = len(test_set)
        self.results = []

        print("\n--- Predictions on Test Set ---")

        for i, row in enumerate(test_set):
            sample_input = row[:-1]
            actual_class = int(row[-1])
            predicted_class = model.predict([sample_input])

            matched_id = None
            sample_input_arr = np.array(sample_input, dtype=int)

            for idx, orig_row in self.original_data.iterrows():
                orig_features = orig_row[1:-1].to_numpy(dtype=int)
                if np.allclose(orig_features, sample_input_arr, atol=1e-6):
                    matched_id = orig_row['id']
                    break

            self.results.append({
                'id': int(matched_id),
                'predicted': predicted_class,
                'actual': actual_class
            })

            input_str = ", ".join(f"{val:.3f}" for val in sample_input)
            pred_str = 'ckd' if predicted_class == 1 else 'notckd'
            actual_str = 'ckd' if actual_class == 1 else 'notckd'

            print(
                f"Sample {i + 1} (ID: {int(matched_id)}): Features = [{input_str}] => Predicted: {pred_str}, Actual: {actual_str}"
            )

            if predicted_class == actual_class:
                correct += 1

        accuracy = (correct / test_size) if test_size > 0 else 0
        self.accuracy = accuracy
        print(accuracy)
        print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")

        # Convert results list to DataFrame
        self.results = pd.DataFrame(self.results)

        return accuracy, self.results


def main():
    input_file = "Kidney_Disease_data_for_Classification_V2.csv"
    sample_percentage = 100
    train_ratio = 75  # percentage
    k = 5  # number of neighbors

    print("🚀 Preprocessing data...")
    processor = DataPreprocessor(input_file, sample_percentage, train_ratio)
    processor.process()

    train_data = processor.training_set  # numpy array
    test_data = processor.testing_set  # numpy array

    # Split features and labels
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    print("\n📊 Data Shapes:")
    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print(f"Testing labels: {y_test.shape}")

    print(f"\n🧠 Creating KNN model with k={k}...")
    knn = KNearestNeighbours(X_train, y_train, X_test, y_test, k, processor.original_data)

    print("\n🔍 Evaluating model...")
    accuracy, results_df = knn.evaluate_model(knn, test_data)

    #
    # results_df.to_csv("knn_predictions.csv", index=False)
    # print("\n💾 Results saved to 'knn_predictions.csv'")


if __name__ == "__main__":
    main()
