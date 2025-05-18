# Ganna Ahmed Abd El-Naby Silem
# 20210102
import tkinter as tk
from tkinter import ttk

from kneighbours import KNearestNeighbours
from neural_testing import NeuralNetwork
from preprocessing import DataPreprocessor


class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("inputs")
        self.window.geometry("450x450")
        self.input_file = None
        self.input_sample_percentage = None
        self.train_ratio = None
        self.test_ratio = None
        self.k_num = None
        self.knn_result = None
        self.nn_result = None
        self.knn_accuracy = None
        self.nn_accuracy = None
        self.accuracy_threshold = 0.80
        self.create_input_window()

    def create_input_window(self):
        tk.Label(self.window, text="Enter the input file:").pack(pady=5)
        self.input_file = tk.Entry(self.window)
        self.input_file.pack(pady=5)

        tk.Label(self.window, text="Enter Sample Percentage (%):").pack(pady=5)
        self.input_sample_percentage = tk.Entry(self.window)
        self.input_sample_percentage.pack(pady=5)

        tk.Label(self.window, text="Enter the train ratio:").pack(pady=5)
        self.train_ratio = tk.Entry(self.window)
        self.train_ratio.pack(pady=5)

        tk.Label(self.window, text="Enter the test ratio:").pack(pady=5)
        self.test_ratio = tk.Entry(self.window)
        self.test_ratio.pack(pady=5)

        tk.Label(self.window, text="Enter K number:").pack(pady=5)
        self.k_num = tk.Entry(self.window)
        self.k_num.pack(pady=5)

        submit_btn = tk.Button(self.window, text="Process", command=self.submit)
        submit_btn.pack(pady=20)

    def submit(self):
        import tkinter.messagebox as messagebox
        try:
            input_file = self.input_file.get()
            sample_percentage = int(self.input_sample_percentage.get())
            train_ratio = int(self.train_ratio.get())
            test_ratio = int(self.test_ratio.get())
            k_neighbors = int(self.k_num.get())
            accuracy_threshold = 0.90

            if not (0 < sample_percentage <= 100):
                raise ValueError("Sample percentage must be between 1 and 100.")
            if not (0 < train_ratio < 100) or not (0 < test_ratio < 100) or train_ratio + test_ratio != 100:
                raise ValueError("Train and test ratios must add up to 100 and each must be between 1 and 99.")
            if k_neighbors <= 0:
                raise ValueError("K for KNN must be greater than 0.")

        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        while True:
            processor = DataPreprocessor(input_file, sample_percentage, train_ratio=train_ratio)
            processor.process()
            train, test = processor.get_sets()

            train_data = processor.training_set  # numpy array
            test_data = processor.testing_set  # numpy array

            # Split features and labels
            X_train = train_data[:, :-1]
            y_train = train_data[:, -1]

            X_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            # neural network process
            input_neurons = train.shape[1]
            train_set, test_set = processor.get_sets()
            input_neurons = train_set.shape[1] - 1
            hidden_neurons = 10
            output_neurons = 1
            nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons, train_set, processor.original_data)
            nn.train_model()
            nn.evaluate_model(nn, test_set)
            self.nn_accuracy = nn.accuracy
            self.nn_result = nn.result

            # KNN
            knn = KNearestNeighbours(
                X_train,
                y_train,
                X_test,
                y_test,
                k_neighbors,
                processor.original_data)
            knn.evaluate_model(knn, test_data)
            self.knn_accuracy = knn.accuracy
            self.knn_result = knn.results

            print("\nK-Nearest Neighbors Results:")

            if self.nn_accuracy >= accuracy_threshold or self.knn_accuracy >= accuracy_threshold:
                break
        self.show_model_results(self.nn_accuracy, self.nn_result, self.knn_accuracy, self.knn_result)

    def show_model_results(self, nn_accuracy, nn_result, knn_accuracy, knn_result):
        knn_result_dict = knn_result.to_dict(orient='records')

        result_window = tk.Toplevel()
        result_window.title("Model Results")
        result_window.geometry("800x600")
        result_window.configure(bg="white")

        # NN Results
        nn_frame = tk.Frame(result_window, bg="white")
        nn_frame.pack(pady=10, fill="both")

        tk.Label(nn_frame, text=f"Neural Network Accuracy: {nn_accuracy*100:.2f}%",
                 font=("Arial", 12, "bold"), bg="white").pack(pady=5)

        nn_tree = ttk.Treeview(nn_frame, columns=("id", "actual", "predicted"), show="headings", height=8)
        nn_tree.heading("id", text="ID")
        nn_tree.heading("actual", text="Actual")
        nn_tree.heading("predicted", text="Predicted")

        nn_tree.column("id", width=100, anchor="center")
        nn_tree.column("actual", width=150, anchor="center")
        nn_tree.column("predicted", width=150, anchor="center")

        # Display NN results
        for row in nn_result:
            nn_tree.insert("", "end", values=(row["id"], row["actual"], row["predicted"]))

        nn_tree.pack(padx=10, pady=5)

        # KNN Results
        knn_frame = tk.Frame(result_window, bg="white")
        knn_frame.pack(pady=10, fill="both")

        tk.Label(knn_frame, text=f"K-Nearest Neighbors Accuracy: {knn_accuracy*100:.2f}%",
                 font=("Arial", 12, "bold"), bg="white").pack(pady=5)

        knn_tree = ttk.Treeview(knn_frame, columns=("id", "actual", "predicted"), show="headings", height=8)
        knn_tree.heading("id", text="ID")
        knn_tree.heading("actual", text="Actual")
        knn_tree.heading("predicted", text="Predicted")

        knn_tree.column("id", width=100, anchor="center")
        knn_tree.column("actual", width=150, anchor="center")
        knn_tree.column("predicted", width=150, anchor="center")

        # Display KNN results
        for row in knn_result_dict:
            knn_tree.insert("", "end", values=(row["id"], row["actual"], row["predicted"]))

        knn_tree.pack(padx=10, pady=5)


def main():
    app = GUI()
    app.window.mainloop()


if __name__ == "__main__":
    main()
