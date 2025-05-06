import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, input_file, sample_percentage, train_ratio=0.75):
        self.input_file = input_file
        self.sample_percentage = sample_percentage
        self.train_ratio = train_ratio
        self.training_set = None
        self.testing_set = None
        self.class_column = None
        self.dataset = None

        self.load_file()
        self.split_data()

    def load_file(self):
        df = pd.read_csv(self.input_file)

        print("Original number of rows:", df.shape[0])
        df.dropna(how='all', inplace=True)
        print("Number of rows after dropping empty rows:", df.shape[0])

        df = df.sample(frac=self.sample_percentage / 100, random_state=42)

        self.class_column = df.columns[-1]
        self.dataset = df

    def split_data(self):
        X = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_ratio, random_state=42
        )

        self.training_set = pd.concat([X_train, y_train], axis=1)
        self.testing_set = pd.concat([X_test, y_test], axis=1)

    def get_sets(self):
        return self.training_set, self.testing_set


def main():
    # === User Inputs ===
    input_file = input("Enter the path to your CSV file: ").strip()
    sample_percentage = float(input("Enter sample percentage (e.g. 100 for all): "))
    train_percent = float(input("Enter training percentage (e.g. 75 for 75% training): "))

    # Convert training percent to ratio
    train_ratio = train_percent / 100.0

    # === Process the Data ===
    processor = DataPreprocessor(input_file, sample_percentage, train_ratio)
    train_set, test_set = processor.get_sets()

    # === Output Results ===
    print("\n--- Training Set Preview ---")
    print(train_set.head())
    print(f"Training set size: {train_set.shape[0]} rows")

    print("\n--- Testing Set Preview ---")
    print(test_set.head())
    print(f"Testing set size: {test_set.shape[0]} rows")


# Run the main function
if __name__ == "__main__":
    main()
