import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, input_file, sample_percentage=100, train_ratio=0.75):
        self.input_file = input_file
        self.sample_percentage = sample_percentage
        self.train_ratio = train_ratio
        self.training_set = None
        self.testing_set = None
        self.dataset = None
        self.original_data = None

    def process(self):
        self.load_file()
        self.clean_data()
        self.encode_data()
        self.normalize_data()
        self.split_data()
        return self

    def load_file(self):
        self.dataset = pd.read_csv(self.input_file)
        self.dataset = self.dataset.sample(frac=self.sample_percentage / 100, )

    def drop_id(self):
        if 'id' in self.dataset.columns:
            self.dataset.drop(columns=['id'], inplace=True)

    def clean_data(self):
        # Handle missing values
        initial_count = len(self.dataset)
        self.dataset.dropna(inplace=True)
        final_count = len(self.dataset)
        print(f"Removed {initial_count - final_count} rows with missing values")

    def encode_data(self):
        categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        encoder = LabelEncoder()
        for col in categorical_columns:
            self.dataset[col] = encoder.fit_transform(self.dataset[col])

    def normalize_data(self):
        numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        scaler = MinMaxScaler()
        self.dataset[numerical_cols] = scaler.fit_transform(self.dataset[numerical_cols])

    def split_data(self):
        self.dataset = self.dataset.sample(frac=1)
        self.original_data = self.dataset.copy()
        self.drop_id()

        X = self.dataset.drop('classification', axis=1)
        y = self.dataset['classification']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=self.train_ratio,
            stratify=y
        )

        self.training_set = pd.concat([X_train, y_train], axis=1).values
        self.testing_set = pd.concat([X_test, y_test], axis=1).values

    def get_sets(self):
        return self.training_set, self.testing_set


def main():
    sample_percentage = int(input("Enter sample percentage (e.g., 100 for all): ") or 100)
    train_percent = int(input("Enter training percentage (e.g., 75 for 75% training): ") or 75)

    # Create and run preprocessor
    processor = DataPreprocessor(
        input_file="Kidney_Disease_data_for_Classification_V2.csv",
        sample_percentage=sample_percentage,
        train_ratio=train_percent / 100
    ).process()

    # Get processed data
    train_set, test_set = processor.get_sets()

    # Convert numpy arrays to DataFrame for preview
    train_df = pd.DataFrame(train_set, columns=processor.dataset.drop('classification', axis=1).columns.tolist() + [
        'classification'])
    test_df = pd.DataFrame(test_set, columns=processor.dataset.drop('classification', axis=1).columns.tolist() + [
        'classification'])

    print("\n--- Original Dataset Preview (Before Processing) ---")
    print(processor.original_data.head())
    # Preview results
    print("\n--- Training Set Preview ---")
    print(train_df.head())
    print(f"\nTraining set size: {train_df.shape[0]} rows")
    # Print original data before encoding/normalization

    print("\n--- Testing Set Preview ---")
    print(test_df.head())
    print(f"\nTesting set size: {test_df.shape[0]} rows")


if __name__ == "__main__":
    main()
