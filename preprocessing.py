import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def process(self):
        """Execute the complete preprocessing pipeline"""
        self.load_file()
        self.clean_data()
        self.encode_data()
        self.normalize_data()
        self.split_data()
        return self

    def load_file(self):
        """Load and sample the dataset"""
        try:
            self.dataset = pd.read_csv(self.input_file)
            # print(f"Original dataset shape: {self.dataset.shape}")

            if self.sample_percentage < 100:
                self.dataset = self.dataset.sample(
                    frac=self.sample_percentage / 100,
                    random_state=42
                )
                # print(f"Sampled dataset shape: {self.dataset.shape}")

        except Exception as e:
            print(f"Error loading file: {e}")
            raise

    def clean_data(self):
        """Clean the dataset by handling missing values and dropping columns"""
        # Drop 'id' column if exists
        if 'id' in self.dataset.columns:
            self.dataset.drop(columns=['id'], inplace=True)
            # print("Dropped 'id' column")

        # Handle missing values
        initial_count = len(self.dataset)
        self.dataset.dropna(inplace=True)
        final_count = len(self.dataset)
        # print(f"Removed {initial_count - final_count} rows with missing values")

    def encode_data(self):
        categorical_columns = ['rbc','pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        encoder = LabelEncoder()
        for col in categorical_columns:
            self.dataset[col] = encoder.fit_transform(self.dataset[col])

        # print("Encoded dataset:", self.dataset.head())

        # Encode target column
        if 'classification' in self.dataset.columns:
            self.dataset['classification'] = self.label_encoder.fit_transform(self.dataset['classification'])
            # print("Encoded target column 'classification'")

    def normalize_data(self):
        """Normalize numerical columns"""
        numerical_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        numerical_cols = [col for col in numerical_cols if col in self.dataset.columns]

        if numerical_cols:
            self.dataset[numerical_cols] = self.scaler.fit_transform(self.dataset[numerical_cols])

    def split_data(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

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
        """Return processed training and testing sets as numpy arrays"""
        if self.training_set is None or self.testing_set is None:
            raise ValueError("Data not processed yet. Call process() first.")
        return self.training_set, self.testing_set

def main():
    """Example usage of the DataPreprocessor class"""
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
    train_df = pd.DataFrame(train_set, columns=processor.dataset.drop('classification', axis=1).columns.tolist() + ['classification'])
    test_df = pd.DataFrame(test_set, columns=processor.dataset.drop('classification', axis=1).columns.tolist() + ['classification'])

    # Preview results
    print("\n--- Training Set Preview ---")
    print(train_df.head())
    print(f"\nTraining set size: {train_df.shape[0]} rows")

    print("\n--- Testing Set Preview ---")
    print(test_df.head())
    print(f"\nTesting set size: {test_df.shape[0]} rows")


if __name__ == "__main__":
    main()
