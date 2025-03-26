# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")

# Step 2: Explore the dataset
def explore_data(df):
    print("\n📊 Data Overview:")
    print(df.head())
    print("\n🔍 Data Info:")
    print(df.info())
    print("\n📏 Descriptive Statistics:")
    print(df.describe())

# Step 3: Clean the dataset
def clean_data(df):
    print("\n🧹 Cleaning data...")
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    print("✅ Data cleaned!")
    return df

# Step 4: Analyze the dataset
def analyze_data(df):
    print("\n📈 Correlation Matrix:")
    print(df.corr())

    # Example: Find top 5 most frequent values in a column
    print("\n🔢 Most Common Values:")
    print(df['your_column'].value_counts().head(5))

# Step 5: Visualize the dataset
def visualize_data(df):
    print("\n📊 Visualizing Data...")
    sns.pairplot(df, diag_kind='kde')
    plt.title("Feature Relationships")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    file_path = 'your_dataset.csv'
    df = load_data(file_path)

    if df is not None:
        explore_data(df)
        df = clean_data(df)
        analyze_data(df)
        visualize_data(df)
