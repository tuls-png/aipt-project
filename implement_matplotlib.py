import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Data Cleaning
    print("\n--- Data Cleaning ---")
    print("Initial data overview:")
    print(df.info())
    
    # Handling missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\nMissing values handled.")
    
    # Removing duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    
    # Data Preprocessing
    print("\n--- Data Preprocessing ---")
    print("Descriptive statistics:")
    print(df.describe())
    
    # Encoding categorical variables if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"\nCategorical columns {list(categorical_cols)} encoded.")
    else:
        print("No categorical columns found.")
    
    # Normalizing numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    print("Numeric columns normalized.")
    
    # Data Analysis and Visualization
    print("\n--- Data Analysis and Visualization ---")
    
    # Plotting correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    
    # Histogram of all numeric columns
    df[numeric_cols].hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
    plt.suptitle("Histogram of Numeric Features")
    plt.show()
    
    # Scatter plots for top correlations
    correlations = df.corr().unstack().sort_values(ascending=False)
    strong_pairs = correlations[(correlations > 0.5) & (correlations < 1)]
    
    if len(strong_pairs) > 0:
        for (col1, col2), corr_value in strong_pairs.items():
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=col1, y=col2, alpha=0.7)
            plt.title(f"Scatter Plot: {col1} vs {col2} (Correlation = {corr_value:.2f})")
            plt.show()
    else:
        print("No strong correlations found for scatter plots.")
    
    print("Analysis complete!")

# Example usage
file_path = r'C:\Users\tulik\Desktop\IGDTUW\AI PT\aipt_lab.csv' # Replace with your dataset's file path
analyze_data(file_path)
