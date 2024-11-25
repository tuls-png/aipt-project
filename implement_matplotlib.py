
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import itertools

class DataProcessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.num_cols = None
        self.cat_cols = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object']).columns
        return self.df

    def describe_data(self, output_file):
        with open(output_file, "w") as file:
            file.write("Dataset Info:\n")
            self.df.info(buf=file)
            file.write("\n\nDescription:\n")
            file.write(str(self.df.describe(include='all').T))

    def normalize_data(self):
        self.df[self.num_cols] = (self.df[self.num_cols] - self.df[self.num_cols].min()) / (self.df[self.num_cols].max() - self.df[self.num_cols].min())
        print("Normalization complete.")

class Visualizer:
    def __init__(self, data_processor):
        self.df = data_processor.df
        self.output_dir = f"output_plots_{file_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_and_show_plot(self, plot_func, title, filename, *args, **kwargs):
        plt.figure()
        plot_func(*args, **kwargs)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.show()
        plt.close()


    def plot_distributions(self):
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.save_and_show_plot(
                    self.df[col].hist,
                    title=f"Distribution of {col}",
                    filename=f"{col}_distribution.png",
                    bins=10, color="#87CEEB", edgecolor="black"
                )
            else:
                self.save_and_show_plot(
                    self.df[col].value_counts().plot,
                    title=f"Bar Chart of {col}",
                    filename=f"{col}_bar.png",
                    kind="bar", color="#FFA07A", edgecolor="black"
                )


    def correlation_heatmap(self):
        # Select numeric columns only
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numeric_columns].corr()

        self.save_and_show_plot(
            plot_func=lambda: sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5),
            title="Correlation Heatmap",
            filename="correlation_heatmap.png"
        )


    def scatter_plots(self):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col1, col2 in itertools.combinations(numeric_columns, 2):
            self.save_and_show_plot(
                plot_func=lambda: sns.scatterplot(x=self.df[col1], y=self.df[col2], color="skyblue", alpha=0.7),
                title=f"Scatter Plot: {col1} vs {col2}",
                filename=f"scatter_{col1}_vs_{col2}.png"
            )


    def pie_chart(self, column):
        self.save_and_show_plot(
            plot_func=lambda: self.df[column].value_counts().plot(kind="pie", autopct='%1.1f%%', startangle=90),
            title=f"Pie Chart of {column}",
            filename=f"{column}_pie_chart.png"
        )


file_name = 'customer_demographics_purchase'
file_path = rf"C:\Users\tulik\Desktop\IGDTUW\AI PT\aipt-project\{file_name}.csv"

processor = DataProcessor(file_path, "purchase amount")

processor.load_data()
processor.describe_data(f"{file_name}_summary.txt")
processor.normalize_data()

visualizer = Visualizer(processor)
visualizer.plot_distributions()
visualizer.correlation_heatmap()
visualizer.scatter_plots()
visualizer.pie_chart(processor.target_column)
