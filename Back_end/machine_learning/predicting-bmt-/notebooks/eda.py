import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# For treemap (optional, install via pip install squarify if needed)
try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False

# Set up the "plots" folder relative to this script's location (machine_learning folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def save_plot(filename: str) -> None:
    """
    Save the current matplotlib figure to the plots folder,
    display it (blocking) until the user closes it, and then close the figure.
    """
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Plot saved to: {filepath}")
    if plt.get_backend().lower() not in ["agg", "pdf", "svg", "ps"]:
        plt.show()
    plt.close()

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load the BMT dataset from a CSV file into a pandas DataFrame using a path relative to the script's location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading CSV file from:", full_path)
    df = pd.read_csv(full_path)
    return df

def basic_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the DataFrame.
    """
    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== DESCRIBE ===")
    print(df.describe(include='all'))

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

def plot_bar_chart(df: pd.DataFrame) -> None:
    """
    Create a bar plot for 'Recipientgender'.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Recipientgender')
    plt.title('Count of Recipientgender')
    save_plot("bar_chart_recipientgender.png")

def plot_scatter(df: pd.DataFrame) -> None:
    """
    Create a scatter plot for 'Donorage' vs. 'CD34kgx10d6', colored by 'Recipientgender'.
    """
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='Donorage', y='CD34kgx10d6', hue='Recipientgender')
    plt.title('Scatter: Donorage vs. CD34+ Cell Dose')
    save_plot("scatter_donorage_cd34.png")

def plot_boxplot(df: pd.DataFrame) -> None:
    """
    Create a box plot for 'Donorage' by 'Stemcellsource'.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Stemcellsource', y='Donorage', orientation='vertical')

    plt.title('Boxplot of Donorage by Stemcellsource')
    save_plot("boxplot_donorage_by_stemcellsource.png")

def plot_histogram(df: pd.DataFrame) -> None:
    """
    Create a histogram of 'Donorage'.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='Donorage', bins=30, kde=True)
    plt.title('Histogram of Donor Age')
    save_plot("histogram_donorage.png")

def plot_area(df: pd.DataFrame) -> None:
    """
    Create an area plot for 'Donorage'.
    """
    df_sorted = df.sort_values(by='Donorage')
    plt.figure(figsize=(6, 4))
    plt.fill_between(df_sorted.index, df_sorted['Donorage'], alpha=0.5)
    plt.title('Area Plot of Donorage (Example)')
    save_plot("area_plot_donorage.png")

def plot_pie_chart(df: pd.DataFrame) -> None:
    """
    Create a pie chart of 'Disease' distribution.
    """
    disease_counts = df['Disease'].value_counts()
    labels = disease_counts.index
    sizes = disease_counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Disease Types')
    plt.axis('equal')
    save_plot("pie_chart_disease.png")

def plot_treemap(df: pd.DataFrame) -> None:
    """
    Create a treemap of 'Disease' distribution.
    """
    if not HAS_SQUARIFY:
        print("squarify is not installed. Skipping treemap plot.")
        return
    disease_counts = df['Disease'].value_counts()
    labels = disease_counts.index
    sizes = disease_counts.values

    plt.figure(figsize=(8, 6))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    squarify.plot(sizes=sizes, label=labels, alpha=0.8,
                   color=cmap(np.linspace(0, 1, len(sizes))))
    plt.title('Treemap of Disease Distribution')
    plt.axis('off')
    save_plot("treemap_disease.png")

def plot_missing_values_heatmap(df: pd.DataFrame) -> None:
    """
    Create a heatmap of missing values.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    save_plot("missing_values_heatmap.png")

def plot_missing_percentage(df: pd.DataFrame) -> None:
    """
    Plot the percentage of missing values for each attribute.
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print("Percentage of Missing Values per Attribute:")
    print(missing_percent)
    plt.figure(figsize=(10, 6))
    missing_percent.sort_values(ascending=False).plot(kind='bar')
    plt.ylabel("Percentage")
    plt.title("Missing Value Percentage per Attribute")
    save_plot("missing_percentage.png")

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Create a larger correlation matrix heatmap for numeric attributes,
    with only two decimal places.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f",
                annot_kws={"size": 8})
    plt.title("Correlation Matrix for Numeric Attributes", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot("correlation_matrix.png")

# --- Additional Useful Plots ---

def plot_violin(df: pd.DataFrame) -> None:
    """
    Create a violin plot for 'Donorage' by 'Recipientgender'.
    """
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x='Recipientgender', y='Donorage')
    plt.title('Violin Plot of Donor Age by Recipient Gender')
    save_plot("violin_donorage_recipientgender.png")

def plot_jointplot(df: pd.DataFrame) -> None:
    """
    Create a joint plot for 'Donorage' vs. 'CD34kgx10d6' with a regression line.
    """
    jp = sns.jointplot(data=df, x='Donorage', y='CD34kgx10d6', kind='reg', height=6)
    jp.fig.suptitle('Joint Plot: Donor Age vs. CD34+ Cell Dose')
    jp.fig.subplots_adjust(top=0.93)
    jointplot_filepath = os.path.join(plots_dir, "jointplot_donorage_cd34.png")
    jp.fig.savefig(jointplot_filepath, bbox_inches="tight")
    print(f"Plot saved to: {jointplot_filepath}")
    plt.show()  # Blocks until the jointplot window is closed
    plt.close(jp.fig)

def plot_count_stemcellsource(df: pd.DataFrame) -> None:
    """
    Create a count plot for 'Stemcellsource'.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Stemcellsource')
    plt.title('Count of Stem Cell Source')
    save_plot("count_stemcellsource.png")

# --- New Additional Useful Plots ---

def plot_boxplot_cd34_by_recipientgender(df: pd.DataFrame) -> None:
    """
    Create a box plot for 'CD34kgx10d6' by 'Recipientgender'.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Recipientgender', y='CD34kgx10d6', orientation='vertical')
    plt.title('Boxplot of CD34+ Cell Dose by Recipient Gender')
    save_plot("boxplot_cd34_by_recipientgender.png")

def plot_bar_mean_cd34_by_stemcellsource(df: pd.DataFrame) -> None:
    """
    Create a bar plot for the mean 'CD34kgx10d6' by 'Stemcellsource'.
    """
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='Stemcellsource', y='CD34kgx10d6', ci='sd')
    plt.title('Mean CD34+ Cell Dose by Stem Cell Source')
    save_plot("bar_mean_cd34_by_stemcellsource.png")

def plot_swarm_donorage_by_disease(df: pd.DataFrame) -> None:
    """
    Create a swarm plot for 'Donorage' by 'Disease'.
    """
    plt.figure(figsize=(8, 4))
    sns.swarmplot(data=df, x='Disease', y='Donorage')
    plt.title('Swarm Plot of Donor Age by Disease')
    plt.xticks(rotation=45, ha="right")
    save_plot("swarm_donorage_by_disease.png")

def main():
    csv_path = os.path.join("..", "data", "processed", "bmt_dataset.csv")
    df = load_data(csv_path)
    basic_info(df)
    
    # Existing plots
    plot_missing_values_heatmap(df)
    plot_missing_percentage(df)
    plot_correlation_matrix(df)
    plot_bar_chart(df)
    plot_scatter(df)
    plot_boxplot(df)
    plot_histogram(df)
    plot_area(df)
    plot_pie_chart(df)
    plot_treemap(df)
    
    # Additional useful plots
    plot_violin(df)
    plot_jointplot(df)
    plot_count_stemcellsource(df)
    
    # New additional useful plots
    plot_boxplot_cd34_by_recipientgender(df)
    plot_bar_mean_cd34_by_stemcellsource(df)
    plot_swarm_donorage_by_disease(df)

if __name__ == "__main__":
    main()
