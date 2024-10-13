import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

# Load the diamonds dataset and keep only numeric columns
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["int32", "int64", "float32", "float64"]).dropna()

# Set display options for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

def threshold(dataframe, column, q1=0.25, q3=0.75):
    """Calculate the lower and upper thresholds for outlier detection using IQR."""
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    down_limit = quartile1 - 1.5 * iqr
    return down_limit, up_limit

def check_outlier(dataframe, col_name):
    """Check if there are any outliers in a given column of the dataframe."""
    down, up = threshold(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < down)].any(axis=None)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Identify categorical and numeric columns in the dataframe."""
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = list(set(cat_cols) - set(cat_but_car)) + num_but_cat

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "object" and col not in num_but_cat]

    # Print observations about the dataframe
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numeric columns: {len(num_cols)}")
    print(f"Categorical but cardinal: {len(cat_but_car)}")
    print(f"Numeric but categorical: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def grab_outliers(dataframe, col_name, q1=0.25, q3=0.75, index=False):
    """Identify outliers in a specified column of a DataFrame."""
    down, up = threshold(dataframe, col_name, q1, q3)
    outliers = pd.DataFrame()

    # Identify upper and lower outliers
    upper_outliers = dataframe[dataframe[col_name] > up][col_name]
    lower_outliers = dataframe[dataframe[col_name] < down][col_name]

    # Create a DataFrame for outliers
    outliers = pd.concat([
        pd.DataFrame({col_name: upper_outliers, "How": "up"}),
        pd.DataFrame({col_name: lower_outliers, "How": "down"})
    ])

    if index:
        return outliers, dataframe[dataframe.index.isin(outliers.index)], outliers.index, down, up

    return outliers, dataframe[dataframe.index.isin(outliers.index)], down, up

def no_outlier_df(dataframe, col_name):
    """Return a DataFrame without outliers in a specified column."""
    down, up = threshold(dataframe, col_name)
    return dataframe[~((dataframe[col_name] < down) | (dataframe[col_name] > up))]

def replace_with_thresholds(dataframe, col_name):
    """Replace outliers with upper/lower threshold values."""
    df_copy = dataframe.copy()
    down, up = threshold(df_copy, col_name)
    df_copy.loc[df_copy[col_name] < down, col_name] = down
    df_copy.loc[df_copy[col_name] > up, col_name] = up
    return df_copy

# Display the initial data and its info
print(df.head())
print("\n###################################\n")
print(df.info())

# Check for outliers in each column
print("\n################## Check Outliers #################\n")
for col in df.columns:
    print(f"{col}: {check_outlier(df, col)}")

# Grab outliers using different quantile ranges
for q1, q3 in [(0.25, 0.75), (0.05, 0.95)]:
    print(f"\n################## Grab Outliers for q1={q1} & q3={q3} #################\n")
    for col in df.columns:
        results = grab_outliers(df, col, q1=q1, q3=q3, index=True)
        outlier_ratio = 100 * len(results[2]) / len(df[col])
        print(f"{col.upper()} Outlier Ratio: {outlier_ratio:.2f}%")
        print(f"How many outliers for {col}: {results[2].shape[0]}\n")

# Apply Local Outlier Factor for outlier detection
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_


# Worst 5 scores (most negative)
print("Worst 5 LOF scores:", np.sort(df_scores)[0:5])

# Visualize the outlier scores
sorted_scores = pd.DataFrame(np.sort(df_scores))
sorted_scores.plot(stacked=True, xlim=[0, 20], style=".-")
plt.show()

# Identify outliers using LOF scores
th = np.sort(df_scores)[3]
df_outliers = df[df_scores < th]
print("Identified Outliers:\n", df_outliers)

# Create a DataFrame without outliers
df_without_outliers = df.drop(df_outliers.index)

# Compare original and non-outlier data lengths
print(f"\n# Observations for all data: {len(df)}")
print(f"# Observations for Non-Outlier data: {len(df_without_outliers)}")

# Display descriptive statistics for comparison
print("\nDescriptive Statistics:\n", df.describe())
