import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load dataset
def load(path):
    dataframe = pd.read_csv(path)
    return dataframe

# Calculate thresholds for given column
def threshold(dataframe, column, q1=0.25, q3=0.75):
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    down_limit = quartile1 - 1.5 * iqr
    return down_limit, up_limit

# Check does given column has any null cell
def check_outlier(dataframe, col_name):
    down, up = threshold(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < down)].any(axis=None)

# Distinguish column data types
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].nunique() < cat_th) & (dataframe[col].dtypes != "O")]
    cat_but_car = [col for col in dataframe.columns if (dataframe[col].nunique() > car_th) & (dataframe[col].dtypes == "O")]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "object"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car:{len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# Return outliers with respect to given quantile values
def grab_outliers(dataframe, col_name, q1=0.25, q3=0.75, index=False):
    down, up = threshold(dataframe, col_name, q1, q3)
    upper_outliers = dataframe[dataframe[col_name] > up][col_name]
    lower_outliers = dataframe[dataframe[col_name] < down][col_name]

    upper_outliers_df = pd.DataFrame({col_name: upper_outliers, "How": "up"})
    lower_outliers_df = pd.DataFrame({col_name: lower_outliers, "How": "down"})
    df_outliers = pd.concat([upper_outliers_df, lower_outliers_df])

    if index:
        outlier_index = df_outliers.index
        return df_outliers, outlier_index, down, up
    return df_outliers, down, up

# Delete outliers of given column
def noOutlierDf(dataframe, col_name):
    down, up = threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < down) | (dataframe[col_name] > up))]
    return df_without_outliers

# Replace outliers with thresholds
def replace_with_thresholds(dataframe, col_name):
    dataframe_copy = dataframe.copy()
    down, up = threshold(dataframe_copy, col_name)
    dataframe_copy.loc[dataframe_copy[col_name] < down, col_name] = down
    dataframe_copy.loc[dataframe_copy[col_name] > up, col_name] = up
    return dataframe_copy

# Analysis missing values
def missing_values_table(dataframe, na_columns=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (100 * dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_columns:
        return na_columns

# Load Dataset
df = sns.load_dataset("titanic")

print("\n##################### 1 ####################\n")
print(df.isnull().any(axis=0))

print("\n##################### 2 ####################\n")
print(df.isnull().any())

print("\n##################### 3 ####################\n")
print(df.isnull().values.any())

print("\n##################### 4 ####################\n")
print(df.isnull())

print("\n##################### 5 ####################\n")
print(df.isnull().sum())

print("\n##################### 6 ####################\n")
print(df.isnull().sum().sum())

print("\n##################### 7 ####################\n")
print(df.isnull().sum().sort_values(ascending=False))

print("\n##################### 8 ####################\n")
print((100 * df.isnull().sum() / df.shape[0]).sort_values(ascending=False))

print("\n##################### 9 ####################\n")
print(df.isnull().sum().sort_values(ascending=False))

print("\n##################### 10 ####################\n")
na_columns = [col for col in df.columns if df[col].isnull().any()]
print(na_columns)

print("\n##################### 11 ####################\n")
na_columns = missing_values_table(df)
print(na_columns)

##############################################
#     Solution Types for Missing Values
##############################################
dframe = df.copy()

print("\n##################### 1. DELETE ALL NaN ROWS #####################\n")
# Delete all rows containing NaN values.
print(f"Original shape: {dframe.shape}")
print(f"Shape after dropping NaN rows: {dframe.dropna().shape}")

print("\n##################### 2. FILL NaN CELLS MANUALLY #####################\n")
# Fill cells manually
print(dframe["age"].fillna(dframe["age"].mean()))  # Using mean
dframe["age"].fillna(dframe["age"].median())       # Using median
dframe["age"].fillna(0)                            # Using zero

print("\n##################### 3. FILL MISSING VALUES WITH LAMBDA FUNCTION #####################\n")
# Fill missing values using lambda function
filled_lambda = dframe.apply(lambda x: x.fillna(x.mean()) if x.dtype not in ["object", "category"] else x.fillna(x.mode()[0]), axis=0)
print(filled_lambda)

# Separate filled DataFrames for further analysis
dframe_filled_by_mean = dframe.apply(lambda x: x.fillna(x.mean()) if x.dtype not in ["object", "category"] else x, axis=0)
dframe_filled_by_mode = dframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype in ["object", "category"]) & (x.nunique() <= 10) else x, axis=0)

print("\n######## Original DataFrame NaN counts ########")
print(dframe.isnull().sum().sort_values(ascending=False))

print("\n######## DataFrame with missing values filled by mean ########")
print(dframe_filled_by_mean.isnull().sum().sort_values(ascending=False))

print("\n##################### 4. FILLING ACCORDING TO GROUPS #####################\n")
# More efficient way: Filling age with gender-based mean
print(f"Overall mean of age: {df['age'].mean()}")
print(f"Mean age when grouped by sex:\n{df.groupby('sex')['age'].mean()}")
print(f"Using .agg:\n{df.groupby('sex').agg({'age': 'mean'})}")

# Filling NaN in 'age' using group-based mean
df["age"] = df["age"].fillna(df.groupby("sex")["age"].transform("mean"))

# Alternative using .loc
df.loc[(df["age"].isnull()) & (df["sex"] == "female"), "age"] = df.groupby("sex")["age"].mean()["female"]
df.loc[(df["age"].isnull()) & (df["sex"] == "male"), "age"] = df.groupby("sex")["age"].mean()["male"]

print(f"Remaining NaN values in 'age': {df['age'].isnull().sum()}")

print("\n##################### 5. FILL NaN CELLS WITH ML MODEL/PREDICTIONS #####################\n")

# Reload dataset to avoid using filled values in previous steps
df = load("feature_engineering/datasets/titanic.csv")

# Divide columns by type
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Remove unnecessary 'PassengerId' from numeric columns
num_cols.remove("PassengerId")

# Print categorized columns for verification
col_dir = {
    "Categorical": cat_cols,
    "Numeric": num_cols,
    "Categorical but Car": cat_but_car
}

# Print the column groups for inspection
for k, v in col_dir.items():
    print(f"{k}: {v}")

# Apply One Hot Encoding only to categorical columns (excluding cat_but_car)
# The resulting DataFrame contains encoded categorical columns and original numerical columns.
# Setting `drop_first=True` helps to avoid multicollinearity.
df_ohEncoded = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

# Compare the original and encoded DataFrames
print(df.head(5))
print("\n########################################\n")
print(df_ohEncoded.head(5))

# Reason for `drop_first=True`: Avoiding multicollinearity.
# For example, if `sex_male` is False, then `sex` is female.
# Similarly, if both `Embarked` columns are False, then it means `Embarked_C` is True.

# Standardize the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_ohEncoded_scaled = pd.DataFrame(scaler.fit_transform(df_ohEncoded), columns=df_ohEncoded.columns)

# Observe the scaled values
print(df_ohEncoded_scaled.describe())

# Import and use KNN Imputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
# Fill NaN values based on the nearest neighbors
df_filled = pd.DataFrame(imputer.fit_transform(df_ohEncoded_scaled), columns=df_ohEncoded_scaled.columns)
print(df_filled.head())

# Inverse the scaling to interpret the imputed values in their original scale
df_inversed = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df_filled.columns)
print(df_inversed.head())

# Check if there are any NaN cells left
print(f"Are there any NaN cells in the DataFrame? {df_inversed.isnull().values.any()}")

# Track the imputed values by adding a new column to the original DataFrame
df["Age_imputed"] = df_inversed["Age"]
print(df.loc[df["Age"].isnull(), ["Age", "Age_imputed"]])

# Display the first few rows to verify the new column
print(df.head(5))

# Visualize the structure of missing values using `missingno`
import missingno as msno
import matplotlib.pyplot as plt

# Bar chart of non-null variables
# msno.bar(df)
# plt.show()

# Matrix plot to detect patterns in missing data
# msno.matrix(df)
# plt.show()

# Heatmap to observe correlations in missing data
# Scenario 1: Values are missed together, Scenario 2: Negative correlation in missing values.
# msno.heatmap(df)
# plt.show()

# Load the Titanic dataset again to restart with a clean version
df = load("feature_engineering/datasets/titanic.csv")

# Function to display the missing values table
na_cols = missing_values_table(df, True)
print("\nThe columns which have null values:")
print(na_cols)

# Function to analyze the relationship between missing values and the target variable
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    # Create a new column for each na_column, marking nulls as 1 and non-nulls as 0.
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    # Extract the columns that were created to flag null values
    na_flags = [col for col in temp_df.columns if "_NA_FLAG" in col]

    # Analyze how the presence of missing values correlates with the target variable
    for col in na_flags:
        analysis_dict = {
            "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
            "COUNT": temp_df.groupby(col)[target].count()
        }
        print(pd.DataFrame(analysis_dict), end="\n\n\n")

# Analyze the relationship between missing values and the "Survived" target variable
print("\n############### Analysis Result ###################\n")
missing_vs_target(df, "Survived", na_cols)
