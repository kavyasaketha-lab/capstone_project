#Imports#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

sns.set_theme(style="whitegrid")
sns.load_dataset("titanic")

#Load Titanic exactly once
df = sns.load_dataset("titanic")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

df.head()
# Immediately save CSV
df.to_csv("titanic.csv", index=False)

print("titanic.csv saved successfully.")
# Profile the dataset
print("========== DATASET INFO ==========")
df.info()

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DESCRIPTIVE STATISTICS ==========")
display(df.describe(include="all"))

# Missing-value percentages
missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

print("Missing-value report:")
display(missing_report)
#Appplying Missing-value strategy
print("Missing-value strategy")
print("-" * 60)

for column in missing_report.index:
    pct = missing_percentage[column]

    if pct < 5:
        strategy = "Drop rows because missingness is below 5%."
    elif pct <= 30:
        strategy = "Impute because missingness is between 5% and 30%."
    else:
        strategy = "Drop column because missingness is above 30%."

    print(f"{column}: {pct:.2f}% missing -> {strategy}")
    
# Clean the dataset
df_clean = df.copy()

# Drop high-missingness column
if "deck" in df_clean.columns:
    df_clean.drop(columns=["deck"], inplace=True)

# Impute age using median
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

# Impute embarked using mode
df_clean["embarked"] = df_clean["embarked"].fillna(
    df_clean["embarked"].mode()[0]
)

# Handle embark_town consistently
if "embark_town" in df_clean.columns:
    df_clean["embark_town"] = df_clean["embark_town"].fillna(
        df_clean["embark_town"].mode()[0]
    )

print("Remaining missing values:")
display(df_clean.isnull().sum())
# Save the final cleaned CSV
df_clean.to_csv("titanic.csv", index=False)

print("Final cleaned titanic.csv saved.")
print("Shape:", df_clean.shape)
pd.read_csv("titanic.csv")
# Age and Fare analysis: Histograms

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_clean["age"], kde=True, ax=axes[0])
axes[0].set_title("Age Distribution")

sns.histplot(df_clean["fare"], kde=True, ax=axes[1])
axes[1].set_title("Fare Distribution")

plt.tight_layout()
plt.show()

# Box plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(y=df_clean["age"], ax=axes[0])
axes[0].set_title("Age Box Plot")

sns.boxplot(y=df_clean["fare"], ax=axes[1])
axes[1].set_title("Fare Box Plot")

plt.tight_layout()
plt.show()
# IQR outliers
def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]
    
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    }


age_outliers = iqr_outlier_count(df_clean["age"])
fare_outliers = iqr_outlier_count(df_clean["fare"])

print("AGE OUTLIER REPORT")
print(age_outliers)

print("\nFARE OUTLIER REPORT")
print(fare_outliers)
# Fare mean, median and mode
fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]

print(f"Fare Mean   : {fare_mean:.4f}")
print(f"Fare Median : {fare_median:.4f}")
print(f"Fare Mode   : {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    skew_conclusion = "Fare is right-skewed."
elif fare_mean < fare_median < fare_mode:
    skew_conclusion = "Fare is left-skewed."
else:
    skew_conclusion = "Fare does not show the classic mean-median-mode ordering of strong skewness."

print(skew_conclusion)

# Survival analysis

survival_by_sex = (
    df_clean.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print("Survival rate by sex:")
print(survival_by_sex)

# Survival by passenger class
survival_by_class = (
    df_clean.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print("Survival rate by passenger class:")
print(survival_by_class)
# : Sex + passenger class
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")

plt.show()

# Age vs Fare
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title("Age, Fare and Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()

# Standardization sanity check

scaler = StandardScaler()

df_standardized = df_clean.copy()

df_standardized[["age_z", "fare_z"]] = scaler.fit_transform(
    df_clean[["age", "fare"]]
)

print("BEFORE STANDARDIZATION")
print(
    df_clean[["age", "fare"]].agg(["mean", "std"]).round(4)
)

print("\nAFTER STANDARDIZATION")
print(
    df_standardized[["age_z", "fare_z"]].agg(["mean", "std"]).round(4)
)