# ============================================================
# THIRANEX - EXPLORORY DATA ANSIS (EDA) PROJECTALY
# ==================AT==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS PROJECT")
print("=" * 60)

print("\nDataset loaded successfully!")


# ============================================================
# 2. FIRST LOOK AT DATA
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# 3. DATA TYPES AND INFORMATION
# ============================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing)

print("\nMissing Value Percentage:")

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print(missing_percentage)


# ============================================================
# 5. DUPLICATE VALUES
# ============================================================

print("\n========== DUPLICATES ==========")

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())


print("\n========== COMPLETE STATISTICAL SUMMARY ==========")

print(
    df.describe(include="all")
)


# ============================================================
# 7. NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\n========== NUMERICAL COLUMNS ==========")
print(numeric_columns.tolist())

print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_columns.tolist())


# ============================================================
# 8. UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:

    print(
        column,
        "->",
        df[column].nunique(),
        "unique values"
    )


# ============================================================
# 9. VALUE COUNTS FOR CATEGORICAL DATA
# ============================================================

print("\n========== CATEGORICAL VALUE COUNTS ==========")

for column in categorical_columns:

    if df[column].nunique() <= 20:

        print("\nColumn:", column)

        print(
            df[column].value_counts()
        )


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

if len(numeric_columns) >= 2:

    correlation = df[numeric_columns].corr()

    print("\n========== CORRELATION MATRIX ==========")

    print(correlation)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "correlation_heatmap.png"
    )

    plt.show()


# ============================================================
# 11. HISTOGRAM / DISTRIBUTION ANALYSIS
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(
        "Distribution of " + column
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = (
        "distribution_"
        + str(column)
        + ".png"
    )

    plt.savefig(filename)

    plt.show()


# ============================================================
# 12. BOXPLOT / OUTLIER ANALYSIS
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df[column]
    )

    plt.title(
        "Boxplot of " + column
    )

    plt.xlabel(column)

    plt.tight_layout()

    filename = (
        "boxplot_"
        + str(column)
        + ".png"
    )

    plt.savefig(filename)

    plt.show()


# ============================================================
# 13. CATEGORICAL DATA VISUALIZATION
# ============================================================

for column in categorical_columns:

    if df[column].nunique() <= 15:

        plt.figure(figsize=(9, 5))

        sns.countplot(
            data=df,
            x=column
        )

        plt.title(
            "Count of " + column
        )

        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        filename = (
            "categorical_"
            + str(column)
            + ".png"
        )

        plt.savefig(filename)

        plt.show()


# ============================================================
# 14. SCATTER PLOT
# ============================================================

if len(numeric_columns) >= 2:

    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column
    )

    plt.title(
        str(x_column)
        + " vs "
        + str(y_column)
    )

    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.tight_layout()

    plt.savefig(
        "scatter_relationship.png"
    )

    plt.show()


# ============================================================
# 15. PAIR PLOT
# ============================================================

if len(numeric_columns) >= 2:

    selected_columns = list(
        numeric_columns[:5]
    )

    sns.pairplot(
        df[selected_columns]
    )

    plt.savefig(
        "pairplot.png"
    )

    plt.show()


# ============================================================
# 16. FIND STRONG CORRELATIONS
# ============================================================

if len(numeric_columns) >= 2:

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    print(
        "\n========== STRONG CORRELATIONS =========="
    )

    for i in range(
        len(correlation_matrix.columns)
    ):

        for j in range(i + 1,
                       len(correlation_matrix.columns)):

            value = correlation_matrix.iloc[i, j]

            if abs(value) >= 0.5:

                col1 = (
                    correlation_matrix.columns[i]
                )

                col2 = (
                    correlation_matrix.columns[j]
                )

                print(
                    col1,
                    "<->",
                    col2,
                    ":",
                    round(value, 2)
                )


# ============================================================
# 17. TOP NUMERICAL STATISTICS
# ============================================================

print("\n========== MEAN VALUES ==========")

for column in numeric_columns:

    print(
        column,
        ":",
        round(df[column].mean(), 2)
    )


print("\n========== MEDIAN VALUES ==========")

for column in numeric_columns:

    print(
        column,
        ":",
        round(df[column].median(), 2)
    )


# ============================================================
# 18. FINAL INSIGHTS
# ============================================================

print("\n")
print("=" * 60)
print("KEY EDA INSIGHTS")
print("=" * 60)

print(
    "\n1. Dataset contains",
    df.shape[0],
    "rows and",
    df.shape[1],
    "columns."
)

print(
    "\n2. Numerical columns:",
    len(numeric_columns)
)

print(
    "3. Categorical columns:",
    len(categorical_columns)
)

print(
    "4. Duplicate rows:",
    duplicates
)

print(
    "5. Total missing values:",
    df.isnull().sum().sum()
)


if len(numeric_columns) >= 2:

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    correlation_matrix = (
        correlation_matrix
        .abs()
    )

    np.fill_diagonal(
        correlation_matrix.values,
        0
    )

    max_correlation = (
        correlation_matrix
        .stack()
        .idxmax()
    )

    max_value = (
        correlation_matrix
        .stack()
        .max()
    )

    print(
        "\n6. Strongest relationship found between:",
        max_correlation[0],
        "and",
        max_correlation[1]
    )

    print(
        "   Correlation strength:",
        round(max_value, 2)
    )


print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
