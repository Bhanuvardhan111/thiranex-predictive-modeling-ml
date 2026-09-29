# ============================================================
# THIRANEX - REAL-WORLD DATA PROJECT
# DOMAIN: RETAIL SALES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("retail_sales.csv")

print("=" * 60)
print("REAL-WORLD RETAIL DATA ANALYSIS")
print("=" * 60)

print("\nDataset loaded successfully!")


# ============================================================
# 2. BASIC DATA EXPLORATION
# ============================================================

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Convert date column
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

# Remove rows with invalid dates
if "Date" in df.columns:
    df = df.dropna(subset=["Date"])


# ============================================================
# 4. STATISTICAL SUMMARY
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ============================================================
# 5. CREATE SALES AND PROFIT COLUMNS
# ============================================================

# If Sales column does not exist but Quantity and Price exist
if (
    "Sales" not in df.columns
    and "Quantity" in df.columns
    and "Price" in df.columns
):

    df["Sales"] = (
        df["Quantity"] * df["Price"]
    )


# If Profit does not exist, calculate an estimated profit
if (
    "Profit" not in df.columns
    and "Sales" in df.columns
):

    if "Cost" in df.columns:

        df["Profit"] = (
            df["Sales"] - df["Cost"]
        )

    else:

        # Example estimated profit margin
        df["Profit"] = (
            df["Sales"] * 0.15
        )


# ============================================================
# 6. BASIC BUSINESS METRICS
# ============================================================

print("\n========== BUSINESS METRICS ==========")

if "Sales" in df.columns:

    total_sales = df["Sales"].sum()

    average_sales = df["Sales"].mean()

    print(
        "Total Sales:",
        round(total_sales, 2)
    )

    print(
        "Average Sales:",
        round(average_sales, 2)
    )


if "Profit" in df.columns:

    total_profit = df["Profit"].sum()

    average_profit = df["Profit"].mean()

    print(
        "Total Profit:",
        round(total_profit, 2)
    )

    print(
        "Average Profit:",
        round(average_profit, 2)
    )


# ============================================================
# 7. MONTHLY SALES ANALYSIS
# ============================================================

if (
    "Date" in df.columns
    and "Sales" in df.columns
):

    df["Month"] = (
        df["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    print("\n========== MONTHLY SALES ==========")
    print(monthly_sales)

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o"
    )

    plt.title(
        "Monthly Sales Trend"
    )

    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "monthly_sales.png"
    )

    plt.show()


# ============================================================
# 8. CATEGORY SALES ANALYSIS
# ============================================================

if (
    "Category" in df.columns
    and "Sales" in df.columns
):

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== SALES BY CATEGORY ==========")
    print(category_sales)

    plt.figure(figsize=(9, 6))

    category_sales.plot(
        kind="bar"
    )

    plt.title(
        "Sales by Product Category"
    )

    plt.xlabel("Category")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "category_sales.png"
    )

    plt.show()


# ============================================================
# 9. TOP PRODUCTS
# ============================================================

if (
    "Product" in df.columns
    and "Sales" in df.columns
):

    top_products = (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print("\n========== TOP 10 PRODUCTS ==========")
    print(top_products)

    plt.figure(figsize=(10, 6))

    top_products.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Products by Sales"
    )

    plt.xlabel("Sales")
    plt.ylabel("Product")

    plt.tight_layout()

    plt.savefig(
        "top_products.png"
    )

    plt.show()


# ============================================================
# 10. REGIONAL SALES
# ============================================================

if (
    "Region" in df.columns
    and "Sales" in df.columns
):

    regional_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\n========== REGIONAL SALES ==========")
    print(regional_sales)

    plt.figure(figsize=(9, 6))

    regional_sales.plot(
        kind="bar"
    )

    plt.title(
        "Sales by Region"
    )

    plt.xlabel("Region")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "regional_sales.png"
    )

    plt.show()


# ============================================================
# 11. PROFIT ANALYSIS
# ============================================================

if (
    "Category" in df.columns
    and "Profit" in df.columns
):

    category_profit = (
        df.groupby("Category")["Profit"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\n========== PROFIT BY CATEGORY ==========")
    print(category_profit)

    plt.figure(figsize=(9, 6))

    category_profit.plot(
        kind="bar"
    )

    plt.title(
        "Profit by Product Category"
    )

    plt.xlabel("Category")
    plt.ylabel("Profit")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "profit_analysis.png"
    )

    plt.show()


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) >= 2:

    correlation = (
        df[numeric_columns]
        .corr()
    )

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
        "Retail Data Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "correlation_heatmap.png"
    )

    plt.show()


# ============================================================
# 13. SALES DISTRIBUTION
# ============================================================

if "Sales" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        df["Sales"],
        kde=True
    )

    plt.title(
        "Sales Distribution"
    )

    plt.xlabel("Sales")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "sales_distribution.png"
    )

    plt.show()


# ============================================================
# 14. KEY INSIGHTS
# ============================================================

print("\n")
print("=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)


if "Sales" in df.columns:

    print(
        "\nTotal Sales:",
        round(df["Sales"].sum(), 2)
    )

    print(
        "Average Sales:",
        round(df["Sales"].mean(), 2)
    )


if "Profit" in df.columns:

    print(
        "Total Profit:",
        round(df["Profit"].sum(), 2)
    )


if (
    "Category" in df.columns
    and "Sales" in df.columns
):

    best_category = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    print(
        "\nBest performing category:",
        best_category
    )


if (
    "Region" in df.columns
    and "Sales" in df.columns
):

    best_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    print(
        "Best performing region:",
        best_region
    )


if (
    "Product" in df.columns
    and "Sales" in df.columns
):

    best_product = (
        df.groupby("Product")["Sales"]
        .sum()
        .idxmax()
    )

    print(
        "Top selling product:",
        best_product
    )


print("\n========================================")
print("REAL-WORLD DATA PROJECT COMPLETED")
print("========================================")