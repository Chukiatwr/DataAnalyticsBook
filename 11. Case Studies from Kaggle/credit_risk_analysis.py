"""
Credit Risk Analysis: Full Workflow

This script downloads a credit risk dataset from Kaggle and performs:

1. Data understanding
2. Data preparation
3. Exploratory data analysis
4. Outlier analysis
5. Class imbalance analysis
6. Predictive modeling
7. Model evaluation
8. Threshold tuning
9. Cost-sensitive analysis
10. Feature importance analysis
11. Risk segmentation
12. Simple fairness / subgroup analysis
13. Plain-text analysis report printed to screen

Dataset:
    laotse/credit-risk-dataset

Required packages:
    pip install kagglehub pandas numpy matplotlib seaborn scikit-learn

Outputs:
    credit_risk_prepared.csv
    credit_risk_model_ready.csv
    credit_risk_predictions.csv
    credit_risk_scored_with_segments.csv
    tables_credit_risk/
    figures_credit_risk/
"""

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, 
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

np.random.seed(123)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

sns.set_theme(style="whitegrid")

FIG_DIR = Path("figures_credit_risk")
TABLE_DIR = Path("tables_credit_risk")
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

TARGET_COL = "loan_status"
POSITIVE_LABEL = 1


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def save_plot(filename):
    """Save and show the current plot."""
    output_path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {output_path}")


def save_table(df, filename):
    """Save a dataframe to the tables folder."""
    output_path = TABLE_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return output_path


def safe_rate(series):
    """Return mean rate safely."""
    if len(series) == 0:
        return np.nan
    return series.mean()


# ------------------------------------------------------------
# 1. Download dataset from Kaggle
# ------------------------------------------------------------

def download_dataset():
    """Download the credit risk dataset from Kaggle and return the CSV path."""
    dataset_path = kagglehub.dataset_download("laotse/credit-risk-dataset")

    print("Dataset downloaded to:", dataset_path)

    csv_files = list(Path(dataset_path).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset folder.")

    print("CSV file found:", csv_files[0])
    return csv_files[0]


# ------------------------------------------------------------
# 2. Data Understanding
# ------------------------------------------------------------

def understand_data(df):
    """Create basic data understanding summary."""
    print("\n" + "=" * 55)
    print("DATA UNDERSTANDING")
    print("=" * 55)

    print("\nShape of dataset:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData information:")
    df.info()

    numeric_summary = df.describe().T.reset_index().rename(columns={"index": "column"})
    categorical_summary = df.describe(include="object").T.reset_index().rename(columns={"index": "column"})
    missing_summary = (
        pd.DataFrame({
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_percent": df.isnull().mean().values * 100,
            "unique_values": [df[col].nunique() for col in df.columns],
        })
        .sort_values("missing_percent", ascending=False)
    )

    save_table(numeric_summary, "01_numeric_summary.csv")
    save_table(categorical_summary, "02_categorical_summary.csv")
    save_table(missing_summary, "03_missing_summary.csv")

    print("\nNumerical summary:")
    print(numeric_summary)

    print("\nCategorical summary:")
    print(categorical_summary)

    print("\nMissing summary:")
    print(missing_summary)

    if TARGET_COL in df.columns:
        target_dist = (
            df[TARGET_COL]
            .value_counts(dropna=False)
            .rename_axis(TARGET_COL)
            .reset_index(name="count")
        )
        target_dist["percent"] = target_dist["count"] / target_dist["count"].sum() * 100
        save_table(target_dist, "04_target_distribution.csv")

        print("\nTarget distribution:")
        print(target_dist)

    return {
        "shape": df.shape,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "missing_summary": missing_summary,
    }


# ------------------------------------------------------------
# 3. Data Preparation
# ------------------------------------------------------------

def prepare_data(df):
    """Clean, transform, and enrich the credit risk dataset."""
    print("\n" + "=" * 55)
    print("DATA PREPARATION")
    print("=" * 55)

    data = df.copy()

    # Clean column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Fix common typo in column name if found
    if "cb_preson_cred_hist_length" in data.columns:
        data = data.rename(
            columns={"cb_preson_cred_hist_length": "cb_person_cred_hist_length"}
        )

    # Remove duplicated rows
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        data = data.drop_duplicates()
        print(f"Removed duplicated rows: {duplicate_count}")
    else:
        print("No duplicated rows found.")

    # Remove clearly impossible values, but keep valid extreme values
    rows_before = len(data)

    if "person_age" in data.columns:
        data = data[(data["person_age"] >= 18) & (data["person_age"] <= 100)]

    if "person_income" in data.columns:
        data = data[data["person_income"] > 0]

    if "loan_amnt" in data.columns:
        data = data[data["loan_amnt"] > 0]

    rows_after = len(data)
    print(f"Removed impossible-value rows: {rows_before - rows_after}")

    # Separate numerical and categorical columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing values for EDA dataset
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
            print(f"Filled missing numerical values in {col} with median: {median_value}")

    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            print(f"Filled missing categorical values in {col} with mode: {mode_value}")

    # Convert categorical columns to category type
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    # Create readable target label
    if TARGET_COL in data.columns:
        data["loan_status_label"] = data[TARGET_COL].map(
            {
                0: "Non-default",
                1: "Default",
            }
        )

    # Create age group
    if "person_age" in data.columns:
        data["age_group"] = pd.cut(
            data["person_age"],
            bins=[0, 25, 35, 45, 60, 100],
            labels=["<=25", "26-35", "36-45", "46-60", "60+"],
        )

    # Create income group
    if "person_income" in data.columns:
        data["income_group"] = pd.qcut(
            data["person_income"],
            q=4,
            labels=["Low", "Lower-Middle", "Upper-Middle", "High"],
            duplicates="drop",
        )

    # Create loan amount group
    if "loan_amnt" in data.columns:
        data["loan_amount_group"] = pd.qcut(
            data["loan_amnt"],
            q=4,
            labels=["Small", "Medium", "Large", "Very Large"],
            duplicates="drop",
        )

    # Create interest rate group
    if "loan_int_rate" in data.columns:
        data["interest_rate_group"] = pd.qcut(
            data["loan_int_rate"],
            q=4,
            labels=["Low", "Medium", "High", "Very High"],
            duplicates="drop",
        )

    # Create credit history group
    if "cb_person_cred_hist_length" in data.columns:
        data["credit_history_group"] = pd.cut(
            data["cb_person_cred_hist_length"],
            bins=[0, 3, 7, 15, 50],
            labels=["Short", "Moderate", "Long", "Very Long"],
            include_lowest=True,
        )

    model_ready_data = pd.get_dummies(
        data.drop(columns=["loan_status_label"], errors="ignore"),
        drop_first=True,
    )

    print("\nPrepared data sample:")
    print(data.head())

    print("\nModel-ready data shape:")
    print(model_ready_data.shape)

    return data, model_ready_data


# ------------------------------------------------------------
# 4. Exploratory Data Analysis and Visualization
# ------------------------------------------------------------

def analyze_default_rates(data):
    """Analyze default rates by important groups."""
    print("\n" + "=" * 55)
    print("DEFAULT RATE ANALYSIS")
    print("=" * 55)

    group_cols = [
        "age_group",
        "income_group",
        "loan_amount_group",
        "interest_rate_group",
        "credit_history_group",
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    default_rate_tables = {}

    for col in group_cols:
        if {col, TARGET_COL}.issubset(data.columns):
            table = (
                data.groupby(col, observed=True)
                .agg(
                    applicant_count=(TARGET_COL, "size"),
                    default_count=(TARGET_COL, "sum"),
                    default_rate=(TARGET_COL, "mean"),
                )
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            default_rate_tables[col] = table
            save_table(table, f"default_rate_by_{col}.csv")
            print(f"\nDefault rate by {col}:")
            print(table)

    return default_rate_tables


def visualize_data(data):
    """Create exploratory visualizations for credit risk analysis."""
    print("\n" + "=" * 55)
    print("DATA VISUALIZATION")
    print("=" * 55)

    # Target variable distribution
    if "loan_status_label" in data.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=data, x="loan_status_label")
        plt.title("Loan Status Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Number of Applicants")
        save_plot("01_loan_status_distribution.png")

    # Age distribution
    if "person_age" in data.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=data, x="person_age", bins=30, kde=True)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        save_plot("02_age_distribution.png")

    # Income distribution
    if "person_income" in data.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=data, x="person_income", bins=40, kde=True)
        plt.title("Annual Income Distribution")
        plt.xlabel("Annual Income")
        plt.ylabel("Frequency")
        save_plot("03_income_distribution.png")

    # Loan amount distribution
    if "loan_amnt" in data.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=data, x="loan_amnt", bins=30, kde=True)
        plt.title("Loan Amount Distribution")
        plt.xlabel("Loan Amount")
        plt.ylabel("Frequency")
        save_plot("04_loan_amount_distribution.png")

    # Interest rate distribution
    if "loan_int_rate" in data.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=data, x="loan_int_rate", bins=30, kde=True)
        plt.title("Interest Rate Distribution")
        plt.xlabel("Interest Rate")
        plt.ylabel("Frequency")
        save_plot("05_interest_rate_distribution.png")

    # Loan intent distribution
    if "loan_intent" in data.columns:
        plt.figure(figsize=(8, 4))
        order = data["loan_intent"].value_counts().index
        sns.countplot(data=data, y="loan_intent", order=order)
        plt.title("Loan Intent Distribution")
        plt.xlabel("Number of Applicants")
        plt.ylabel("Loan Intent")
        save_plot("06_loan_intent_distribution.png")

    # Default rate by loan intent
    if {"loan_intent", TARGET_COL}.issubset(data.columns):
        default_rate_by_intent = (
            data.groupby("loan_intent", observed=True)[TARGET_COL]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(data=default_rate_by_intent, x=TARGET_COL, y="loan_intent")
        plt.title("Default Rate by Loan Intent")
        plt.xlabel("Default Rate")
        plt.ylabel("Loan Intent")
        save_plot("07_default_rate_by_loan_intent.png")

    # Default rate by loan grade
    if {"loan_grade", TARGET_COL}.issubset(data.columns):
        default_rate_by_grade = (
            data.groupby("loan_grade", observed=True)[TARGET_COL]
            .mean()
            .reset_index()
            .sort_values("loan_grade")
        )

        plt.figure(figsize=(7, 4))
        sns.barplot(data=default_rate_by_grade, x="loan_grade", y=TARGET_COL)
        plt.title("Default Rate by Loan Grade")
        plt.xlabel("Loan Grade")
        plt.ylabel("Default Rate")
        save_plot("08_default_rate_by_loan_grade.png")

    # Income group vs loan status
    if {"income_group", "loan_status_label"}.issubset(data.columns):
        income_status_table = pd.crosstab(data["income_group"], data["loan_status_label"])

        plt.figure(figsize=(7, 4))
        sns.heatmap(income_status_table, annot=True, fmt="d", cmap="Blues")
        plt.title("Income Group vs Loan Status")
        plt.xlabel("Loan Status")
        plt.ylabel("Income Group")
        save_plot("09_income_group_vs_loan_status.png")

    # Loan amount vs interest rate
    if {"loan_amnt", "loan_int_rate", "loan_status_label"}.issubset(data.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=data.sample(min(3000, len(data)), random_state=123),
            x="loan_amnt",
            y="loan_int_rate",
            hue="loan_status_label",
            alpha=0.7,
        )
        plt.title("Loan Amount vs Interest Rate")
        plt.xlabel("Loan Amount")
        plt.ylabel("Interest Rate")
        plt.legend(title="Loan Status")
        save_plot("10_loan_amount_vs_interest_rate.png")

    # Loan percent income vs interest rate
    if {"loan_percent_income", "loan_int_rate", "loan_status_label"}.issubset(data.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=data.sample(min(3000, len(data)), random_state=123),
            x="loan_percent_income",
            y="loan_int_rate",
            hue="loan_status_label",
            alpha=0.7,
        )
        plt.title("Loan Percent Income vs Interest Rate")
        plt.xlabel("Loan Percent Income")
        plt.ylabel("Interest Rate")
        plt.legend(title="Loan Status")
        save_plot("11_loan_percent_income_vs_interest_rate.png")

    # Boxplot: interest rate by loan status
    if {"loan_status_label", "loan_int_rate"}.issubset(data.columns):
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=data, x="loan_status_label", y="loan_int_rate")
        plt.title("Interest Rate by Loan Status")
        plt.xlabel("Loan Status")
        plt.ylabel("Interest Rate")
        save_plot("12_interest_rate_by_loan_status.png")

    # Boxplot: loan percent income by loan status
    if {"loan_status_label", "loan_percent_income"}.issubset(data.columns):
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=data, x="loan_status_label", y="loan_percent_income")
        plt.title("Loan Percent Income by Loan Status")
        plt.xlabel("Loan Status")
        plt.ylabel("Loan Percent Income")
        save_plot("13_loan_percent_income_by_loan_status.png")

    # Correlation heatmap
    numeric_data = data.select_dtypes(include=["int64", "float64"])

    if numeric_data.shape[1] > 1:
        plt.figure(figsize=(9, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        save_plot("14_correlation_heatmap.png")


# ------------------------------------------------------------
# 5. Outlier Analysis
# ------------------------------------------------------------

def analyze_outliers(data):
    """Analyze outliers using the IQR rule."""
    print("\n" + "=" * 55)
    print("OUTLIER ANALYSIS")
    print("=" * 55)

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    rows = []

    for col in numeric_cols:
        if col == TARGET_COL:
            continue

        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outlier_percent = outlier_count / len(data) * 100

        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_percent,
            }
        )

    outlier_table = pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)
    save_table(outlier_table, "outlier_summary.csv")
    print(outlier_table)

    return outlier_table


# ------------------------------------------------------------
# 6. Predictive Modeling
# ------------------------------------------------------------

def build_preprocessor(X):
    """Build preprocessing pipeline for numerical and categorical variables."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_model(name, y_true, y_pred, y_prob):
    """Evaluate classification model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_default": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_default": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_default": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    return metrics


def train_and_evaluate_models(data):
    """Train classification models and compare performance."""
    print("\n" + "=" * 55)
    print("PREDICTIVE MODELING")
    print("=" * 55)

    if TARGET_COL not in data.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    # Exclude analysis-only columns from modeling
    drop_cols = [
        TARGET_COL,
        "loan_status_label",
        "age_group",
        "income_group",
        "loan_amount_group",
        "interest_rate_group",
        "credit_history_group",
    ]

    X = data.drop(columns=drop_cols, errors="ignore")
    y = data[TARGET_COL].astype(int)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=123,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=123,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=123,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=123,
            n_jobs=-1,
        ),
    }

    pipelines = {}
    metrics_rows = []
    predictions = pd.DataFrame(index=X_test.index)
    predictions["actual"] = y_test

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(name, y_test, y_pred, y_prob)
        metrics_rows.append(metrics)

        predictions[f"{name}_prob_default"] = y_prob
        predictions[f"{name}_pred_default"] = y_pred

        pipelines[name] = pipeline

        print(f"\n{name}")
        print(classification_report(y_test, y_pred, target_names=["Non-default", "Default"]))

    metrics_df = pd.DataFrame(metrics_rows).sort_values("recall_default", ascending=False)
    save_table(metrics_df, "model_comparison_metrics.csv")

    print("\nModel comparison:")
    print(metrics_df)

    # Select best model primarily by recall for default, then PR-AUC
    best_model_name = (
        metrics_df
        .sort_values(["recall_default", "pr_auc"], ascending=False)
        .iloc[0]["model"]
    )

    print(f"\nBest model selected for credit-risk focus: {best_model_name}")

    predictions.to_csv("credit_risk_predictions.csv", index=True)
    print("Saved predictions: credit_risk_predictions.csv")

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipelines": pipelines,
        "metrics_df": metrics_df,
        "predictions": predictions,
        "best_model_name": best_model_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


# ------------------------------------------------------------
# 7. Threshold Tuning and Cost-Sensitive Analysis
# ------------------------------------------------------------

def threshold_tuning(model_results):
    """Tune classification threshold for the selected model."""
    print("\n" + "=" * 55)
    print("THRESHOLD TUNING")
    print("=" * 55)

    best_model_name = model_results["best_model_name"]
    y_test = model_results["y_test"]
    predictions = model_results["predictions"]

    y_prob = predictions[f"{best_model_name}_prob_default"].values

    thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_default": precision_score(y_test, y_pred, zero_division=0),
                "recall_default": recall_score(y_test, y_pred, zero_division=0),
                "f1_default": f1_score(y_test, y_pred, zero_division=0),
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "true_negative": tn,
            }
        )

    threshold_df = pd.DataFrame(rows)
    save_table(threshold_df, "threshold_tuning.csv")

    print(threshold_df)

    # Choose a practical threshold with recall >= 0.80 if possible and highest F1
    candidate_df = threshold_df[threshold_df["recall_default"] >= 0.80]

    if len(candidate_df) > 0:
        selected_row = candidate_df.sort_values("f1_default", ascending=False).iloc[0]
    else:
        selected_row = threshold_df.sort_values("f1_default", ascending=False).iloc[0]

    selected_threshold = float(selected_row["threshold"])

    print(f"\nSelected threshold: {selected_threshold:.2f}")

    plt.figure(figsize=(7, 4))
    plt.plot(threshold_df["threshold"], threshold_df["precision_default"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall_default"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1_default"], marker="o", label="F1-score")
    plt.title(f"Threshold Tuning: {best_model_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    save_plot("15_threshold_tuning.png")

    return threshold_df, selected_threshold


def cost_sensitive_analysis(model_results, selected_threshold):
    """Estimate expected cost under different false-positive and false-negative costs."""
    print("\n" + "=" * 55)
    print("COST-SENSITIVE ANALYSIS")
    print("=" * 55)

    best_model_name = model_results["best_model_name"]
    y_test = model_results["y_test"]
    predictions = model_results["predictions"]

    y_prob = predictions[f"{best_model_name}_prob_default"].values
    y_pred = (y_prob >= selected_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    cost_scenarios = [
        {"scenario": "Conservative", "false_positive_cost": 1, "false_negative_cost": 10},
        {"scenario": "Moderate", "false_positive_cost": 1, "false_negative_cost": 5},
        {"scenario": "Growth-oriented", "false_positive_cost": 2, "false_negative_cost": 5},
    ]

    rows = []

    for item in cost_scenarios:
        total_cost = (
            item["false_positive_cost"] * fp
            + item["false_negative_cost"] * fn
        )

        rows.append(
            {
                "scenario": item["scenario"],
                "false_positive_cost": item["false_positive_cost"],
                "false_negative_cost": item["false_negative_cost"],
                "false_positive": fp,
                "false_negative": fn,
                "estimated_total_cost": total_cost,
            }
        )

    cost_df = pd.DataFrame(rows)
    save_table(cost_df, "cost_sensitive_analysis.csv")

    print(cost_df)

    return cost_df


# ------------------------------------------------------------
# 8. Feature Importance and Interpretation
# ------------------------------------------------------------

def get_feature_names(pipeline, numeric_features, categorical_features):
    """Extract feature names from preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]

    output_features = []

    output_features.extend(numeric_features)

    if categorical_features:
        onehot = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_feature_names = onehot.get_feature_names_out(categorical_features)
        output_features.extend(cat_feature_names)

    return output_features


def feature_importance_analysis(model_results):
    """Compute feature importance for the best model and Random Forest."""
    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 55)

    pipelines = model_results["pipelines"]
    numeric_features = model_results["numeric_features"]
    categorical_features = model_results["categorical_features"]

    importance_outputs = {}

    for model_name in ["Logistic Regression", "Random Forest", "Decision Tree"]:
        if model_name not in pipelines:
            continue

        pipeline = pipelines[model_name]
        model = pipeline.named_steps["model"]
        feature_names = get_feature_names(pipeline, numeric_features, categorical_features)

        if hasattr(model, "coef_"):
            values = model.coef_[0]
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": values,
                    "absolute_coefficient": np.abs(values),
                    "direction": np.where(values >= 0, "Higher default risk", "Lower default risk"),
                }
            ).sort_values("absolute_coefficient", ascending=False)

            filename = "feature_importance_logistic_regression.csv"
            save_table(importance_df, filename)

            top_plot = importance_df.head(15).sort_values("coefficient")

            plt.figure(figsize=(8, 6))
            sns.barplot(data=top_plot, x="coefficient", y="feature")
            plt.title("Top Logistic Regression Coefficients")
            plt.xlabel("Coefficient")
            plt.ylabel("Feature")
            save_plot("16_logistic_regression_coefficients.png")

            importance_outputs[model_name] = importance_df

        elif hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": values,
                }
            ).sort_values("importance", ascending=False)

            safe_name = model_name.lower().replace(" ", "_")
            save_table(importance_df, f"feature_importance_{safe_name}.csv")

            top_plot = importance_df.head(15).sort_values("importance")

            plt.figure(figsize=(8, 6))
            sns.barplot(data=top_plot, x="importance", y="feature")
            plt.title(f"Top Feature Importances: {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            save_plot(f"17_feature_importance_{safe_name}.png")

            importance_outputs[model_name] = importance_df

    return importance_outputs


# ------------------------------------------------------------
# 9. Risk Segmentation
# ------------------------------------------------------------

def risk_segmentation(data, model_results, selected_threshold):
    """Create default probability and risk groups using the selected model."""
    print("\n" + "=" * 55)
    print("RISK SEGMENTATION")
    print("=" * 55)

    best_model_name = model_results["best_model_name"]
    pipeline = model_results["pipelines"][best_model_name]

    drop_cols = [
        TARGET_COL,
        "loan_status_label",
        "age_group",
        "income_group",
        "loan_amount_group",
        "interest_rate_group",
        "credit_history_group",
    ]

    X_all = data.drop(columns=drop_cols, errors="ignore")
    result = data.copy()

    result["pd_default_probability"] = pipeline.predict_proba(X_all)[:, 1]

    result["risk_segment"] = pd.cut(
        result["pd_default_probability"],
        bins=[0.0, 0.20, 0.40, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
        include_lowest=True,
    )

    result["decision_suggestion"] = pd.cut(
        result["pd_default_probability"],
        bins=[0.0, 0.20, selected_threshold, 0.60, 1.0],
        labels=[
            "Approve",
            "Approve with caution",
            "Manual review",
            "Reject or require strong mitigation",
        ],
        include_lowest=True,
        duplicates="drop",
    )

    segment_summary = (
        result.groupby("risk_segment", observed=True)
        .agg(
            applicant_count=(TARGET_COL, "size"),
            actual_default_count=(TARGET_COL, "sum"),
            actual_default_rate=(TARGET_COL, "mean"),
            average_predicted_pd=("pd_default_probability", "mean"),
        )
        .reset_index()
    )

    save_table(segment_summary, "risk_segment_summary.csv")
    result.to_csv("credit_risk_scored_with_segments.csv", index=False)

    print(segment_summary)
    print("Saved scored data: credit_risk_scored_with_segments.csv")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=result, x="risk_segment")
    plt.title("Applicant Count by Risk Segment")
    plt.xlabel("Risk Segment")
    plt.ylabel("Number of Applicants")
    save_plot("18_risk_segment_distribution.png")

    return result, segment_summary


# ------------------------------------------------------------
# 10. Simple Fairness / Subgroup Analysis
# ------------------------------------------------------------

def subgroup_analysis(scored_data):
    """Analyze default prediction behavior by broad groups."""
    print("\n" + "=" * 55)
    print("SUBGROUP ANALYSIS")
    print("=" * 55)

    group_cols = [
        "age_group",
        "income_group",
        "person_home_ownership",
        "loan_intent",
    ]

    subgroup_outputs = {}

    for col in group_cols:
        if col not in scored_data.columns:
            continue

        table = (
            scored_data.groupby(col, observed=True)
            .agg(
                applicant_count=(TARGET_COL, "size"),
                actual_default_rate=(TARGET_COL, "mean"),
                average_predicted_pd=("pd_default_probability", "mean"),
            )
            .reset_index()
            .sort_values("average_predicted_pd", ascending=False)
        )

        subgroup_outputs[col] = table
        save_table(table, f"subgroup_analysis_by_{col}.csv")

        print(f"\nSubgroup analysis by {col}:")
        print(table)

    return subgroup_outputs


# ------------------------------------------------------------
# 11. Plain-Text Report Generation
# ------------------------------------------------------------

def fmt_percent(x):
    """Format a number as percentage."""
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def print_section(title):
    """Print a simple report section header."""
    print("\n" + "=" * 55)
    print(title.upper())
    print("=" * 55)


def print_analysis_report(
    data,
    understanding_summary,
    outlier_table,
    default_rate_tables,
    model_results,
    threshold_df,
    selected_threshold,
    cost_df,
    importance_outputs,
    segment_summary,
    subgroup_outputs,
):
    """Print a plain-text analysis report directly to the screen."""
    print_section("Credit Risk Analysis Report")

    metrics_df = model_results["metrics_df"]
    best_model_name = model_results["best_model_name"]

    target_rate = data[TARGET_COL].mean()
    best_metrics = metrics_df[metrics_df["model"] == best_model_name].iloc[0]

    print("\n1. EXECUTIVE SUMMARY")
    print("-" * 80)
    print(
        f"This analysis used {len(data):,} prepared records and "
        f"{data.shape[1]:,} prepared columns."
    )
    print(
        "The target variable is loan_status, where 1 represents Default "
        "and 0 represents Non-default."
    )
    print(f"Observed default rate: {fmt_percent(target_rate)}")
    print(
        f"Selected model for credit-risk use: {best_model_name}. "
        f"Default recall = {fmt_percent(best_metrics['recall_default'])}, "
        f"PR-AUC = {best_metrics['pr_auc']:.3f}."
    )

    print("\n2. DATA UNDERSTANDING")
    print("-" * 80)
    print(f"Prepared dataset shape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print(f"Default rate: {fmt_percent(target_rate)}")
    print(
        "Important variables include borrower age, income, home ownership, "
        "loan intent, loan grade, loan amount, interest rate, loan percent income, "
        "and credit history length."
    )

    print("\n3. DATA PREPARATION PERFORMED")
    print("-" * 80)
    prep_steps = [
        "Cleaned column names by stripping spaces, using lowercase, and replacing spaces with underscores.",
        "Fixed the typo cb_preson_cred_hist_length to cb_person_cred_hist_length when present.",
        "Removed duplicated rows.",
        "Removed clearly impossible records such as age outside 18-100, non-positive income, or non-positive loan amount.",
        "Filled missing numerical values with the median.",
        "Filled missing categorical values with the mode.",
        "Converted categorical variables to pandas category type.",
        "Created analysis groups: age group, income group, loan amount group, interest rate group, and credit history group.",
        "Created a model-ready dataset using one-hot encoding.",
    ]

    for i, step in enumerate(prep_steps, start=1):
        print(f"{i}. {step}")

    print("\n4. OUTLIER ANALYSIS")
    print("-" * 80)
    print("Outliers were identified using the IQR rule.")
    print("Top variables by outlier percentage:")
    print(
        outlier_table[
            ["column", "outlier_count", "outlier_percent"]
        ].head(5).to_string(index=False)
    )
    print(
        "\nOutliers were not automatically removed because extreme values in credit risk "
        "data may still represent valid applicants. Only clearly impossible values were removed."
    )

    print("\n5. DEFAULT RATE ANALYSIS")
    print("-" * 80)
    for col, table in default_rate_tables.items():
        if len(table) == 0:
            continue

        highest = table.iloc[0]
        group_value = highest[col]

        print(
            f"Highest default rate by {col}: {group_value} "
            f"with default rate {fmt_percent(highest['default_rate'])} "
            f"based on {int(highest['applicant_count']):,} applicants."
        )

    print("\n6. MODEL COMPARISON")
    print("-" * 80)
    model_cols = [
        "model",
        "accuracy",
        "precision_default",
        "recall_default",
        "f1_default",
        "roc_auc",
        "pr_auc",
        "false_positive",
        "false_negative",
    ]
    print(metrics_df[model_cols].to_string(index=False))
    print(
        "\nFor credit risk, recall for Default is especially important because "
        "a false negative means an actual defaulting borrower is predicted as Non-default."
    )

    print("\n7. THRESHOLD TUNING")
    print("-" * 80)
    selected_metrics = threshold_df[np.isclose(threshold_df["threshold"], selected_threshold)].iloc[0]
    print(f"Selected probability threshold: {selected_threshold:.2f}")
    print(f"Precision for Default: {fmt_percent(selected_metrics['precision_default'])}")
    print(f"Recall for Default: {fmt_percent(selected_metrics['recall_default'])}")
    print(f"F1-score for Default: {fmt_percent(selected_metrics['f1_default'])}")
    print(
        "A lower threshold usually increases default recall but also increases false positives. "
        "A higher threshold usually reduces false positives but may miss more risky borrowers."
    )

    print("\n8. COST-SENSITIVE ANALYSIS")
    print("-" * 80)
    print(cost_df.to_string(index=False))
    print(
        "\nThis table shows that the preferred decision threshold depends on the relative cost "
        "of approving a borrower who defaults versus rejecting a borrower who would have repaid."
    )

    print("\n9. FEATURE IMPORTANCE AND INTERPRETATION")
    print("-" * 80)

    if "Logistic Regression" in importance_outputs:
        print("\nTop logistic regression coefficients by absolute value:")
        top_logit = importance_outputs["Logistic Regression"].head(10)
        print(top_logit[["feature", "coefficient", "direction"]].to_string(index=False))

    if "Random Forest" in importance_outputs:
        print("\nTop Random Forest feature importances:")
        top_rf = importance_outputs["Random Forest"].head(10)
        print(top_rf[["feature", "importance"]].to_string(index=False))

    print(
        "\nPositive logistic regression coefficients indicate variables associated with higher "
        "default risk, while negative coefficients indicate variables associated with lower "
        "default risk. Tree-based importance values indicate predictive contribution, "
        "but not direction."
    )

    print("\n10. RISK SEGMENTATION")
    print("-" * 80)
    print(segment_summary.to_string(index=False))
    print(
        "\nRisk segmentation converts predicted default probabilities into practical groups "
        "that can support approval, approval with caution, manual review, or stronger mitigation."
    )

    print("\n11. SUBGROUP ANALYSIS")
    print("-" * 80)
    for col, table in subgroup_outputs.items():
        if len(table) == 0:
            continue

        highest = table.iloc[0]
        print(
            f"Highest average predicted default probability by {col}: {highest[col]} "
            f"with average predicted PD {fmt_percent(highest['average_predicted_pd'])}."
        )

    print(
        "\nSubgroup analysis is not a full fairness audit, but it is a useful starting point "
        "for checking whether predictions differ substantially across borrower groups."
    )

    print("\n12. RECOMMENDED BUSINESS INTERPRETATION")
    print("-" * 80)
    recommendations = [
        "Use default recall as a core metric because missing risky borrowers is costly.",
        "Review applicants with high predicted default probability rather than using one rigid approval rule.",
        "Treat high loan-percent-income, high interest rate, weak credit history, and risky loan grades as warning signals when they appear among top model drivers.",
        "Use threshold tuning to balance portfolio growth and credit loss.",
        "Use risk segmentation to support differentiated actions such as approval, manual review, collateral requirement, adjusted pricing, or rejection.",
        "Validate the model on more recent data before operational use.",
    ]

    for i, item in enumerate(recommendations, start=1):
        print(f"{i}. {item}")

    print("\n13. GENERATED OUTPUT FILES")
    print("-" * 80)
    outputs = [
        "credit_risk_prepared.csv",
        "credit_risk_model_ready.csv",
        "credit_risk_predictions.csv",
        "credit_risk_scored_with_segments.csv",
        "tables_credit_risk/",
        "figures_credit_risk/",
    ]

    for item in outputs:
        print(f"- {item}")

    print("\nEnd of report.")


# ------------------------------------------------------------
# 12. Main Program
# ------------------------------------------------------------

def main():
    """Run the full credit risk analysis workflow."""
    csv_path = download_dataset()

    df = pd.read_csv(csv_path)

    understanding_summary = understand_data(df)

    prepared_data, model_ready_data = prepare_data(df)

    prepared_data.to_csv("credit_risk_prepared.csv", index=False)
    model_ready_data.to_csv("credit_risk_model_ready.csv", index=False)

    print("\nPrepared dataset saved as: credit_risk_prepared.csv")
    print("Model-ready dataset saved as: credit_risk_model_ready.csv")

    default_rate_tables = analyze_default_rates(prepared_data)

    visualize_data(prepared_data)

    outlier_table = analyze_outliers(prepared_data)

    model_results = train_and_evaluate_models(prepared_data)

    threshold_df, selected_threshold = threshold_tuning(model_results)

    cost_df = cost_sensitive_analysis(model_results, selected_threshold)

    importance_outputs = feature_importance_analysis(model_results)

    scored_data, segment_summary = risk_segmentation(
        prepared_data,
        model_results,
        selected_threshold,
    )

    subgroup_outputs = subgroup_analysis(scored_data)

    print_analysis_report(
        prepared_data,
        understanding_summary,
        outlier_table,
        default_rate_tables,
        model_results,
        threshold_df,
        selected_threshold,
        cost_df,
        importance_outputs,
        segment_summary,
        subgroup_outputs,
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
