from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency in some environments
    plt = None
    sns = None

if plt is not None:
    try:
        plt.style.use("fivethirtyeight")
    except OSError:
        pass


def plot_univariate_analysis(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if plt is None or sns is None:
        raise ImportError("matplotlib and seaborn are required for univariate analysis plotting")

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    output_path.mkdir(parents=True, exist_ok=True)

    continuous_columns = [
        col for col in ["ApplicantIncome", "CoapplicantIncome", "Credit_History"] if col in df.columns
    ]
    categorical_columns = [
        col for col in ["Gender", "Education", "Property_Area"] if col in df.columns
    ]

    if continuous_columns:
        fig, axes = plt.subplots(1, len(continuous_columns), figsize=(6 * len(continuous_columns), 4))
        if len(continuous_columns) == 1:
            axes = [axes]
        for ax, column in zip(axes, continuous_columns):
            sns.histplot(df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
        continuous_plot_path = output_path / "univariate_continuous.png"
        fig.tight_layout()
        fig.savefig(continuous_plot_path, dpi=150)
        plt.close(fig)
    else:
        continuous_plot_path = output_path / "univariate_continuous.png"
        continuous_plot_path.touch()

    if categorical_columns:
        fig, axes = plt.subplots(1, len(categorical_columns), figsize=(6 * len(categorical_columns), 4))
        if len(categorical_columns) == 1:
            axes = [axes]
        for ax, column in zip(axes, categorical_columns):
            sns.countplot(data=df, x=column, ax=ax)
            ax.set_title(f"Count plot for {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
        categorical_plot_path = output_path / "univariate_categorical.png"
        fig.tight_layout()
        fig.savefig(categorical_plot_path, dpi=150)
        plt.close(fig)
    else:
        categorical_plot_path = output_path / "univariate_categorical.png"
        categorical_plot_path.touch()

    return {
        "continuous_plot": continuous_plot_path,
        "categorical_plot": categorical_plot_path,
        "summary": {
            "continuous_columns": continuous_columns,
            "categorical_columns": categorical_columns,
        },
    }


def plot_multivariate_analysis(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if plt is None or sns is None:
        raise ImportError("matplotlib and seaborn are required for multivariate analysis plotting")

    output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    output_path.mkdir(parents=True, exist_ok=True)

    required_columns = ["Property_Area", "Education", "Loan_Amount_Term"]
    if not all(col in df.columns for col in required_columns):
        multivariate_plot_path = output_path / "multivariate_swarmplot.png"
        multivariate_plot_path.touch()
        return {
            "multivariate_plot": multivariate_plot_path,
            "summary": {"required_columns": required_columns},
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.swarmplot(
        data=df,
        x="Property_Area",
        y="Loan_Amount_Term",
        hue="Education",
        dodge=True,
        ax=ax,
    )
    ax.set_title("Loan Term Distribution by Property Area and Education")
    ax.set_xlabel("Property Area")
    ax.set_ylabel("Loan Amount Term")
    ax.legend(title="Education", loc="upper right")

    multivariate_plot_path = output_path / "multivariate_swarmplot.png"
    fig.tight_layout()
    fig.savefig(multivariate_plot_path, dpi=150)
    plt.close(fig)

    return {
        "multivariate_plot": multivariate_plot_path,
        "summary": {"required_columns": required_columns},
    }


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from train_model import read_dataset

    dataset_path = repo_root / "data" / "loan_prediction.csv"
    df = read_dataset(dataset_path)
    plot_univariate_analysis(df, output_dir=repo_root / "data")
    plot_multivariate_analysis(df, output_dir=repo_root / "data")
