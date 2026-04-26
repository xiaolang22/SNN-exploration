"""
Interactive cross-round generalization analysis for in-vitro neural network data.

Workflow:
1. Pop up a file picker to select one .xlsx file.
2. Validate that the workbook has exactly three sheets, each with 120 rows
   and 101 columns. The last column must be the label column.
3. Evaluate cross-round generalization with Random Forest, Ridge Classifier
   and Logistic Regression:
   - train on one sheet, test on another sheet
   - train on two sheets, test on the remaining sheet
4. Create one result folder containing:
   - the cross-round accuracy figure
   - the accuracy table
   - a copy of the source Excel file
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_ORDER = ["一一", "一二", "一三", "二一", "二二", "二三"]
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one Excel file and run cross-round generalization analysis."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Excel file path. If omitted, a file picker dialog is shown.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_results"),
        help="Root directory for the result folder. Default: analysis_results",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stochastic models. Default: 42",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for Random Forest. Default is 1 for Windows compatibility.",
    )
    return parser.parse_args()


def select_excel_file() -> Path:
    try:
        from tkinter import Tk, filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is not available. Please run with --file <path>.") from exc

    root = Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select experimental output Excel file",
        filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
    )
    root.destroy()

    if not selected:
        raise RuntimeError("No Excel file was selected.")
    return Path(selected)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")

    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_shared_strings(xlsx: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []

    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for si in root.findall(f"{{{MAIN_NS}}}si"):
        text_parts = [node.text or "" for node in si.iter() if local_name(node.tag) == "t"]
        shared_strings.append("".join(text_parts))
    return shared_strings


def workbook_sheet_map(excel_file: Path) -> dict[str, str]:
    with zipfile.ZipFile(excel_file) as xlsx:
        workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
        rels_root = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))

        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }

        sheet_map: dict[str, str] = {}
        for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = rel_targets[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_map[sheet_name] = target.replace("\\", "/")
    return sheet_map


def sheet_names(excel_file: Path) -> list[str]:
    try:
        return pd.ExcelFile(excel_file).sheet_names
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        return list(workbook_sheet_map(excel_file))


def read_xlsx_sheet_without_openpyxl(excel_file: Path, sheet_name: str) -> pd.DataFrame:
    sheet_map = workbook_sheet_map(excel_file)
    if sheet_name not in sheet_map:
        raise ValueError(f"Sheet '{sheet_name}' was not found. Available sheets: {list(sheet_map)}")

    with zipfile.ZipFile(excel_file) as xlsx:
        shared_strings = read_shared_strings(xlsx)
        worksheet = ET.fromstring(xlsx.read(sheet_map[sheet_name]))

    rows: list[list[Any]] = []
    max_cols = 0
    for row in worksheet.findall(f".//{{{MAIN_NS}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            cell_ref = cell.attrib.get("r", "")
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{{{MAIN_NS}}}v")
            inline_node = cell.find(f"{{{MAIN_NS}}}is")
            value: Any = None

            if cell_type == "s" and value_node is not None:
                value = shared_strings[int(value_node.text or 0)]
            elif cell_type == "inlineStr" and inline_node is not None:
                text_parts = [
                    node.text or "" for node in inline_node.iter() if local_name(node.tag) == "t"
                ]
                value = "".join(text_parts)
            elif value_node is not None:
                value = value_node.text

            if cell_ref:
                col_idx = column_index(cell_ref)
                values[col_idx] = value
                max_cols = max(max_cols, col_idx + 1)

        if values:
            row_values = [None] * max(max_cols, max(values) + 1)
            for col_idx, value in values.items():
                row_values[col_idx] = value
            rows.append(row_values)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    normalized_rows = [row + [None] * (width - len(row)) for row in rows]
    return pd.DataFrame(normalized_rows)


def read_sheet(excel_file: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        return read_xlsx_sheet_without_openpyxl(excel_file, sheet_name)


def validate_sheet_names(names: list[str]) -> list[str]:
    if len(names) != 3:
        raise ValueError(f"Workbook must contain exactly 3 sheets, but found {len(names)}: {names}")
    return names


def load_and_validate_sheet(excel_file: Path, sheet_name: str) -> tuple[np.ndarray, np.ndarray]:
    df = read_sheet(excel_file, sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.shape != (120, 101):
        raise ValueError(
            f"Sheet '{sheet_name}' must be exactly 120 rows x 101 columns after removing "
            f"empty rows/columns, but got {df.shape[0]} rows x {df.shape[1]} columns."
        )

    feature_df = df.iloc[:, :100].apply(pd.to_numeric, errors="coerce")
    if feature_df.isna().any().any():
        raise ValueError(f"Sheet '{sheet_name}' contains non-numeric or missing feature values.")

    label_series = df.iloc[:, 100].astype(str).str.strip()
    unexpected_labels = sorted(set(label_series) - set(LABEL_ORDER))
    if unexpected_labels:
        raise ValueError(
            f"Sheet '{sheet_name}' contains unexpected labels {unexpected_labels}. "
            f"Expected labels: {LABEL_ORDER}"
        )

    counts = label_series.value_counts()
    bad_counts = {
        label: int(counts.get(label, 0))
        for label in LABEL_ORDER
        if int(counts.get(label, 0)) != 20
    }
    if bad_counts:
        raise ValueError(
            f"Sheet '{sheet_name}' must contain exactly 20 rows per label. "
            f"Invalid counts: {bad_counts}"
        )

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    x = feature_df.to_numpy(dtype=float)
    y = label_series.map(label_to_index).to_numpy(dtype=int)
    return x, y


def build_models(random_state: int, n_jobs: int) -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Ridge Classifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier(alpha=1.0)),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_train_test(
    models: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for model_name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        y_pred = fitted_model.predict(x_test)
        accuracies[model_name] = float(accuracy_score(y_test, y_pred))
    return accuracies


def run_cross_round_analysis(
    models: dict[str, Any],
    data_by_sheet: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sheet_list = list(data_by_sheet)

    for train_sheet in sheet_list:
        x_train, y_train = data_by_sheet[train_sheet]
        for test_sheet in sheet_list:
            if train_sheet == test_sheet:
                continue

            x_test, y_test = data_by_sheet[test_sheet]
            accuracies = evaluate_train_test(models, x_train, y_train, x_test, y_test)
            for model_name, accuracy in accuracies.items():
                rows.append(
                    {
                        "analysis": "single_sheet_train",
                        "model": model_name,
                        "train_sheets": train_sheet,
                        "test_sheet": test_sheet,
                        "accuracy": accuracy,
                    }
                )

    for train_sheets in combinations(sheet_list, 2):
        test_sheet = next(sheet for sheet in sheet_list if sheet not in train_sheets)
        train_sets = [data_by_sheet[sheet] for sheet in train_sheets]
        x_train = np.vstack([item[0] for item in train_sets])
        y_train = np.concatenate([item[1] for item in train_sets])
        x_test, y_test = data_by_sheet[test_sheet]

        accuracies = evaluate_train_test(models, x_train, y_train, x_test, y_test)
        for model_name, accuracy in accuracies.items():
            rows.append(
                {
                    "analysis": "two_sheet_train",
                    "model": model_name,
                    "train_sheets": " + ".join(train_sheets),
                    "test_sheet": test_sheet,
                    "accuracy": accuracy,
                }
            )

    return pd.DataFrame(rows)


def unique_result_dir(output_root: Path, excel_stem: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"{excel_stem}_cross_round_result"
    if not base.exists():
        base.mkdir()
        return base

    index = 2
    while True:
        candidate = output_root / f"{excel_stem}_cross_round_result_{index}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        index += 1


def plot_cross_round_results(results: pd.DataFrame, sheet_list: list[str], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    model_names = ["Random Forest", "Ridge Classifier", "Logistic Regression"]
    fig = plt.figure(figsize=(8.6, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])

    single_train = results[results["analysis"] == "single_sheet_train"]
    two_train = results[results["analysis"] == "two_sheet_train"]

    heatmap_image = None
    for index, model_name in enumerate(model_names):
        ax = fig.add_subplot(grid[0, index])
        matrix = np.full((len(sheet_list), len(sheet_list)), np.nan)
        model_rows = single_train[single_train["model"] == model_name]
        for _, row in model_rows.iterrows():
            train_index = sheet_list.index(row["train_sheets"])
            test_index = sheet_list.index(row["test_sheet"])
            matrix[train_index, test_index] = row["accuracy"]

        masked_matrix = np.ma.masked_invalid(matrix)
        cmap = plt.cm.Greys.copy()
        cmap.set_bad(color="white")
        heatmap_image = ax.imshow(masked_matrix, vmin=0, vmax=1, cmap=cmap)

        for row_index in range(len(sheet_list)):
            for col_index in range(len(sheet_list)):
                if row_index == col_index:
                    ax.text(col_index, row_index, "-", ha="center", va="center", color="#666666")
                else:
                    ax.text(
                        col_index,
                        row_index,
                        f"{matrix[row_index, col_index]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        ax.set_title(model_name, fontsize=10)
        ax.set_xticks(np.arange(len(sheet_list)))
        ax.set_yticks(np.arange(len(sheet_list)))
        ax.set_xticklabels(sheet_list, rotation=35, ha="right")
        ax.set_yticklabels(sheet_list)
        ax.set_xlabel("Test sheet")
        if index == 0:
            ax.set_ylabel("Train sheet")

    if heatmap_image is not None:
        colorbar = fig.colorbar(heatmap_image, ax=fig.axes[:3], fraction=0.025, pad=0.02)
        colorbar.set_label("Accuracy")

    ax_bar = fig.add_subplot(grid[1, :])
    x_positions = np.arange(len(sheet_list))
    bar_width = 0.23
    colors = ["#1f77b4", "#4d4d4d", "#d62728"]

    for model_index, model_name in enumerate(model_names):
        model_rows = two_train[two_train["model"] == model_name]
        values = []
        for test_sheet in sheet_list:
            value = model_rows.loc[model_rows["test_sheet"] == test_sheet, "accuracy"].iloc[0]
            values.append(value)

        offsets = x_positions + (model_index - 1) * bar_width
        ax_bar.bar(
            offsets,
            values,
            width=bar_width,
            label=model_name,
            color=colors[model_index],
            edgecolor="black",
            linewidth=0.4,
        )

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([f"Test {sheet}" for sheet in sheet_list])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_title("Train on two sheets, test on the remaining sheet", fontsize=10)
    ax_bar.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax_bar.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    excel_file = args.file if args.file is not None else select_excel_file()
    excel_file = excel_file.resolve()

    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    if excel_file.suffix.lower() != ".xlsx":
        raise ValueError(f"Only .xlsx files are supported, but got: {excel_file}")

    selected_sheet_names = validate_sheet_names(sheet_names(excel_file))
    data_by_sheet = {
        sheet_name: load_and_validate_sheet(excel_file, sheet_name)
        for sheet_name in selected_sheet_names
    }
    models = build_models(args.random_state, args.n_jobs)
    result_df = run_cross_round_analysis(models, data_by_sheet)

    result_dir = unique_result_dir(args.output_root, excel_file.stem)
    figure_path = result_dir / "cross_round_accuracy.png"
    accuracy_path = result_dir / "cross_round_accuracy_results.csv"
    excel_copy_path = result_dir / excel_file.name

    plot_cross_round_results(result_df, selected_sheet_names, figure_path)
    result_df.to_csv(accuracy_path, index=False, encoding="utf-8-sig")
    shutil.copy2(excel_file, excel_copy_path)

    print("Cross-round generalization results:")
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Result folder: {result_dir}")
    print(f"Saved plot: {figure_path}")
    print(f"Saved accuracy table: {accuracy_path}")
    print(f"Copied source Excel: {excel_copy_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
