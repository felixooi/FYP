"""
Module 7: Data Partition
Splits data into training, validation, and test sets for ML model training readiness.
"""

import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def partition_data(df, target_col='Resigned', test_size=0.15, val_size=0.15, random_state=42, visualize=True):
    """Partition data into train, validation, and test sets with stratification."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if test_size + val_size >= 1:
        raise ValueError("Sum of test_size and val_size must be less than 1.")
    if df[target_col].nunique() < 2:
        raise ValueError("Target variable must have at least two classes.")

    logging.info("Starting data partitioning...")
    X, y = df.drop(columns=[target_col]), df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    logging.info(f"Data successfully partitioned: "
                 f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sizes = [len(X_train), len(X_val), len(X_test)]
        labels = ['Training', 'Validation', 'Test']
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Dataset Partition Distribution', fontsize=13, fontweight='bold')

        class_dist = pd.DataFrame({
            'Training': y_train.value_counts(normalize=False),
            'Validation': y_val.value_counts(normalize=False),
            'Test': y_test.value_counts(normalize=False)
        }).T
        class_dist.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Class Distribution Across Splits', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Dataset Split')
        axes[1].tick_params(axis='x', rotation=0)
        plt.tight_layout()
        plt.savefig('outputs/13_data_partition.png', dpi=300)
        plt.show()

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_partition_report(X_train, X_val, X_test, y_train, y_val, y_test):
    """Generate partition quality summary."""
    report = pd.DataFrame({
        'Split': ['Training', 'Validation', 'Test'],
        'Samples': [len(X_train), len(X_val), len(X_test)],
        'Class_0': [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()],
        'Class_1': [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()]
    })
    report['Attrition_Rate_%'] = (report['Class_1'] / report['Samples'] * 100).round(2)
    report['Percentage'] = (report['Samples'] / report['Samples'].sum() * 100).round(2)

    # Display report (compatible with both Jupyter and regular Python)
    try:
        from IPython.display import display
        display(report.style.set_caption("Partition Quality Report").set_table_styles([
            {'selector': 'table', 'props': [('overflow-y', 'scroll'), ('height', '250px')]}
        ]))
    except ImportError:
        print("Partition Quality Report")
        print(report.to_string())

    # Stratification quality
    ratios = [y_train.mean(), y_val.mean(), y_test.mean()]
    diff = max(abs(ratios[0] - ratios[1]), abs(ratios[0] - ratios[2]), abs(ratios[1] - ratios[2]))
    logging.info(f"Stratification check: max class-ratio diff = {diff*100:.2f}%")
    return report


def save_partitioned_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data', random_state=42):
    """Save partitioned data and metadata to CSV and JSON."""
    output_dir = output_dir.rstrip('/')
    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    for name, (X, y) in datasets.items():
        df_out = X.copy()
        df_out['Resigned'] = y.values
        path = f"{output_dir}/{name}_data.csv"
        df_out.to_csv(path, index=False)
        logging.info(f"Saved {name} data → {path} ({df_out.shape})")

    # Save metadata JSON
    metadata = {
        'random_state': random_state,
        'samples': {k: len(v[0]) for k, v in datasets.items()},
        'class_distribution': {k: v[1].value_counts(normalize=True).to_dict() for k, v in datasets.items()}
    }
    with open(f"{output_dir}/partition_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info("Partition metadata saved as 'partition_metadata.json'.")

def create_data_partition(df, target_col='Resigned', test_size=0.15, val_size=0.15,
                          random_state=42, save_data=True, output_dir='data', visualize=True):
    """Main partition pipeline for dataset preparation."""
    logging.info("==== DATA PARTITION PIPELINE START ====")
    X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
        df, target_col, test_size, val_size, random_state, visualize
    )
    report = generate_partition_report(X_train, X_val, X_test, y_train, y_val, y_test)
    if save_data:
        save_partitioned_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, random_state)
    logging.info("==== DATA PARTITION PIPELINE COMPLETE ====")
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'report': report
    }
