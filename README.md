# 🧬 BreastCancer Subtyping Framework

## 📖 Overview

This repository presents a robust computational framework designed for the comprehensive subtyping of breast cancer across different platforms. Leveraging an advanced ensemble of genomic features, the project employs a rigorous nested validation approach to ensure the high generalizability and clinical applicability of its classification models. The primary goal is to provide a reliable tool for accurate cancer subtype identification, aiding in personalized medicine and research.

## ✨ Features

-   🎯 **Automated Data Retrieval:** Seamlessly retrieve and integrate external genomic datasets.
-   🧬 **Genomic Data Processing & Annotation:** Comprehensive tools for cleaning, processing, and annotating complex genomic features.
-   🧠 **Ensemble Machine Learning:** Utilizes powerful ensemble methods (XGBoost, LightGBM, CatBoost) for robust model training.
-   🔬 **Nested Cross-Validation:** Implements a rigorous nested cross-validation strategy for unbiased model evaluation.
-   🌐 **External Model Validation:** Capabilities for validating trained models on independent, external datasets to assess real-world performance and generalization.
-   📈 **Detailed Performance Reporting:** Generates extensive reports with key metrics to summarize analysis results and model performance.
-   📊 **Insightful Visualizations:** Tools for creating various plots and figures to visualize data patterns, feature importance, and model outcomes.

## 🛠️ Tech Stack

**Programming Language:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Core Libraries:**
-   **Data Manipulation:**
    ![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
    ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
    ![SciPy](https://img.shields.io/badge/SciPy-8F8F8F?style=for-the-badge&logo=scipy&logoColor=white)
-   **Machine Learning & Statistics:**
    ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
    ![XGBoost](https://img.shields.io/badge/XGBoost-008080?style=for-the-badge&logo=xgboost&logoColor=white)
    ![LightGBM](https://img.shields.io/badge/LightGBM-4CAF50?style=for-the-badge&logo=lightgbm&logoColor=white)
    ![CatBoost](https://img.shields.io/badge/CatBoost-000000?style=for-the-badge&logo=catboost&logoColor=white)
    ![Statsmodels](https://img.shields.io/badge/Statsmodels-406180?style=for-the-badge&logo=statsmodels&logoColor=white)
    ![Lifelines](https://img.shields.io/badge/Lifelines-E91E63?style=for-the-badge&logo=anaconda&logoColor=white)
    ![PyRCCA](https://img.shields.io/badge/PyRCCA-FF4500?style=for-the-badge&logo=jupyter&logoColor=white) <!-- No specific logo, using Jupyter as placeholder for scientific computing -->
-   **Visualization:**
    ![Matplotlib](https://img.shields.io/badge/Matplotlib-EE6633?style=for-the-badge&logo=matplotlib&logoColor=white)
    ![Seaborn](https://img.shields.io/badge/Seaborn-3C9A8B?style=for-the-badge&logo=seaborn&logoColor=white)
-   **File I/O:**
    ![OpenPyXL](https://img.shields.io/badge/OpenPyXL-64B5F6?style=for-the-badge&logo=microsoft-excel&logoColor=white) <!-- No specific logo, using Excel as placeholder -->
    ![XLRD](https://img.shields.io/badge/XLRD-80D8FF?style=for-the-badge&logo=microsoft-excel&logoColor=white) <!-- No specific logo, using Excel as placeholder -->

## 🚀 Quick Start

### Prerequisites
-   **Python 3.10**
-   **pip** (Python package installer)

### Installation

1.  **Download Repository**
    ```bash
    git clone https://github.com/ahmadfauzana/BreastCancer.git
    cd BreastCancer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
### Preparation

1.  **Internal Dataset**
Download the dataset [here](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) - Put the dataset into /TCGA-PANCAN-HiSeq-801x20531 directory.

2.  **External Dataset for Validation**
    Download the dataset [here](https://drive.google.com/drive/folders/1_X79EVs8lJoj7Z-8N4OL-Qg3S1urHf3I?usp=sharing) - Put the external dataset into /external_datasets directory.

### Usage

The `main.py` script is the primary entry point for running the entire cancer subtyping pipeline. Individual scripts can also be executed for specific tasks.

To run the complete analysis pipeline:

```bash
python main.py
```

Please refer to the individual script files (`.py`) for specific arguments, input requirements, and execution details.

## 📁 Project Structure

```
BreastCancer/
├── .gitignore               # Specifies intentionally untracked files to ignore
├── LICENSE                  # MIT License details
├── README.md                # This README file
├── external_validator.py    # Script for performing external validation of trained models
├── gene_annotation.py       # Handles gene-level data annotation and processing
├── main.py                  # The main script orchestrating the entire cancer subtyping pipeline
├── process_data.py          # Functions for initial data loading, cleaning, and preprocessing
├── report.py                # Generates comprehensive reports of the analysis results and model performance
├── requirements.txt         # Lists all Python dependencies required for the project
├── retrieve_ext_dataset.py  # Utility to download and prepare external validation datasets
├── train.py                 # Contains logic for training the ensemble machine learning models
└── visualization.py         # Scripts for generating various plots and figures to visualize data and results
```

## 📚 Results & Visualizations

Upon successful execution of the pipeline, various output files including performance metrics, classification results, and visualizations will be generated. These outputs provide insights into the model's accuracy, robustness, and biological interpretations.

### Development Setup for Contributors
Contributions primarily involve modifying the Python scripts. Ensure your environment meets the prerequisites and dependencies listed above.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/ahmadfauzana/BreastCancer/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful for your research or projects!**

Made with 
 by [ahmadfauzana](https://github.com/ahmadfauzana)

</div>
