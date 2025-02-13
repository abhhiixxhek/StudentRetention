# STUDENT RETENTION ANALYSIS

## Overview

This repository provides a framework for conducting machine learning experiments using MLflow for experiment tracking and hyperparameter tuning. The project is designed to be flexible, enabling you to easily configure various machine learning models and their hyperparameters through a YAML configuration file.

## Features

- **MLflow Integration:** Track and log experiments using MLflow.
- **Configurable Setup:** Customize experiments, data paths, and model parameters via `config.yaml`.
- **Multiple Models:** Supports Gradient Boosting, Random Forest, Logistic Regression, SVM, and KNN.
- **Hyperparameter Tuning:** Grid search capabilities over specified hyperparameter ranges.
- **Modular Code Base:** Organized structure for easy maintenance and scalability.

## Repository Structure


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/abhhiixxhek/StudentRetention.git
   cd your-repo
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m .venv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that `requirements.txt` includes necessary packages like `mlflow`, `scikit-learn`, and `pyyaml`.*

## Configuration

The repository uses a YAML configuration file (`config.yaml`) to manage settings:

- **MLflow Configuration:**
  - `tracking_uri`: The URI of your MLflow tracking server (e.g., `"http://localhost:5000"`).
  - `experiment_name`: The name of the experiment for logging purposes.

- **Data Configuration:**
  - `file_path`: Path to your CSV dataset (e.g., `"data/dataset.csv"`).
  - `target_column`: The column in your dataset that represents the target variable.

- **Model Hyperparameters:**
  - **Gradient Boosting:** Configure parameters such as `n_estimators`, `learning_rate`, `max_depth`, `subsample`, etc.
  - **Random Forest:** Define `n_estimators`, `max_depth`, `min_samples_split`, and other parameters.
  - **Logistic Regression, SVM, and KNN:** Specify relevant hyperparameters like `C`, `penalty`, `kernel`, and `n_neighbors`.

## Running Experiments

1. **Start the MLflow Tracking Server:**

   Before running experiments, ensure your MLflow tracking server is active. For example, you can start it using:

   ```bash
    mlflow server --host 0.0.0.0 --port 5000
   ```

2. **Execute the Main Script:**

   Run the main script to load the configuration, train models, and log results to MLflow:

   ```bash
   python main.py
   ```

   The script will:
   - Load and parse `config.yaml`.
   - Load the dataset from the specified file path.
   - Initialize MLflow with the provided `tracking_uri` and set the experiment name.
   - Train each model using the defined hyperparameter grids.
   - Log performance metrics and experiment details to MLflow.

## Customization and Extension

- **Adding New Models:** Extend `src/models.py` to include additional models or modify existing ones.
- **Improving Evaluation:** Enhance the evaluation metrics in `src/main.py` or `src/utils.py` to better suit your project’s needs.
- **Configuration Tweaks:** Update `config.yaml` for different experiments or data sources without changing the code.

## Troubleshooting

- **MLflow Server Issues:** Verify that the MLflow tracking server is running and the `tracking_uri` in `config.yaml` is correct.
- **Data File Issues:** Ensure the dataset file exists at the specified `file_path` and the `target_column` is correctly named.
- **Dependency Problems:** If you encounter errors, check that all required packages are installed and are compatible with your Python version.

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) – For detailed guidance on experiment tracking.
- The community and contributors who help improve this repository.

---
