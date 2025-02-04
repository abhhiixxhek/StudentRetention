import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import randint, uniform

# --- Unsupervised learning imports (for reference/visualization) ---
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

#############################################
# Load Configuration and Setup MLflow       #
#############################################

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

#############################################
# Load Data                                 #
#############################################

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

#############################################
# Enhanced Data Preprocessing               #
#############################################

def preprocess_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Advanced Data Cleaning ---
    # Impute missing values for numeric columns using median strategy
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # For categorical columns, fill missing with 'missing'
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    X[categorical_cols] = X[categorical_cols].fillna('missing')

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Optional: Feature Selection using ANOVA F-test ---
    selector = SelectKBest(score_func=f_classif, k=min(50, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode target variable
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    return X_train, X_test, y_train_enc, y_test_enc, le

#############################################
# Model Configuration with Improved Tuning  #
#############################################

def configure_models(model_config):
    models = {}

    # Using SMOTE in the pipeline for balanced classes
    common_steps = [("imputer", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=42)),
                    ("scaler", StandardScaler())]

    if "Random Forest" in model_config:
        models["Random Forest"] = {
            "pipeline": ImbPipeline( common_steps + [("model", RandomForestClassifier())] ),
            "params": {
                "model__n_estimators": randint(100, 300),
                "model__max_depth": randint(5, 20),
                "model__min_samples_split": randint(2, 10)
            }
        }

    if "Gradient Boosting" in model_config:
        models["Gradient Boosting"] = {
            "pipeline": ImbPipeline( common_steps + [("model", GradientBoostingClassifier())] ),
            "params": {
                "model__n_estimators": randint(100, 300),
                "model__learning_rate": uniform(0.01, 0.3),
                "model__max_depth": randint(3, 10)
            }
        }

    if "Logistic Regression" in model_config:
        models["Logistic Regression"] = {
            "pipeline": ImbPipeline( common_steps + [("model", LogisticRegression(solver="liblinear", max_iter=500))] ),
            "params": {
                "model__C": uniform(0.1, 10),
                "model__penalty": ["l1", "l2"]
            }
        }

    if "SVM" in model_config:
        models["SVM"] = {
            "pipeline": ImbPipeline( common_steps + [("model", SVC(probability=True))] ),
            "params": {
                "model__C": uniform(0.1, 10),
                "model__kernel": ["linear", "rbf"]
            }
        }

    if "KNN" in model_config:
        models["KNN"] = {
            "pipeline": ImbPipeline( common_steps + [("model", KNeighborsClassifier())] ),
            "params": {
                "model__n_neighbors": randint(3, 15),
                "model__weights": ["uniform", "distance"]
            }
        }

    return models

#############################################
# Extended Model Training with MLflow       #
#############################################

def train_models_with_mlflow(models, X_train, y_train, X_test, y_test):
    best_models = {}
    # Increase n_iter for deeper search
    search_iter = 20
    for name, model_config in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name} ...")
            search = RandomizedSearchCV(
                model_config["pipeline"],
                model_config["params"],
                n_iter=search_iter,
                cv=5,
                scoring="accuracy",
                random_state=42,
                verbose=1,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_models[name] = best_model

            # Log parameters and metrics
            mlflow.log_params(search.best_params_)
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(best_model, name)
            print(f"Best Accuracy for {name}: {acc:.4f}")
    return best_models

#############################################
# Advanced Ensemble with Stacking           #
#############################################

def train_ensemble(best_models, X_train, y_train, X_test, y_test, label_encoder):
    with mlflow.start_run(run_name="Advanced_Ensemble_Model"):
        # Use stacking classifier that uses two levels of models
        estimators = []
        for name, model in best_models.items():
            estimators.append((name, model))
        # Final estimator can be a logistic regression model
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(solver="liblinear"),
            cv=5,
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("ensemble_accuracy", acc)
        mlflow.sklearn.log_model(ensemble, "AdvancedEnsembleModel")
        print(f"Advanced Ensemble Model Accuracy: {acc:.4f}")

        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
        sns.heatmap(conf_matrix, annot=True, fmt="d",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.show()

        return ensemble

#############################################
# Unsupervised Learning Function (Optional)  #
#############################################

def unsupervised_learning():
    # Generate sample data using make_blobs
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def plot_clusters(X, labels, title):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', alpha=0.7)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(title='Cluster')
        plt.show()

    # Elbow Method
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker="o")
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # K-Means Example
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print(f"K-Means Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    plot_clusters(X_scaled, labels, "K-Means Clustering")

#############################################
# Main Execution                            #
#############################################

def main():
    # --- Supervised Learning ---
    df = load_data(config["data"]["file_path"])
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df, config["data"]["target_column"])
    models = configure_models(config["models"])
    best_models = train_models_with_mlflow(models, X_train, y_train, X_test, y_test)
    train_ensemble(best_models, X_train, y_train, X_test, y_test, label_encoder)

    # --- Optional: Unsupervised Analysis ---
    unsupervised_learning()

if __name__ == "__main__":
    main()
