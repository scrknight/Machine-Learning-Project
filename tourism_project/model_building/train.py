"""
Model Building with Experimentation Tracking
---------------------------------------------
Loads the train/test data from the Hugging Face dataset repo, defines a
model + hyperparameter grid, tunes it, logs every run with MLflow,
evaluates the best model, and registers it on the Hugging Face model hub.

Change SELECTED_MODEL below to switch algorithms. Supported options:
"decision_tree", "bagging", "random_forest", "adaboost",
"gradient_boosting", "xgboost"
"""
import os
import joblib
import pandas as pd
import mlflow
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

DATASET_REPO_ID = "scrfrob/tourism-package-dataset"
MODEL_REPO_ID = "scrfrob/tourism-package-model"
SELECTED_MODEL = "xgboost"
MODEL_FILENAME = "best_tourism_model.joblib"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-package-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train/test data directly from the Hugging Face dataset space
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO_ID}/ytest.csv").values.ravel()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# Define the model + hyperparameter grid for every supported algorithm
MODEL_REGISTRY = {
    "decision_tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"decisiontreeclassifier__max_depth": [3, 5, 7, None]},
    ),
    "bagging": (
        BaggingClassifier(random_state=42),
        {"baggingclassifier__n_estimators": [30, 50, 100]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 7, None],
        },
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {"adaboostclassifier__n_estimators": [50, 100, 150]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "gradientboostingclassifier__n_estimators": [100, 150],
            "gradientboostingclassifier__max_depth": [2, 3],
        },
    ),
    "xgboost": (
        xgb.XGBClassifier(
            scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"
        ),
        {
            "xgbclassifier__n_estimators": [50, 100, 150],
            "xgbclassifier__max_depth": [2, 3, 4],
            "xgbclassifier__learning_rate": [0.05, 0.1],
        },
    ),
}

base_model, param_grid = MODEL_REGISTRY[SELECTED_MODEL]
model_pipeline = make_pipeline(preprocessor, base_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried, as nested runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the winning parameters and the algorithm choice on the parent run
    mlflow.log_param("selected_model", SELECTED_MODEL)
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_recall_1": train_report["1"]["recall"],
            "train_f1_1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_recall_1": test_report["1"]["recall"],
            "test_f1_1": test_report["1"]["f1-score"],
        }
    )
    print("Test set performance:", test_report["accuracy"], test_report["1"])

    # Save the best model locally and log it as an MLflow artifact
    joblib.dump(best_model, MODEL_FILENAME)
    mlflow.log_artifact(MODEL_FILENAME, artifact_path="model")

    # Register the best model on the Hugging Face model hub
    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{MODEL_REPO_ID}' not found. Creating it...")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print("Model registered on the Hugging Face Hub.")
