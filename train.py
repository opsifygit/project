# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib


import mlflow

df = pd.read_csv("processed_data.csv")

    # Split data
X = df.drop(["Reading", "Timestamp"], axis=1)
y = df["Reading"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

learning_rate = 0.1
n_estimators = 50
max_depth = 3
subsample = 0.8
colsample_bytree = 0.8

# Model Selection
model = XGBRegressor(
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=42,
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model

joblib.dump(model, "app/model.pkl")


try:
    # MLFLOW
    EXPERIMENT_NAME = "mlflow-xgboost"
    EXPERIMENT_ID = 0
    try:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    except:
        EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    # increment idx based on the number of previous runs in the experiment
    idx = len(mlflow.search_runs(EXPERIMENT_ID)) + 1
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)


        # Track metrics
        mlflow.log_metric("MSE", mse)

        # Track model
        mlflow.sklearn.log_model(model, "xgboost-model")
except:
    print("MLflow Exception")
