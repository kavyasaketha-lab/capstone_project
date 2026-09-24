# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
charts_path=BASE_DIR/"ModelCharts"
if not os.path.exists(charts_path):
    os.makedirs(charts_path)
# ============================================================
# 2. LOAD CLEANED TITANIC DATASET
# ============================================================

# Change the filename here if your cleaned dataset has
# a different filename.

df = pd.read_csv(f'{BASE_DIR}\\titanic.csv')



print("=" * 70)
print("CLEANED TITANIC DATASET")
print("=" * 70)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

missing_required_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required_columns:
    raise ValueError(
        f"The following required columns are missing: "
        f"{missing_required_columns}"
    )

print("\nAll required columns are available.")


# ============================================================
# TASK 7
# TRAIN / TEST SPLIT USING STRATIFICATION
# ============================================================

target = "survived"

X = df.drop(columns=[target])
y = df[target]


# ------------------------------------------------------------
# Class balance before splitting
# ------------------------------------------------------------

class_counts = y.value_counts().sort_index()

class_percentages = (
    y.value_counts(normalize=True)
    .sort_index() * 100
)

class_balance = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percentages
})

class_balance.index = [
    "Not Survived (0)",
    "Survived (1)"
]

print("\n" + "=" * 70)
print("TASK 7 - CLASS BALANCE")
print("=" * 70)

display(class_balance.round(2))


# ------------------------------------------------------------
# Stratified train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape :", X_test.shape)

print("\nTraining target distribution:")
print(
    y_train.value_counts(normalize=True)
    .sort_index()
    .round(3)
)

print("\nTesting target distribution:")
print(
    y_test.value_counts(normalize=True)
    .sort_index()
    .round(3)
)

print(
    "\nStratification is used because the target classes are\n "
    "not perfectly balanced. It preserves approximately the \n"
    "same survivor/non-survivor proportion in both train and \n"
    "test sets."
)


# ============================================================
# TASK 8
# PREPROCESSING
# ============================================================

# We use the following features:
#
# Numeric:
#   pclass, age, sibsp, parch, fare
#
# Categorical:
#   sex, embarked
#
# Missing numeric values:
#   Median imputation
#
# Missing categorical values:
#   Most-frequent imputation
#
# Categorical encoding:
#   One-hot encoding
#
# Numeric scaling:
#   StandardScaler
#
# All preprocessing is fitted only on the training data
# through the Pipeline.


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# ------------------------------------------------------------
# Numeric preprocessing
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# Categorical preprocessing
# ------------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# Combined preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("\n" + "=" * 70)
print("TASK 8 - PREPROCESSING PIPELINE CREATED")
print("=" * 70)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Numeric missing values: median")
print("Categorical missing values: most frequent")
print("Categorical encoding: One-Hot Encoding")
print("Numeric scaling: StandardScaler")


# ============================================================
# TASK 9
# LOGISTIC REGRESSION
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_pipeline.fit(X_train,y_train)

logistic_predictions = logistic_pipeline.predict(X_test)

logistic_probabilities = (logistic_pipeline.predict_proba(X_test)[:, 1])

print("\nLogistic Regression trained successfully.")


# ============================================================
# TASK 9
# DECISION TREE
# ============================================================

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

decision_tree_pipeline.fit(X_train,y_train)

dt_predictions = decision_tree_pipeline.predict(X_test)

dt_probabilities = (decision_tree_pipeline.predict_proba(X_test)[:, 1])

print("Decision Tree trained successfully.")


# ============================================================
# TASK 9
# RANDOM FOREST
# ============================================================

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest_pipeline.fit(X_train,y_train)

rf_predictions = random_forest_pipeline.predict(X_test)

rf_probabilities = (random_forest_pipeline.predict_proba(X_test)[:, 1])

print("Random Forest trained successfully.")


# ============================================================
# TASK 9
# DECISION TREE VISUALIZATION
# ============================================================

dt_preprocessor = (decision_tree_pipeline.named_steps["preprocessor"])
dt_model = (decision_tree_pipeline.named_steps["classifier"])
feature_names = (dt_preprocessor.get_feature_names_out())
plt.figure(figsize=(16, 12))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Classifier")
filename="Decisiontree.png"
plt.savefig(charts_path/filename)
plt.show()


# ============================================================
# TASK 10
# CLASSIFICATION EVALUATION FUNCTION
# ============================================================

def evaluate_classifier(y_true,predictions,probabilities,model_name):

    accuracy = accuracy_score(y_true,predictions)
    precision = precision_score(y_true,predictions,zero_division=0)
    recall = recall_score(y_true,predictions,zero_division=0)
    f1 = f1_score(y_true,predictions,zero_division=0)
    auc = roc_auc_score(y_true,probabilities)
    cm = confusion_matrix(y_true,predictions)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Confusion Matrix": cm
    }


# ============================================================
# TASK 10
# EVALUATE ALL THREE CLASSIFIERS
# ============================================================

classification_results = []

classification_results.append(
    evaluate_classifier(
        y_test,
        logistic_predictions,
        logistic_probabilities,
        "Logistic Regression"
    )
)

classification_results.append(
    evaluate_classifier(
        y_test,
        dt_predictions,
        dt_probabilities,
        "Decision Tree"
    )
)

classification_results.append(
    evaluate_classifier(
        y_test,
        rf_predictions,
        rf_probabilities,
        "Random Forest"
    )
)


classification_comparison = pd.DataFrame(
    classification_results
)

classification_comparison = (
    classification_comparison[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC"
        ]
    ]
)

print("\n" + "=" * 70)
print("TASK 10 - CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

display(
    classification_comparison.round(4)
)


# ============================================================
# TASK 10
# CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1,3,figsize=(18, 5))

models_for_cm = [
    ("Logistic Regression",logistic_predictions),
    ("Decision Tree",dt_predictions),
    ("Random Forest",rf_predictions)
]

for ax, (model_name, predictions) in zip(axes,models_for_cm):

    cm = confusion_matrix(y_test,predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Survived",
            "Survived"
        ]
    )

    disp.plot(ax=ax,values_format="d")
    ax.set_title(model_name)

plt.tight_layout()
filename="ComparingConfusionMatrix.png"
plt.savefig(charts_path/filename)
plt.show()


# ============================================================
# TASK 10
# ROC CURVES AND AUC
# ============================================================

fpr_lr, tpr_lr, _ = roc_curve(y_test,logistic_probabilities)
fpr_dt, tpr_dt, _ = roc_curve(y_test,dt_probabilities)
fpr_rf, tpr_rf, _ = roc_curve(y_test,rf_probabilities)

auc_lr = roc_auc_score(y_test,logistic_probabilities)
auc_dt = roc_auc_score(y_test,dt_probabilities)
auc_rf = roc_auc_score(y_test,rf_probabilities)


plt.figure(figsize=(9, 7))

plt.plot(fpr_lr,tpr_lr,label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_dt,tpr_dt,label=f"Decision Tree (AUC = {auc_dt:.3f})")
plt.plot(fpr_rf,tpr_rf,label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1],[0, 1],linestyle="--",label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Titanic Classification")

plt.legend()
plt.grid()
filename="aucroc_compare.png"
plt.savefig(charts_path/filename)
plt.show()


# ============================================================
# TASK 11
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("TASK 11 - CLASS BALANCE")
print("=" * 70)

display(
    class_balance.round(2)
)


# ============================================================
# TASK 11A
# BASELINE RANDOM FOREST
# ============================================================

baseline_rf = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

baseline_rf.fit(X_train,y_train)

baseline_pred = baseline_rf.predict(X_test)
baseline_precision = precision_score(y_test,baseline_pred)
baseline_recall = recall_score(y_test,baseline_pred)
baseline_f1 = f1_score(y_test,baseline_pred)


# ============================================================
# TASK 11B
# RANDOM FOREST WITH CLASS_WEIGHT = BALANCED
# ============================================================

balanced_rf = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_rf.fit(X_train,y_train)

balanced_pred =balanced_rf.predict(X_test)
balanced_precision = precision_score(y_test,balanced_pred)
balanced_recall = recall_score(y_test,balanced_pred)
balanced_f1 = f1_score(y_test,balanced_pred)


# ============================================================
# TASK 11C
# RANDOM FOREST + SMOTE
# ============================================================

# IMPORTANT:
#
# SMOTE is placed inside the imbalanced-learn Pipeline.
# Therefore SMOTE is fitted/applied only to the training
# portion during .fit().
#
# The test set is NEVER oversampled.

smote_rf = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)



smote_rf.fit(X_train,y_train)

smote_pred =smote_rf.predict(X_test)
smote_precision = precision_score(y_test,smote_pred)
smote_recall = recall_score(y_test,smote_pred)
smote_f1 = f1_score(y_test,smote_pred)

# ============================================================
# TASK 11
# IMBALANCE COMPARISON
# ============================================================

imbalance_results = pd.DataFrame({
    "Strategy": ["Baseline","Class Weight = Balanced","SMOTE"],
    "Precision": [baseline_precision,balanced_precision,smote_precision],
    "Recall": [baseline_recall,balanced_recall,smote_recall],
    "F1": [baseline_f1,balanced_f1,smote_f1]
})

print("\n" + "=" * 70)
print("TASK 11 - IMBALANCE HANDLING COMPARISON")
print("=" * 70)

display(
    imbalance_results.round(4)
)


# ------------------------------------------------------------
# Identify strategy with highest F1
# ------------------------------------------------------------

best_imbalance_row = imbalance_results.loc[imbalance_results["F1"].idxmax()]

best_imbalance_strategy = (best_imbalance_row["Strategy"])

best_imbalance_f1 = (best_imbalance_row["F1"])

print(
    f"\nThe strategy with the highest F1 score in this "
    f"test-set comparison was: "
    f"{best_imbalance_strategy}"
)

print(
    f"F1 score: {best_imbalance_f1:.4f}"
)

print(
    "\nInterpretation: F1 balances precision and recall. "
    "The strategy with the highest F1 provided the strongest "
    "balance between these two measures for this comparison."
)


# ============================================================
# TASK 12
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

rf_for_grid = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {"classifier__n_estimators": [100,200,300],
    "classifier__max_depth": [None,5,10,15],
    "classifier__max_features": ["sqrt","log2"]
}


grid_search = GridSearchCV(
    estimator=rf_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


print("\n" + "=" * 70)
print("TASK 12 - RANDOM FOREST GRID SEARCH")
print("=" * 70)

grid_search.fit(X_train,y_train)


# ------------------------------------------------------------
# Best parameters
# ------------------------------------------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:",round(grid_search.best_score_, 4))


# ------------------------------------------------------------
# OOB score
# ------------------------------------------------------------

best_rf_pipeline = (grid_search.best_estimator_)
best_rf_model = (best_rf_pipeline.named_steps["classifier"])
print("\nBest Random Forest OOB score:",round(best_rf_model.oob_score_, 4))


# ============================================================
# TASK 12
# EVALUATE TUNED RANDOM FOREST
# ============================================================

tuned_rf_pred = (best_rf_pipeline.predict(X_test))

tuned_rf_prob = (best_rf_pipeline.predict_proba(X_test)[:, 1])

tuned_rf_accuracy = accuracy_score(y_test,tuned_rf_pred)
tuned_rf_precision = precision_score(y_test,tuned_rf_pred)
tuned_rf_recall = recall_score(y_test,tuned_rf_pred)
tuned_rf_f1 = f1_score(y_test,tuned_rf_pred)
tuned_rf_auc = roc_auc_score(y_test,tuned_rf_prob)


tuned_rf_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "OOB Score"
    ],

    "Tuned Random Forest": [
        tuned_rf_accuracy,
        tuned_rf_precision,
        tuned_rf_recall,
        tuned_rf_f1,
        tuned_rf_auc,
        best_rf_model.oob_score_
    ]
})


print("\nTuned Random Forest results:")
print(tuned_rf_results.round(4))


# ============================================================
# TASK 13
# REGRESSION: PREDICT FARE
# ============================================================

regression_target = "fare"

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
    "survived"
]


X_reg = df[regression_features]
y_reg = df[regression_target]


# ------------------------------------------------------------
# Regression train/test split
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (train_test_split(X_reg,y_reg,test_size=0.20,random_state=42))

print("\n" + "=" * 70)
print("TASK 13 - REGRESSION")
print("=" * 70)

print("Regression training shape:",X_reg_train.shape)
print("Regression testing shape:",X_reg_test.shape)


# ============================================================
# REGRESSION PREPROCESSING
# ============================================================

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived"
]

reg_categorical_features = [
    "sex",
    "embarked"
]


reg_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


reg_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "cat",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)


# ============================================================
# MULTIVARIATE LINEAR REGRESSION
# ============================================================

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(X_reg_train,y_reg_train)
fare_predictions = (regression_pipeline.predict(X_reg_test))
print("\nLinear regression trained successfully.")


# ============================================================
# TASK 13
# REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(y_reg_test,fare_predictions)
rmse = np.sqrt(mean_squared_error(y_reg_test,fare_predictions))
r2 = r2_score(y_reg_test,fare_predictions)


# ------------------------------------------------------------
# Calculate adjusted R²
# ------------------------------------------------------------

X_reg_test_transformed = (regression_pipeline.named_steps["preprocessor"].transform(X_reg_test))

n = X_reg_test_transformed.shape[0]

p = X_reg_test_transformed.shape[1]


adjusted_r2 = (1-((1 - r2) * (n - 1)/(n - p - 1)))

regression_results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],

    "Multivariate Linear Regression": [
        mae,
        rmse,
        r2,
        adjusted_r2
    ]
})


print("\nRegression metrics:")
print(regression_results.round(4))


# ============================================================
# TASK 13
# RESIDUAL PLOT
# ============================================================

residuals = (y_reg_test-fare_predictions)
plt.figure(figsize=(9, 6))
plt.scatter(fare_predictions,residuals,alpha=0.6)
plt.axhline(y=0,linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")
plt.grid()
filename="residualscatter.png"
plt.savefig(charts_path/filename)
plt.show()


# ------------------------------------------------------------
# Simple automatic indication
# ------------------------------------------------------------

# Calculate residual spread in bins of predicted values.
# This is only a descriptive check. The plot should also
# be inspected visually.

residual_check = pd.DataFrame({
    "Predicted": fare_predictions,
    "Residual": residuals
})

residual_check["Prediction_Bin"] = pd.qcut(residual_check["Predicted"],q=4,duplicates="drop")

residual_spread = (residual_check.groupby("Prediction_Bin")["Residual"].std())

print("\nResidual standard deviation by predicted-fare range:")
print(residual_spread.round(3))

print("\nResidual plot interpretation:"
    "\nInspect the residual plot for a funnel-shaped pattern. "
    "\nA widening or narrowing spread of residuals as predicted "
    "\nfare increases would indicate heteroscedasticity. "
    "\nA relatively constant random spread around zero suggests "
    "\nless evidence of heteroscedasticity."
)


# ============================================================
# TASK 14
# FINAL CLASSIFICATION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TASK 14 - CLASSIFICATION METRICS")
print("=" * 70)

print(classification_comparison.round(4))


# ============================================================
# TASK 14
# REGRESSION METRICS
# ============================================================

print("\n" + "=" * 70)
print("TASK 14 - REGRESSION METRICS")
print("=" * 70)
print(regression_results.round(4))


# ------------------------------------------------------------
# Important distinction
# ------------------------------------------------------------

print("\nNOTE:"
    "\nClassification metrics and regression metrics are "
    "\ndifferent metric groups and should not be directly "
    "\ncompared numerically. Classification uses Accuracy, "
    "\nPrecision, Recall, F1 and AUC, while regression uses "
    "\nMAE, RMSE, R² and Adjusted R²."
)


# ============================================================
# TASK 14
# DATA-DRIVEN CLASSIFIER INFORMATION FOR REPORT
# ============================================================

best_classifier_row = (classification_comparison.loc[classification_comparison["F1"].idxmax()])
best_classifier_name = (best_classifier_row["Model"])
best_classifier_accuracy = (best_classifier_row["Accuracy"])
best_classifier_precision = (best_classifier_row["Precision"])
best_classifier_recall = (best_classifier_row["Recall"])
best_classifier_f1 = (best_classifier_row["F1"])
best_classifier_auc = (best_classifier_row["AUC"])


print("\n" + "=" * 70)
print("DATA-DRIVEN CLASSIFIER SUMMARY")
print("=" * 70)

print( f"Highest F1 classifier:{best_classifier_name}")
print( f"Accuracy             :{best_classifier_accuracy}")
print( f"Precision            :{best_classifier_precision}")
print( f"Recall               :{best_classifier_recall}")
print( f"F1                   :{best_classifier_f1}")
print( f"AUC                  :{best_classifier_auc}")

# ============================================================
# TASK 15
# SAVE COMPLETE BEST PIPELINE
# ============================================================

# The tuned Random Forest pipeline contains:
#
# 1. Missing-value imputation
# 2. One-hot encoding
# 3. StandardScaler
# 4. Random Forest classifier
#
# Therefore it can receive RAW data directly.

full_pipeline = best_rf_pipeline


pipeline_filename = ("titanic_best_pipeline.joblib")


joblib.dump(full_pipeline,pipeline_filename)


print("\n" + "=" * 70)
print("TASK 15 - PIPELINE SAVED")
print("=" * 70)

print(
    f"Saved complete pipeline as: "
    f"{pipeline_filename}"
)


# ============================================================
# TASK 15
# RELOAD COMPLETE PIPELINE
# ============================================================

loaded_pipeline = joblib.load(pipeline_filename)
print("\nComplete pipeline successfully reloaded.")


# ============================================================
# TASK 15
# PREDICT USING RAW INPUT
# ============================================================

# No manual preprocessing is performed here.
#
# The input is raw Titanic data.
# The saved pipeline performs:
#
# raw data
#    ↓
# imputation
#    ↓
# encoding
#    ↓
# scaling
#    ↓
# Random Forest prediction


raw_sample = X_test.iloc[[0]]
loaded_prediction = (loaded_pipeline.predict(raw_sample))
loaded_probability = (loaded_pipeline.predict_proba(raw_sample)[:, 1])
print("\nRaw input:")
print(raw_sample)
print("Predicted class:",loaded_prediction[0])
print("Predicted probability of survival:",round(loaded_probability[0],4))
print("Actual class:",y_test.iloc[0])


# ============================================================
# TASK 15
# VERIFY MULTIPLE RAW PREDICTIONS
# ============================================================

raw_samples = X_test.head(5)


loaded_predictions = (loaded_pipeline.predict(raw_samples))
verification = raw_samples.copy()
verification["Actual"] = (y_test.loc[raw_samples.index])
verification["Predicted"] = (loaded_predictions)
print("\n" + "=" * 70)
print("PIPELINE RAW-INPUT VERIFICATION")
print("=" * 70)
print(verification)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY - TASKS 7 TO 15")
print("=" * 70)
print("\n1. Train/test split: 80/20 stratified split")
print("2. Preprocessing: median numeric imputation, "
    "most-frequent categorical imputation, "
    "\none-hot encoding and StandardScaler")
print("3. Classifiers: Logistic Regression, "
    "Decision Tree and Random Forest"
)

print(f"4. Highest-F1 classifier in this run:{best_classifier_name}")
print(f"5. Highest-F1 classifier F1         :{best_classifier_f1:.4f}")
print(f"6. Highest-F1 classifier AUC        :{best_classifier_auc:.4f}")
print(f"7. Best imbalance strategy by F1    :{best_imbalance_strategy}")
print(f"8. Best imbalance F1                :{best_imbalance_f1:.4f}")
print(f"9. Tuned Random Forest OOB score    :{best_rf_model.oob_score_:.4f}")
print(f"10. Regression MAE                  :{mae:.4f}")
print(f"11. Regression RMSE                 :{rmse:.4f}")
print(f"12. Regression R²                   :{r2:.4f}")
print(f"13. Regression Adjusted R²          :{adjusted_r2:.4f}")
print(f"14. Complete pipeline saved as      :{pipeline_filename}")
print("\nAll Tasks 7-15 completed.")