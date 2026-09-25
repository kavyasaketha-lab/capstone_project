#### Part A : EDA
# Titanic Dataset Analysis
## Overview
This project performs data loading, cleaning, exploratory data analysis (EDA), outlier detection, correlation analysis, and standardization on the **Titanic dataset** using Python.

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Workflow

The project follows these main steps:

1. Load the Titanic dataset using Seaborn.
2. Save the original dataset as `titanic.csv`.
3. Profile the dataset using shape, information, and descriptive statistics.
4. Identify and report missing values.
5. Handle missing values:
   - Drop `deck` due to high missingness.
   - Fill missing `age` values using the median.
   - Fill missing `embarked` values using the mode.
   - Remove rows with missing `embark_town`.
6. Save the cleaned dataset.
7. Analyse age and fare distributions using histograms and box plots.
8. Detect potential outliers using the IQR method.
9. Analyse fare using mean, median, mode, and skewness.
10. Analyse survival rates by sex and passenger class.
11. Create scatter plots and a correlation heatmap.
12. Identify the two strongest numerical correlations.
13. Standardize `age` and `fare` using `StandardScaler`.

## Output

The project generates:

- `titanic.csv` - cleaned Titanic dataset
- `PlotCharts/` - folder containing generated visualizations:
  - Age and Fare Histograms
  - Age and Fare Box Plots
  - Survival Rate by Sex and Passenger Class
  - Age vs Fare Scatter Plot
  - Correlation Heatmap

## How to Run

Install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

Then run the Python script:

```bash
python <script_name>.py
```

The cleaned dataset and charts will be created automatically in the project directory.


# Chart Analysis

## 1. Age and Fare Histograms
The histograms show the distribution of passenger age and fare.
The age histogram helps identify the most common passenger age ranges and the overall age distribution.
The fare histogram shows how ticket prices are distributed.
The fare distribution is typically right-skewed, meaning most passengers paid lower fares while a smaller number paid substantially higher fares.

## 2. Age and Fare Box Plots
Box plots are used to identify the spread of the data and potential outliers.
The age box plot shows the median age, quartile range, and potential unusually high or low ages.
The fare box plot helps identify passengers with unusually expensive tickets.
The project also calculates IQR-based outlier counts for both variables.

## 3. Survival Rate by Sex and Passenger Class
This bar chart compares survival rates across passenger classes and sex.
It helps identify differences in survival rates between male and female passengers.
It also shows how survival varied across first, second, and third class.
The chart allows the relationship between sex, passenger class, and survival to be compared visually.

## 4. Age, Fare and Survival Scatter Plot
The scatter plot compares age with fare, while using survival status to distinguish passengers.
It helps investigate whether there is any visible relationship between:
Passenger age
Ticket fare
Survival outcome
The plot can reveal clusters or patterns, although a visual relationship does not necessarily indicate causation.

## 5. Correlation Heatmap

The correlation heatmap displays the relationships between numerical variables such as:
survived
pclass
age
sibsp
parch
fare

Correlation values range from -1 to +1:
Values close to +1 indicate a strong positive relationship.
Values close to -1 indicate a strong negative relationship.
Values close to 0 indicate a weak or limited linear relationship.
The code also identifies the two strongest correlations based on the absolute correlation value.

## 6. Standardization Analysis
The project standardizes age and fare using StandardScaler.
Before standardization, the variables have different scales. After standardization:
The mean should be approximately 0.
The standard deviation should be approximately 1.
This demonstrates how numerical features can be transformed to a common scale before being used in machine-learning models.
Key Findings
The analysis provides insights into:
Passenger age and fare distributions.
Potential outliers in age and fare.
Differences in survival rates by sex and passenger class.
Relationships between numerical passenger features.
The effect of standardization on numerical variables.
The charts are saved automatically in the PlotCharts folder for further review.


# Part B : Regression ,Decision Tree and Random Forest# Titanic Machine Learning Analysis

## 1. Project Overview

This project applies machine-learning techniques to the cleaned **Titanic dataset** to:

- Predict passenger survival using classification models.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Handle class imbalance using different strategies.
- Tune the Random Forest model using GridSearchCV.
- Predict passenger fare using multivariate linear regression.
- Evaluate model performance using appropriate classification and regression metrics.
- Save and reload the complete best-performing pipeline for future predictions.

---

## 2. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib


# 3. Step-by-Step Process

## Step 1: Import Libraries

The required Python libraries are imported for:

- Data handling: `Pandas`, `NumPy`
- Visualisation: `Matplotlib`
- Machine learning: `Scikit-learn`
- Class imbalance: `SMOTE` from `imbalanced-learn`
- Model saving/loading: `Joblib`

The code also creates a `ModelCharts` folder for storing generated visualisations.


## Step 2: Load and Validate the Dataset

The cleaned `titanic.csv` file is loaded into a Pandas DataFrame.

The code displays:

- Dataset shape
- Column names
- First five rows

It also checks that all required columns are available before continuing. 



## Step 3: Train/Test Split

`survived` is used as the classification target.
The data is divided into:
- **80% training data**
- **20% testing data**
A **stratified split** is used so that the proportion of survivors and non-survivors remains approximately the same in both datasets.
This is important because the target classes are not perfectly balanced.


## Step 4: Data Preprocessing

The features are divided into numerical and categorical variables.

### Numerical Features

- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

Processing:

1. Missing values are replaced using the **median**.
2. Features are standardised using `StandardScaler`.

### Categorical Features

- `sex`
- `embarked`

Processing:

1. Missing values are replaced using the **most frequent value**.
2. Categorical variables are converted using **One-Hot Encoding**.
3. Unknown categories are safely ignored.

All preprocessing is included inside the pipeline and fitted using the training data. 

#### Part A : EDA
# Titanic Dataset Analysis
## Overview
This project performs data loading, cleaning, exploratory data analysis (EDA), outlier detection, correlation analysis, and standardization on the **Titanic dataset** using Python.

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Workflow

The project follows these main steps:

1. Load the Titanic dataset using Seaborn.
2. Save the original dataset as `titanic.csv`.
3. Profile the dataset using shape, information, and descriptive statistics.
4. Identify and report missing values.
5. Handle missing values:
   - Drop `deck` due to high missingness.
   - Fill missing `age` values using the median.
   - Fill missing `embarked` values using the mode.
   - Remove rows with missing `embark_town`.
6. Save the cleaned dataset.
7. Analyse age and fare distributions using histograms and box plots.
8. Detect potential outliers using the IQR method.
9. Analyse fare using mean, median, mode, and skewness.
10. Analyse survival rates by sex and passenger class.
11. Create scatter plots and a correlation heatmap.
12. Identify the two strongest numerical correlations.
13. Standardize `age` and `fare` using `StandardScaler`.

## Output

The project generates:

- `titanic.csv` - cleaned Titanic dataset
- `PlotCharts/` - folder containing generated visualizations:
  - Age and Fare Histograms
  - Age and Fare Box Plots
  - Survival Rate by Sex and Passenger Class
  - Age vs Fare Scatter Plot
  - Correlation Heatmap

## How to Run

Install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

Then run the Python script:

```bash
python <script_name>.py
```

The cleaned dataset and charts will be created automatically in the project directory.


# Chart Analysis

## 1. Age and Fare Histograms
The histograms show the distribution of passenger age and fare.
The age histogram helps identify the most common passenger age ranges and the overall age distribution.
The fare histogram shows how ticket prices are distributed.
The fare distribution is typically right-skewed, meaning most passengers paid lower fares while a smaller number paid substantially higher fares.

## 2. Age and Fare Box Plots
Box plots are used to identify the spread of the data and potential outliers.
The age box plot shows the median age, quartile range, and potential unusually high or low ages.
The fare box plot helps identify passengers with unusually expensive tickets.
The project also calculates IQR-based outlier counts for both variables.

## 3. Survival Rate by Sex and Passenger Class
This bar chart compares survival rates across passenger classes and sex.
It helps identify differences in survival rates between male and female passengers.
It also shows how survival varied across first, second, and third class.
The chart allows the relationship between sex, passenger class, and survival to be compared visually.

## 4. Age, Fare and Survival Scatter Plot
The scatter plot compares age with fare, while using survival status to distinguish passengers.
It helps investigate whether there is any visible relationship between:
Passenger age
Ticket fare
Survival outcome
The plot can reveal clusters or patterns, although a visual relationship does not necessarily indicate causation.

## 5. Correlation Heatmap

The correlation heatmap displays the relationships between numerical variables such as:
survived
pclass
age
sibsp
parch
fare

Correlation values range from -1 to +1:
Values close to +1 indicate a strong positive relationship.
Values close to -1 indicate a strong negative relationship.
Values close to 0 indicate a weak or limited linear relationship.
The code also identifies the two strongest correlations based on the absolute correlation value.

## 6. Standardization Analysis
The project standardizes age and fare using StandardScaler.
Before standardization, the variables have different scales. After standardization:
The mean should be approximately 0.
The standard deviation should be approximately 1.
This demonstrates how numerical features can be transformed to a common scale before being used in machine-learning models.
Key Findings
The analysis provides insights into:
Passenger age and fare distributions.
Potential outliers in age and fare.
Differences in survival rates by sex and passenger class.
Relationships between numerical passenger features.
The effect of standardization on numerical variables.
The charts are saved automatically in the PlotCharts folder for further review.



# Part B : Regression ,Decision Tree and Random Forest# Titanic Machine Learning Analysis

## 1. Project Overview

This project applies machine-learning techniques to the cleaned **Titanic dataset** to:

- Predict passenger survival using classification models.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Handle class imbalance using different strategies.
- Tune the Random Forest model using GridSearchCV.
- Predict passenger fare using multivariate linear regression.
- Evaluate model performance using appropriate classification and regression metrics.
- Save and reload the complete best-performing pipeline for future predictions.

---

## 2. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib

---

# 3. Step-by-Step Process

## Step 1: Import Libraries

The required Python libraries are imported for:

- Data handling: `Pandas`, `NumPy`
- Visualisation: `Matplotlib`
- Machine learning: `Scikit-learn`
- Class imbalance: `SMOTE` from `imbalanced-learn`
- Model saving/loading: `Joblib`

The code also creates a `ModelCharts` folder for storing generated visualisations.

---

## Step 2: Load and Validate the Dataset

The cleaned `titanic.csv` file is loaded into a Pandas DataFrame.

The code displays:

- Dataset shape
- Column names
- First five rows

It also checks that all required columns are available before continuing. 

---

## Step 3: Train/Test Split

`survived` is used as the classification target.

The data is divided into:

- **80% training data**
- **20% testing data**

A **stratified split** is used so that the proportion of survivors and non-survivors remains approximately the same in both datasets.

This is important because the target classes are not perfectly balanced.

---

## Step 4: Data Preprocessing

The features are divided into numerical and categorical variables.

### Numerical Features

- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

Processing:

1. Missing values are replaced using the **median**.
2. Features are standardised using `StandardScaler`.

### Categorical Features

- `sex`
- `embarked`

Processing:

1. Missing values are replaced using the **most frequent value**.
2. Categorical variables are converted using **One-Hot Encoding**.
3. Unknown categories are safely ignored.

All preprocessing is included inside the pipeline and fitted using the training data. 

---

# 5. Classification Models

Three classification models are trained using the same preprocessing approach.
### Logistic Regression
Used as a baseline linear classification model.
### Decision Tree
A Decision Tree classifier is trained with `max_depth=5`.
A tree visualisation is also generated and saved as:
`ModelCharts/Decisiontree.png`
### Random Forest
A Random Forest classifier is trained using:
- `n_estimators = 200`
- `random_state = 42`
The three models are trained and tested using the same train/test split.

# 6. Classification Evaluation

The classification models are evaluated using:
- **Accuracy** – overall proportion of correct predictions.
- **Precision** – proportion of predicted survivors that actually survived.
- **Recall** – proportion of actual survivors correctly identified.
- **F1 Score** – balance between precision and recall.
- **AUC** – measures the model's ability to distinguish between the two classes.

A comparison table is produced for all three classifiers.


# 7. Confusion Matrix and ROC Analysis
### Confusion Matrix
Confusion matrices are generated for:
- Logistic Regression
- Decision Tree
- Random Forest

The output is saved as:
`ModelCharts/ComparingConfusionMatrix.png`

The matrices show correct and incorrect predictions for both survival classes.
### ROC Curve
ROC curves are generated for all three models and compared using AUC.
The output is saved as:
`ModelCharts/aucroc_compare.png`
A model with a higher AUC demonstrates better separation between the two target classes in this test-set evaluation.



# 8. Class Imbalance Analysis
The project compares three approaches for handling the imbalance between survivors and non-survivors.
### 1. Baseline Random Forest
Uses the original class distribution.
### 2. Class Weight = Balanced
Uses:
```python
class_weight="balanced"
```
This gives greater importance to the minority class during model training.
### 3. SMOTE
Synthetic Minority Over-sampling Technique is applied to the training data.
SMOTE is placed inside an `imbalanced-learn` pipeline, ensuring that the **test set is not oversampled**.
The three approaches are compared using:
- Precision
- Recall
- F1 Score
The strategy with the highest F1 score is identified automatically.

# 9. Random Forest Hyperparameter Tuning
`GridSearchCV` is used to find better Random Forest parameters.
The parameters tested are:
- `n_estimators`: 100, 200, 300
- `max_depth`: None, 5, 10, 15
- `max_features`: `sqrt`, `log2`
The search uses:
- **5-fold cross-validation**
- **F1 score** as the optimisation metric
The best parameters and best cross-validation F1 score are reported.
The Random Forest **OOB score** is also recorded.

# 10. Tuned Random Forest Evaluation
The tuned Random Forest is evaluated on the previously held-out test set using:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- OOB Score
This provides a final evaluation of the tuned classification model.


# 11. Fare Prediction Using Linear Regression
The project also performs a regression task where:
**Target:** `fare`
Features used:
- `pclass`
- `sex`
- `age`
- `sibsp`
- `parch`
- `embarked`
- `survived`

The data is split into training and testing sets using an 80/20 split.
A preprocessing pipeline is then created using:
- Median imputation
- Most-frequent categorical imputation
- One-hot encoding
- StandardScaler
A **Multivariate Linear Regression** model is trained using the processed features.

# 12. Regression Evaluation
The fare prediction model is evaluated using:
- **MAE** – average absolute prediction error.
- **RMSE** – penalises larger prediction errors more strongly.
- **R²** – proportion of variance explained by the model.
- **Adjusted R²** – adjusts R² based on the number of predictors.
The results are displayed in a regression metrics table.

# 13. Residual Analysis
A residual plot is created to evaluate the regression model.
The plot compares:
- Predicted fare
- Residuals
The output is saved as:
`ModelCharts/residualscatter.png`
The residuals are also grouped into prediction ranges to examine whether their spread changes as predicted fare increases.
A funnel-shaped pattern may indicate heteroscedasticity, while a relatively random spread around zero provides less evidence of heteroscedasticity.

# 14. Final Model Comparison
The final output reports:
### Classification
- Accuracy
- Precision
- Recall
- F1
- AUC
### Regression
- MAE
- RMSE
- R²
- Adjusted R²

The code correctly treats classification and regression metrics as separate metric groups rather than directly comparing their numerical values.
The classifier with the highest F1 score is also identified automatically from the classification results.


# 15. Save and Reload the Complete Pipeline
The tuned Random Forest pipeline is saved using Joblib as:
```text
titanic_best_pipeline.joblib
```
The saved pipeline contains:
1. Missing-value imputation
2. One-hot encoding
3. StandardScaler
4. Random Forest classifier
Therefore, new raw Titanic records can be passed directly to the saved pipeline without manually repeating the preprocessing steps.
The pipeline is then reloaded and tested using raw test records. The code verifies:
- Actual class
- Predicted class
- Predicted survival probability
It also performs verification on multiple raw records.

# 16. Project Outputs
The project generates:

```text
ModelCharts/
│
├── Decisiontree.png
├── ComparingConfusionMatrix.png
├── aucroc_compare.png
└── residualscatter.png

titanic_best_pipeline.joblib
```
The console output also provides the model comparison, imbalance analysis, tuned Random Forest results, regression metrics, and final summary.

---

# 17. Final Workflow Summary

```text
Cleaned Titanic Dataset
        ↓
Required Column Validation
        ↓
80/20 Stratified Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
 ┌───────────────────────────────┐
 │ Logistic Regression           │
 │ Decision Tree                 │
 │ Random Forest                 │
 └───────────────────────────────┘
        ↓
Classification Evaluation
        ↓
Confusion Matrix + ROC/AUC
        ↓
Class Imbalance Comparison
        ↓
Random Forest GridSearchCV
        ↓
Tuned Random Forest
        ↓
Final Classification Results
        ↓
Fare Regression
        ↓
MAE / RMSE / R² / Adjusted R²
        ↓
Residual Analysis
        ↓
Save Best Pipeline
        ↓
Reload Pipeline
        ↓
Prediction Using Raw Input
```

## Conclusion

This project demonstrates an end-to-end machine-learning workflow using the Titanic dataset, including **data splitting, preprocessing, classification, model evaluation, imbalance handling, hyperparameter tuning, regression, residual analysis, and model deployment through a saved pipeline**.

All major outputs and model performance metrics are generated automatically when the script is executed.


