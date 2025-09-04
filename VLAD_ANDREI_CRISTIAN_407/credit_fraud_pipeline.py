import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import math

import numpy as np
from scipy.stats import ks_2samp

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mishra5001/credit-card")

print("Path to dataset files:", path)

files = os.listdir(path)
print("Files:", files)

# Read all CSV files in the directory into a dictionary of DataFrames
dfs = {}
for file in files:
    if file.endswith(".csv"):
        file_path = os.path.join(path, file)
        dfs[file] = pd.read_csv(file_path, encoding='cp1252')

# Access individual DataFrames using their filenames as keys
application_data_df = dfs['application_data.csv']
columns_description_df = dfs['columns_description.csv']
previous_application_df = dfs['previous_application.csv']

application_data_df.head()

# Display the columns description and show full description
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):  # more options can be specified also
    display(columns_description_df)

# Bronze Layer 
  
df_application_bronze = application_data_df.copy()
df_previous_application_bronze = previous_application_df.copy()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

df_application_bronze.head()

df_previous_application_bronze.head()

# Check for missing values in application_data
print("Missing values in application table:")
print(df_application_bronze.isnull().sum())
print("\nPercentage of missing values:")
print((df_application_bronze.isnull().sum() / len(df_application_bronze)) * 100)

# Check for missing values in previous_application
print("Missing values in previous application table:")
print(df_previous_application_bronze.isnull().sum())
print("\nPercentage of missing values:")
print((df_previous_application_bronze.isnull().sum() / len(df_previous_application_bronze)) * 100)

def plot_feature_distributions(df, features, target_col="TARGET", bins=100, cols=2):
    """
    Plots distribution histograms for multiple features split by target classes.

    Parameters:
    - df : pd.DataFrame
    - features : list of str -> columns to plot
    - target_col : str -> name of target column (default: 'TARGET')
    - bins : int -> number of bins for histogram
    - cols : int -> number of columns in subplot grid
    """
    n_features = len(features)
    rows = math.ceil(n_features / cols)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(df.loc[df[target_col] == 0, feature],
                     bins=bins, color="blue", label="Normal",
                     stat="density", alpha=0.5, ax=ax)
        sns.histplot(df.loc[df[target_col] == 1, feature],
                     bins=bins, color="red", label="Anomaly",
                     stat="density", alpha=0.5, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.legend()

    # Remove unused subplots if features < rows*cols
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
y = df_application_bronze['TARGET']
# Select numerical features excluding those starting with 'FLAG_DOCUMENT'
features = [col for col in df_application_bronze.columns if not col.startswith("FLAG_DOCUMENT")]

plot_feature_distributions(df_application_bronze, features, target_col="TARGET", bins=50, cols=3)

# Get categorical features
cat_features = [col for col in df_application_bronze.select_dtypes(include=['object','category']).columns
                if not col.startswith("FLAG_DOCUMENT")]

plot_feature_distributions(df_application_bronze, cat_features, target_col="TARGET", bins=50, cols=3)

#Silver Layer

# Silver Layer: Feature Engineering and Enrichment
print("\nSilver Layer: Creating New Features")

# Start with a copy of the main application data
df_silver = df_application_bronze.copy()

# 1. Feature Engineering on Numerical Columns
print("  - Engineering numerical features...")
# Convert days to years (absolute values for clarity)
df_silver['AGE_YEARS'] = abs(df_silver['DAYS_BIRTH']) / 365
df_silver['EMPLOYMENT_YEARS'] = abs(df_silver['DAYS_EMPLOYED']) / 365

# Create ratio features
df_silver['ANNUITY_INCOME_RATIO'] = df_silver['AMT_ANNUITY'] / df_silver['AMT_INCOME_TOTAL']
df_silver['CREDIT_INCOME_RATIO'] = df_silver['AMT_CREDIT'] / df_silver['AMT_INCOME_TOTAL']
df_silver['CREDIT_ANNUITY_RATIO'] = df_silver['AMT_CREDIT'] / df_silver['AMT_ANNUITY']
df_silver['EMPLOYMENT_AGE_RATIO'] = df_silver['EMPLOYMENT_YEARS'] / df_silver['AGE_YEARS']
df_silver['EMPLOYMENT_EXPERIENCE_RATIO'] = df_silver['DAYS_EMPLOYED'] / df_silver['DAYS_BIRTH']

# 2. Aggregation from previous applications
print("  - Aggregating features from previous applications...")
# Group by client ID and aggregate numerical features
prev_app_agg = df_previous_application_bronze.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT': ['mean', 'sum'],
    'AMT_ANNUITY': ['mean'],
    'DAYS_DECISION': ['max']
})

# Rename columns for clarity
prev_app_agg.columns = ['_'.join(col).strip() for col in prev_app_agg.columns.values]
prev_app_agg = prev_app_agg.reset_index()

# Merge the aggregated features with the main dataframe
df_silver = pd.merge(df_silver, prev_app_agg, on='SK_ID_CURR', how='left')

print(df_silver[['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'ANNUITY_INCOME_RATIO', 'CREDIT_INCOME_RATIO', 'CREDIT_ANNUITY_RATIO','EMPLOYMENT_AGE_RATIO', 'EMPLOYMENT_EXPERIENCE_RATIO', 'AMT_CREDIT_mean', 'AMT_CREDIT_sum']].head())

# Verify ratio for frauds
raw_ratio_features = [
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "EMPLOYMENT_YEARS",
    "AGE_YEARS",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH"
]

plot_feature_distributions(df_silver, raw_ratio_features)

# Make a silver dataframe with mean of flag_document columns and target

df_flag_mean = pd.DataFrame()

df_flag_mean["FLAG_DOCUMENT_MEAN"] = df_silver.filter(like='FLAG_DOCUMENT').mean(axis=1)
df_flag_mean["TARGET"] = df_silver["TARGET"]

df_flag_mean.head()


# Model Training and Evaluation
flag_mean_ratio_column = ['FLAG_DOCUMENT_MEAN']

plot_feature_distributions(df_flag_mean, flag_mean_ratio_column, target_col="TARGET", bins=50, cols=1)

# Verify ratio for frauds
ratio_features = [
    'ANNUITY_INCOME_RATIO',
    'CREDIT_INCOME_RATIO',
    'CREDIT_ANNUITY_RATIO',
    'EMPLOYMENT_AGE_RATIO',
    'EMPLOYMENT_EXPERIENCE_RATIO'
]

plot_feature_distributions(df_silver, ratio_features)

# Statistical test for CREDIT_INCOME_RATIO
stat, p = ks_2samp(df_silver.loc[y==0, "CREDIT_INCOME_RATIO"],
                   df_silver.loc[y==1, "CREDIT_INCOME_RATIO"])
print(f"KS p-value = {p}")

# A p-value < 0.05 indicates a significant difference in distributions
df_ext_sources = df_silver[
    [
    # "SK_ID_PREV",
    # "SK_ID_CURR",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "TARGET"
    ]
].copy()

df_ext_sources.head()

# Calculate global fraud rate
global_rate = df_ext_sources["TARGET"].mean()

# Engineer features from EXT_SOURCE_1
df_ext_sources_transformed = pd.DataFrame()

df_ext_sources["EXT_SOURCE_1_LOG"] = np.log1p(df_ext_sources["EXT_SOURCE_1"]) # log(1+x) to handle zero values
df_ext_sources["EXT_SOURCE_1_SQRT"] = np.sqrt(df_ext_sources["EXT_SOURCE_1"]) # square root transformation
df_ext_sources["EXT_SOURCE_1_SQUARE"] = df_ext_sources["EXT_SOURCE_1"] ** 2 # square transformation
df_ext_sources["EXT_SOURCE_1_BINNED"] = pd.qcut(df_ext_sources["EXT_SOURCE_1"], q=10, labels=False) # quantile-based binning

bins = np.linspace(0, 1, 20)  # split into 20 bins
df_ext_sources["EXT_SOURCE_1_BIN"] = np.digitize(df_ext_sources["EXT_SOURCE_1"], bins) # binning

risk_map = df_ext_sources.groupby("EXT_SOURCE_1_BIN")["TARGET"].mean() / global_rate # risk mapping
df_ext_sources["EXT_SOURCE_1_RISK"] = df_ext_sources["EXT_SOURCE_1_BIN"].map(risk_map) # map risk to bins

df_ext_sources["EXT_SOURCE_2_LOG"] = np.log1p(df_ext_sources["EXT_SOURCE_2"])
df_ext_sources["EXT_SOURCE_2_SQRT"] = np.sqrt(df_ext_sources["EXT_SOURCE_2"])
df_ext_sources["EXT_SOURCE_2_SQUARE"] = df_ext_sources["EXT_SOURCE_2"] ** 2
df_ext_sources["EXT_SOURCE_2_BINNED"] = pd.qcut(df_ext_sources["EXT_SOURCE_2"], q=10, labels=False)

bins = np.linspace(0, 1, 20)  # split into 20 bins
df_ext_sources["EXT_SOURCE_2_BIN"] = np.digitize(df_ext_sources["EXT_SOURCE_2"], bins)

risk_map = df_ext_sources.groupby("EXT_SOURCE_2_BIN")["TARGET"].mean() / global_rate
df_ext_sources["EXT_SOURCE_2_RISK"] = df_ext_sources["EXT_SOURCE_2_BIN"].map(risk_map)

df_ext_sources["EXT_SOURCE_3_LOG"] = np.log1p(df_ext_sources["EXT_SOURCE_3"])
df_ext_sources["EXT_SOURCE_3_SQRT"] = np.sqrt(df_ext_sources["EXT_SOURCE_3"])
df_ext_sources["EXT_SOURCE_3_SQUARE"] = df_ext_sources["EXT_SOURCE_3"] ** 2
df_ext_sources["EXT_SOURCE_3_BINNED"] = pd.qcut(df_ext_sources["EXT_SOURCE_3"], q=10, labels=False)

bins = np.linspace(0, 1, 20)  # split into 20 bins
df_ext_sources["EXT_SOURCE_3_BIN"] = np.digitize(df_ext_sources["EXT_SOURCE_3"], bins)

risk_map = df_ext_sources.groupby("EXT_SOURCE_3_BIN")["TARGET"].mean() / global_rate
df_ext_sources["EXT_SOURCE_3_RISK"] = df_ext_sources["EXT_SOURCE_3_BIN"].map(risk_map)

df_ext_sources.head()

# Plot distributions of transformed EXT_SOURCE features
ext_source_features = [
    'EXT_SOURCE_1_LOG',
    'EXT_SOURCE_1_SQRT',
    'EXT_SOURCE_1_SQUARE',
    'EXT_SOURCE_1_BINNED',
    'EXT_SOURCE_1_BIN',
    'EXT_SOURCE_1_RISK',
    'EXT_SOURCE_2_LOG',
    'EXT_SOURCE_2_SQRT',
    'EXT_SOURCE_2_SQUARE',
    'EXT_SOURCE_2_BINNED',
    'EXT_SOURCE_2_BIN',
    'EXT_SOURCE_2_RISK',
    'EXT_SOURCE_3_LOG',
    'EXT_SOURCE_3_SQRT',
    'EXT_SOURCE_3_SQUARE',
    'EXT_SOURCE_3_BINNED',
    'EXT_SOURCE_3_BIN',
    'EXT_SOURCE_3_RISK'
]

plot_feature_distributions(df_ext_sources, ext_source_features, target_col="TARGET", bins=100, cols=3)

# Merge engineered EXT_SOURCE features back to silver dataframe
risk_map = df_silver.groupby("ORGANIZATION_TYPE")["TARGET"].mean() / global_rate
df_silver["ORGANIZATION_TYPE_RISK"] = df_silver["ORGANIZATION_TYPE"].map(risk_map)

risk_map = df_silver.groupby("OCCUPATION_TYPE")["TARGET"].mean() / global_rate
df_silver["OCCUPATION_TYPE_RISK"] = df_silver["OCCUPATION_TYPE"].map(risk_map)

# Check Raw Categorical Features
df_silver[["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]].head()

# Check the new risk features
df_silver[["ORGANIZATION_TYPE_RISK", "OCCUPATION_TYPE_RISK"]].head()

categoricals_for_risk = [
    "ORGANIZATION_TYPE_RISK",
    "OCCUPATION_TYPE_RISK"
]
# Plot distributions of categorical risk features
plot_feature_distributions(df_silver, categoricals_for_risk, target_col="TARGET", bins=50, cols=1)

# CREDIT_ANNUITY_RATIO × OCCUPATION_TYPE_RISK

df_silver["CREDIT_ANNUITY_OCCUPATION_RISK"] = df_silver["CREDIT_ANNUITY_RATIO"] * df_silver["OCCUPATION_TYPE_RISK"]
df_silver[["CREDIT_ANNUITY_OCCUPATION_RISK"]].head()

categoricals_for_risk = [
    "CREDIT_ANNUITY_OCCUPATION_RISK"
]

plot_feature_distributions(df_silver, categoricals_for_risk, target_col="TARGET", bins=50, cols=1)


# Bin CREDIT_ANNUITY_OCCUPATION_RISK

bins = np.linspace(df_silver["CREDIT_ANNUITY_OCCUPATION_RISK"].min(), df_silver["CREDIT_ANNUITY_OCCUPATION_RISK"].max(), 35)
df_silver["CREDIT_ANNUITY_OCCUPATION_RISK_BIN"] = np.digitize(df_silver["CREDIT_ANNUITY_OCCUPATION_RISK"], bins)
df_silver[["CREDIT_ANNUITY_OCCUPATION_RISK_BIN"]].head()

categoricals_for_risk = [
    "CREDIT_ANNUITY_OCCUPATION_RISK_BIN"
]

plot_feature_distributions(df_silver, categoricals_for_risk, target_col="TARGET", bins=50, cols=1)

# ORGANIZATION_TYPE_RISK × OCCUPATION_TYPE_RISK

df_silver["ORGANIZATION_OCCUPATION_RISK"] = df_silver["ORGANIZATION_TYPE_RISK"] * df_silver["OCCUPATION_TYPE_RISK"]
df_silver[["ORGANIZATION_OCCUPATION_RISK"]].head()

# Plot distributions of ORGANIZATION_OCCUPATION_RISK
categoricals_for_risk = [
    "ORGANIZATION_OCCUPATION_RISK"
]

plot_feature_distributions(df_silver, categoricals_for_risk, target_col="TARGET", bins=50, cols=1)

# function to evaluate feature goodness -> are they meaningful for anomaly detection?
def evaluate_feature_goodness(df, features, target_col="is_anomaly"):
    results = []

    for feature in features: 
        try:
            # Drop NaN for clean comparison
            data = df[[feature, target_col]].dropna()

            # Separate distributions
            normal = data[data[target_col] == 0][feature]
            anomaly = data[data[target_col] == 1][feature]

            # KS test
            ks_stat, ks_pvalue = ks_2samp(normal, anomaly)

            # AUROC (feature as classifier)
            try:
                auc = roc_auc_score(data[target_col], data[feature])
            except Exception:
                auc = np.nan  # For categorical encodings that aren’t numeric

            results.append({
                "feature": feature,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "auc": auc,
                "normal_mean": normal.mean(),
                "anomaly_mean": anomaly.mean()
            })

        except Exception as e:
            print(f"Skipping {feature} due to error: {e}")

    return pd.DataFrame(results).sort_values(by="auc", ascending=False)


# set pandas to display all rows and full column width
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


features = [
    "CREDIT_ANNUITY_RATIO",
    "AMT_INCOME_TOTAL",
    "CREDIT_ANNUITY_OCCUPATION_RISK",
    "CREDIT_ANNUITY_OCCUPATION_RISK_BIN",
    "ORGANIZATION_TYPE_RISK",
    "OCCUPATION_TYPE_RISK",
    "ORGANIZATION_OCCUPATION_RISK",
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE"
]

results = evaluate_feature_goodness(df_silver, features, target_col="TARGET")
print(results)


ext_src_features = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1_LOG",
    "EXT_SOURCE_1_SQRT",
    "EXT_SOURCE_1_SQUARE",
    "EXT_SOURCE_1_BINNED",
    "EXT_SOURCE_1_BIN",
    "EXT_SOURCE_1_RISK",
    "EXT_SOURCE_2_LOG",
    "EXT_SOURCE_2_SQRT",
    "EXT_SOURCE_2_SQUARE",
    "EXT_SOURCE_2_BINNED",
    "EXT_SOURCE_2_BIN",
    "EXT_SOURCE_2_RISK",
    "EXT_SOURCE_3_LOG",
    "EXT_SOURCE_3_SQRT",
    "EXT_SOURCE_3_SQUARE",
    "EXT_SOURCE_3_BINNED",
    "EXT_SOURCE_3_BIN",
    "EXT_SOURCE_3_RISK"
]

results = evaluate_feature_goodness(df_ext_sources, ext_src_features, target_col="TARGET")
print(results)


# Gold Layer: Final Feature Set and Model Training
df_silver.drop(columns=["ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO", "EMPLOYMENT_AGE_RATIO", "EMPLOYMENT_EXPERIENCE_RATIO" ], inplace=True)
df_silver.head()

df_gold = pd.DataFrame()

# Gold Layer: Final Model-Ready Data
df_gold = df_silver[[
    "ORGANIZATION_OCCUPATION_RISK",
    "OCCUPATION_TYPE_RISK",
    "ORGANIZATION_TYPE_RISK",
    "CREDIT_ANNUITY_OCCUPATION_RISK"
]].copy()

df_gold = df_gold.join(
    df_ext_sources[[
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "EXT_SOURCE_1_RISK",
        "EXT_SOURCE_2_RISK",
        "EXT_SOURCE_3_RISK"
    ]],
    how="left"
)

df_gold["TARGET"] = df_silver["TARGET"]

# 1. Handle Categorical Features
print("\n Gold Layer: Handling Categorical Features ")
# Use one-hot encoding for categorical columns
categorical_cols = [col for col in df_gold.columns if df_gold[col].dtype == "object"]
if categorical_cols:
    df_gold = pd.get_dummies(df_gold, columns=categorical_cols, dummy_na=False)

df_gold.head()

# Select Features and Target
print("  - Selecting features and target...")
# Separate features (X) from the target (y)
X = df_gold.drop(['TARGET'], axis=1) # Exclude TARGET and ID for unsupervised model training
y = df_gold['TARGET'] # Save TARGET for later evaluation

# Add missingness indicators
for col in X.columns:
    if X[col].isna().any():
        X[f"{col}_missing"] = X[col].isna().astype(int)

# Impute missing values and scale features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Gold Layer: Data Ready for Model Training")
print(f"  - Features (X_scaled): {X_scaled.shape}")
print(f"  - Target (y): {y.shape}")


# X_scaled: The standardized feature data
# y: The target variable (TARGET) for evaluation

# Split the data
X_train_scaled, X_test_scaled, _, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
fraud_rate = y.value_counts()[1] / len(y) # Calculate fraud rate in the dataset

print(" Implementing One-Class SVM ")

# The 'nu' parameter is an upper bound on the fraction of training errors
# and a lower bound of the fraction of support vectors.
oc_svm = OneClassSVM(kernel='rbf', nu=fraud_rate)

# Train the One-Class SVM model on the training data.
oc_svm.fit(X_train_scaled)

print(" Implementing Isolation Forest ---")
# The 'contamination' parameter is the expected proportion of outliers in the data.

iso_forest = IsolationForest(contamination=fraud_rate, n_estimators=300, random_state=42)


# Train the Isolation Forest model on the training data.
# It also does not use the target variable for training.
iso_forest.fit(X_train_scaled)

print("\nModels have been trained successfully!")


# Make predictions on the test set
# For One-Class SVM, a positive prediction (1) means normal, negative (-1) means an anomaly.
oc_svm_preds = oc_svm.predict(X_test_scaled)

# For Isolation Forest, the output is also 1 for inliers and -1 for outliers.
iso_forest_preds = iso_forest.predict(X_test_scaled)

# Convert the true labels (y_test) to the same -1/1 format for comparison
# 0 -> 1 (normal), 1 -> -1 (anomaly)
y_test_converted = y_test.apply(lambda x: -1 if x == 1 else 1)

print("\n Evaluation: One-Class SVM ")
print("Confusion Matrix:")
print(confusion_matrix(y_test_converted, oc_svm_preds))
print("\nClassification Report:")
print(classification_report(y_test_converted, oc_svm_preds, target_names=['Anomaly', 'Normal']))


print("\n Evaluation: Isolation Forest ")
print("Confusion Matrix:")
print(confusion_matrix(y_test_converted, iso_forest_preds))
print("\nClassification Report:")
print(classification_report(y_test_converted, iso_forest_preds, target_names=['Anomaly', 'Normal']))


numerical_features_for_pca = application_data_df.select_dtypes(include=np.number).columns.tolist()
if 'SK_ID_CURR' in numerical_features_for_pca:
    numerical_features_for_pca.remove('SK_ID_CURR')
if 'TARGET' in numerical_features_for_pca:
    numerical_features_for_pca.remove('TARGET')

# Handle missing values before scaling/PCA (e.g., imputation or dropping)
# For simplicity, we'll drop rows with NaNs in numerical features
X_pca = application_data_df[numerical_features_for_pca].dropna()


# Scale the features for plotting
y_test_plot = y_test.apply(lambda x: -1 if x == 1 else 1)

# We reduce the data to 2 principal components for plotting.
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# Add the reduced features and predictions to a DataFrame for easy plotting
plot_data = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2'])
plot_data['True_Label'] = y_test_plot.values
plot_data['OC_SVM_Pred'] = oc_svm_preds
plot_data['Iso_Forest_Pred'] = iso_forest_preds

# Visualize One-Class SVM Predictions ---
print("\n--- Visualizing One-Class SVM Predictions ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='OC_SVM_Pred',
    style='True_Label',
    palette='viridis',
    data=plot_data,
    s=20,
    alpha=0.6
)
plt.title('One-Class SVM: Predicted vs. True Labels (2D PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Prediction', loc='upper right')
plt.show()

# Visualize Isolation Forest Predictions ---
print("\n--- Visualizing Isolation Forest Predictions ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Iso_Forest_Pred',
    style='True_Label',
    palette='viridis',
    data=plot_data,
    s=20,
    alpha=0.6
)
plt.title('Isolation Forest: Predicted vs. True Labels (2D PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Prediction', loc='upper right')
plt.show()

# Provided confusion matrix values for One-Class SVM
cm_oc_svm = np.array([[1166, 6247],
                      [6417, 78424]])

# Labels for the matrix
labels = ['Anomaly', 'Normal']

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_oc_svm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: One-Class SVM')
plt.show()

cm_oc_svm = np.array([[1517, 5896],
                      [6033, 78808]])

# Labels for the matrix
labels = ['Anomaly', 'Normal']

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_oc_svm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Isolation Forrest')
plt.show()

# Compare with a supervised baseline (Logistic Regression) and random chance

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train/test split for supervised baseline (use y, not converted labels)
X_train_sup, X_test_sup, y_train_sup, y_test_sup = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_sup, y_train_sup)

y_pred_sup = log_reg.predict(X_test_sup)

print("\nSupervised Baseline (Logistic Regression):")
print(classification_report(y_test_sup, y_pred_sup, digits=4))

print(confusion_matrix(y_test_sup, y_pred_sup))

# Confusion matrix for logistic regression

cm = confusion_matrix(y_test_sup, y_pred_sup)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.xticks(ticks=[0.5, 1.5], labels=['Normal (0)', 'Anomaly (1)'])
plt.yticks(ticks=[0.5, 1.5], labels=['Normal (0)', 'Anomaly (1)'], rotation=0)
plt.show()
df_results = pd.DataFrame()

for model, metrics in results.items():
    # Figure out which label represents fraud in this report
    if "-1" in metrics:   # unsupervised anomaly detection
        fraud_label = "-1"
    else:                 # supervised baseline (fraud is "1")
        fraud_label = "1"
    
    df_results[model] = {
        "Precision (Fraud)": metrics[fraud_label]["precision"],
        "Recall (Fraud)": metrics[fraud_label]["recall"],
        "F1 (Fraud)": metrics[fraud_label]["f1-score"]
    }

df_results = df_results.T

print("\nModel Comparison (fraud class only):")
print(df_results)