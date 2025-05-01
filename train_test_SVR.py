import torch
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


# Load the train data
data_train = "" # All HC Training data

# Load the test data
data_test_bos_hc = "" # All HC Testing data
data_test_bos_an = "" # All AN Testing data

# Read the Excel sheets
data_all_hc_train = pd.read_excel(data_train)
data_test_bos_hc = pd.read_excel(data_test_bos_hc)
data_test_bos_an = pd.read_excel(data_test_bos_an)

X_all_hc_train_data = pd.DataFrame(data_all_hc_train)
X_all_hc_train_data_con = X_all_hc_train_data
print(X_all_hc_train_data_con.shape)

X_training_data = X_all_hc_train_data_con.drop(columns=["Filenames", "target"])
y_training_data = X_all_hc_train_data_con.iloc[:, -1]

training_X = X_training_data
training_y = y_training_data.values.reshape(-1, 1)
print(training_X.shape)
print(training_y.shape)

x_test_bos_hc = data_test_bos_hc.drop(columns=["Filenames", "target"])
y_test_bos_hc = data_test_bos_hc.iloc[:, -1]

x_test_bos_an = data_test_bos_an.drop(columns=["Filenames", "target"])
y_test_bos_an = data_test_bos_an.iloc[:, -1]

# Normalization
scalar = StandardScaler()

training_X = scalar.fit_transform(training_X)
X_test_bos_hc = scalar.transform(x_test_bos_hc)
X_test_bos_anx = scalar.transform(x_test_bos_an)

# Validation set scoop
df = pd.DataFrame(training_X)
df['Age'] = training_y

# Define age ranges and the number of samples to select from each range
age_ranges = [(5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 45)]

# Percentage of samples to select (5%)
percentage = 0.05

# Collect samples
selected_indices = []
for start_age, end_age in age_ranges:
    # Filter data by age range
    age_group = df[(df['Age'] >= start_age) & (df['Age'] <= end_age)]

    # Calculate 5% of the available samples in this age group
    samples_to_select = round(len(age_group) * percentage)

    # Ensure that at least one sample is selected if the group is not empty
    if samples_to_select > 0 and len(age_group) > 0:
        selected_indices.extend(age_group.sample(n=samples_to_select, random_state=None).index.tolist())

# Create the validation set using the selected indices
X_val = df.loc[selected_indices].drop(columns=['Age']).values
y_val = df.loc[selected_indices]['Age'].values

print(f"Total samples selected: {len(selected_indices)}")

# Create the training set by excluding the selected indices
df_train = df.drop(index=selected_indices)
X_train = df_train.drop(columns=['Age']).values
y_train = df_train['Age'].values

# Assuming X_train is a numpy array
if np.isnan(X_train).any():
    print("There are NaN values in X_train.")
else:
    print("There are no NaN values in X_train.")

if np.isnan(y_train).any():
    print("There are NaN values in y_train.")
else:
    print("There are no NaN values in y_train.")

# Check the shapes
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

param_grid = {
    'kernel': ['rbf'],
    'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 50, 100, 1000],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.5, 1, 2]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=kfold, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())
print(f"Best Parameters: {grid_search.best_params_}")

# Best model
best_svr = grid_search.best_estimator_

# 3. Model Evaluation on validation data
y_pred_val = best_svr.predict(X_val)
y_pred_test_bos_hc = best_svr.predict(X_test_bos_hc)
y_pred_test_bos_anx = best_svr.predict(X_test_bos_anx)

# Error Graph
errors_val = y_pred_val.flatten() - y_val.squeeze()
errors_test_bos_hc = (y_pred_test_bos_hc - y_test_bos_hc).to_numpy()
errors_test_bos_anx = (y_pred_test_bos_anx - y_test_bos_an).to_numpy()

# Steps to calculate Age Bias in Initial Test Data
# Fit a linear regression model
bias_model = LinearRegression().fit(y_val.reshape(-1, 1), y_pred_val)
intercept = bias_model.intercept_
slope = bias_model.coef_[0]

print(f'Intercept: {intercept}')
print(f'Slope: {slope}')


# Apply the Correction to New Test Data
def correct_age_bias(predictions, intercept, slope):
    return (predictions - intercept) / slope

# Apply correction
y_pred_val_corrected = correct_age_bias(y_pred_val, intercept, slope)
error_val_corr = y_pred_val_corrected.flatten() - y_val.squeeze()

y_pred_test_bos_hc_corrected = correct_age_bias(y_pred_test_bos_hc, intercept, slope)
errors_test_bos_hc_corr = y_pred_test_bos_hc_corrected - y_pred_test_bos_hc

y_pred_test_bos_anx_corrected = correct_age_bias(y_pred_test_bos_anx, intercept, slope)
errors_test_bos_anx_corr = y_pred_test_bos_anx_corrected - y_pred_test_bos_anx

def plot_predictions(plt, y_actual, y_pred, title, row_index, col_index, num_rows, num_cols):
    # Convert tensors to numpy arrays if needed
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # Ensure y_actual and y_pred are 1D
    y_actual = y_actual
    y_pred = y_pred.flatten()

    # Calculate Pearson correlation
    r_value, _ = pearsonr(y_actual, y_pred)
    annotation = f'r = {r_value:.2f}'

    # Create a DataFrame for seaborn
    data = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

    plt.subplot(num_rows, num_cols, row_index * num_cols + col_index + 1)
    # plt.scatter(y_actual, y_pred, alpha=0.5)
    # sns.scatterplot(x=y_actual, y=y_pred, data=data, alpha=0.5)
    sns.regplot(x=y_actual, y=y_pred, data=data, scatter_kws={'alpha': 0.5})  #  robust=True

    # plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='red', linestyle='--')

    # Fit a linear regression line
    # coef = np.polyfit(y_actual, y_pred, 1)
    # poly1d_fn = np.poly1d(coef)
    # plt.plot(y_actual, poly1d_fn(y_actual), color='blue', linestyle='-')

    # Plot the 45-degree line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)


    plt.xlabel('Actual Age (years)')
    plt.ylabel('Predicted Age (years')
    plt.title(title)
    plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                 ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))


# Create a figure with subplots
plt.figure(figsize=(14, 6))
# Scatter plot for validation set
plot_predictions(plt, y_val, y_pred_val, 'Actual vs. Predicted values for validation',
                 0, 0, 1, 2)
plot_predictions(plt, y_val, y_pred_val_corrected, 'Actual vs. Corrected predicted values for validation',
                 0, 1, 1, 2)

plt.figure(figsize=(14, 6))
plot_predictions(plt, y_test_bos_hc, y_pred_test_bos_hc,
                 'Actual vs. Predicted values for bos_hc',
                 0, 0, 1, 2)
plot_predictions(plt, y_test_bos_hc, y_pred_test_bos_hc_corrected,
                 'Actual vs. Corrected predicted values for bos_hc',
                 0, 1, 1, 2)

plt.figure(figsize=(14, 6))
plot_predictions(plt, y_test_bos_an, y_pred_test_bos_anx,
                 'Actual vs. Predicted values for bos_an',
                 0, 0, 1, 2)
plot_predictions(plt, y_test_bos_an, y_pred_test_bos_anx_corrected,
                 'Actual vs. Corrected predicted values for bos_an',
                 0, 1, 1, 2)

plt.show()

# Metrics
# Function to compute metrics
def compute_metrics(y_true, y_pred):
    # Ensure y_pred and y_true are numpy arrays
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    elif isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()  # Ensure tensors are moved to CPU and detached
    elif isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    elif isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()  # Ensure tensors are moved to CPU and detached
    elif isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    # Compute residuals
    residuals = y_pred - y_true

    # Calculate Pearson correlation between y_true and residuals
    age_bias, _ = pearsonr(y_true, residuals)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    return correlation, mse, mae, rmse, age_bias


# Compute metrics
def print_metrics(title, y_actual, y_pred):
    correlation, _, mae, rmse, age_bias = compute_metrics(y_actual, y_pred)
    print(f"{title}:")
    print(f"Age Bias: {age_bias}")
    print(f"Correlation Coefficient: {correlation}")
    print(f"Mean Absolute Error: {mae}")
    print(f"RMSE: {rmse}\n")


# Function to evaluate test results metrics for corrected age biased
def test_results(predictions, true_labels):
    # Ensure both arrays are 1D
    predictions = predictions.flatten()
    true_labels = true_labels

    # Compute residuals
    residuals_corr = predictions - true_labels

    # Calculate Pearson correlation between y_true and residuals
    age_bias_corr, _ = pearsonr(true_labels, residuals_corr)

    # Calculate correlation
    correlation = np.corrcoef(predictions, true_labels)[0, 1]

    # Calculate other metrics
    mse = np.mean((predictions - true_labels) ** 2)
    mae = np.mean(np.abs(predictions - true_labels))
    rmse = np.sqrt(mse)

    return correlation, mse, mae, rmse, age_bias_corr


def print_metrics_corr(title, true_labels, predictions):
    correlation, _, mae, rmse, age_bias_corr = test_results(true_labels, predictions)
    print(f"{title}:")
    print(f"Age Bias for corrected: {age_bias_corr}")
    print(f'Correlation for corrected: {correlation}')
    print(f'MAE for corrected: {mae}')
    print(f'RMSE for corrected: {rmse}')

print_metrics("Validation-Set", y_val, y_pred_val)
print_metrics_corr("Validation-Set-CORR", y_pred_val_corrected, y_val)

print_metrics("BOS-HC", y_test_bos_hc, y_pred_test_bos_hc)
print_metrics_corr("BOS-HC-CORR", y_pred_test_bos_hc_corrected, y_test_bos_hc)

print_metrics("BOS-AN", y_test_bos_an, y_pred_test_bos_anx)
print_metrics_corr("BOS-AN-CORR", y_pred_test_bos_anx_corrected, y_test_bos_an)
















