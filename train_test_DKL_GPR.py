import torch
import gpytorch
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from torch import nn
import itertools

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

# Convert to torch tensors
train_x = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.float32).squeeze()
test_x = torch.tensor(X_val, dtype=torch.float32).cuda()
test_y = torch.tensor(y_val, dtype=torch.float32).squeeze().cuda()

X_test_bos_hc = torch.tensor(X_test_bos_hc, dtype=torch.float32).cuda()
y_test_bos_hc = torch.tensor(y_test_bos_hc.values, dtype=torch.float32).cuda()

X_test_bos_anx = torch.tensor(X_test_bos_anx, dtype=torch.float32).cuda()
y_test_bos_anx = torch.tensor(y_test_bos_an.values, dtype=torch.float32).cuda()

print("train_x:", train_x.shape)
print("train_y:", train_y.shape)
print("test_x:", test_x.shape)
print("test_y:", test_y.shape)
print("X_test_bos_hc:", X_test_bos_hc.shape)
print("y_test_bos_hc:", y_test_bos_hc.shape)
print("X_test_bos_anx:", X_test_bos_anx.shape)
print("y_test_bos_anx:", y_test_bos_anx.shape)

data_dim = train_x.size(-1)

class DKLFeatureExtractor(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(DKLFeatureExtractor, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DKLGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, lengthscale=1.0, outputscale=1.0):
        super(DKLGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
        )

        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = outputscale

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define hyperparameter grid
layer_options = [[data_dim, 100, 50], [data_dim, 200, 100, 50]]
lengthscale_options = [0.5, 1.0]
outputscale_options = [0.5, 1.0]

# Create combinations of hyperparameters
param_grid = list(itertools.product(layer_options, lengthscale_options, outputscale_options))

# Store best results
best_mae = float('inf')
best_params = None

# Use 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for idx, (layers, lengthscale_val, outputscale_val) in enumerate(param_grid):
    print(f"\nHyperparam Combo {idx+1}: Layers={layers}, Lengthscale={lengthscale_val}, Outputscale={outputscale_val}")

    fold_maes = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        print(f"  Fold {fold+1}/5")

        X_fold_train = train_x[train_idx].cuda()
        y_fold_train = train_y[train_idx].cuda()
        X_fold_val = train_x[val_idx].cuda()
        y_fold_val = train_y[val_idx].cuda()

        # Define model
        feature_extractor = DKLFeatureExtractor(input_dim=4, layer_sizes=layers)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DKLGPRegressionModel(X_fold_train, y_fold_train, likelihood, feature_extractor, lengthscale_val, outputscale_val).cuda()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()
        epoch_loop = 100
        for i in range(epoch_loop):
            optimizer.zero_grad()
            output = model(X_fold_train)
            loss = -mll(output, y_fold_train)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred = likelihood(model(X_fold_val)).mean.cpu().numpy().flatten()
            true = y_fold_val.cpu().numpy().flatten()
            mae = np.mean(np.abs(pred - true))
            fold_maes.append(mae)

    mean_mae = np.mean(fold_maes)
    print(f"  Avg MAE: {mean_mae:.3f}")
    if mean_mae < best_mae:
        best_mae = mean_mae
        best_params = (layers, lengthscale_val, outputscale_val)

print("\nBest Hyperparameters:")
print(f"Layers: {best_params[0]}, Lengthscale: {best_params[1]}, Outputscale: {best_params[2]}")
print(f"Best MAE: {best_mae:.3f}")

best_layer = best_params[0]
best_length_scale = best_params[1]
best_output_scale = best_params[2]

# Exact DKL (Deep Kernel Learning)
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_features, output_features):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_features, best_layer[0]))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('batchnorm1', torch.nn.BatchNorm1d(best_layer[0]))  # Batch normalization

        self.add_module('linear2', torch.nn.Linear(best_layer[0], best_layer[1]))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('batchnorm2', torch.nn.BatchNorm1d(best_layer[1]))  # Batch normalization

        self.add_module('linear3', torch.nn.Linear(best_layer[1], best_layer[2]))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('batchnorm3', torch.nn.BatchNorm1d(best_layer[2]))  # Batch normalization

        self.add_module('linear4', torch.nn.Linear(best_layer[2], output_features))


# Define the GP regression model
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, ), outputscale_prior=outputscale_prior)

        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = best_length_scale
        self.covar_module.outputscale = best_output_scale

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

feature_extractor = LargeFeatureExtractor(data_dim, 4)

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))

# Changing the constraint after the module has been created
likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

model = GPRegressionModel(train_x, train_y, likelihood).cuda()

print(
    model.likelihood.noise_covar.noise.item(),
    model.covar_module.base_kernel.lengthscale.item(),
    model.covar_module.outputscale.item()
)
num_epochs = 1000

# Define the optimizer with the best learning rate
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.scale_to_bounds.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.001)

# # Using the GPU
train_x = train_x.cuda()
train_y = train_y.cuda()
likelihood = likelihood.cuda()

model.train()
likelihood.train()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# mll.jitter = 1e-3  # Increase jitter value

for i in range(num_epochs):
    # Zero backprop gradients
    optimizer.zero_grad()
    # Get output from model
    output = model(train_x)
    # Calc loss and backprop derivatives
    loss = -mll(output, train_y)
    loss.backward()
    # iterator.set_postfix(loss=loss.item())

    optimizer.step()

    print('Iter %d/%d - Loss: %.3f' % (i + 1, num_epochs, loss.item()))

    torch.cuda.empty_cache()

torch.save(model.state_dict(), "Test_Model.pt")

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Pass test data through the model to get predictions
with torch.no_grad(), gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():

    observed_y_pred_val = likelihood(model(test_x))
    observed_y_pred_test_bos_hc = likelihood(model(X_test_bos_hc))
    observed_y_pred_test_bos_anx = likelihood(model(X_test_bos_anx))

    y_pred_val = observed_y_pred_val.mean.cpu().detach().numpy()
    y_pred_test_bos_hc = observed_y_pred_test_bos_hc.mean.cpu().detach().numpy()
    y_pred_test_bos_anx = observed_y_pred_test_bos_anx.mean.cpu().detach().numpy()

    # Error Graph
    errors_val = y_pred_val.flatten() - y_val.squeeze()
    errors_test_bos_hc = y_pred_test_bos_hc - y_test_bos_hc.cpu().detach().numpy()
    errors_test_bos_anx = y_pred_test_bos_anx - y_test_bos_anx.cpu().detach().numpy()

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
errors_test_bos_hc_corr = y_pred_test_bos_hc_corrected - y_test_bos_hc.cpu().detach().numpy()

y_pred_test_bos_anx_corrected = correct_age_bias(y_pred_test_bos_anx, intercept, slope)
errors_test_bos_anx_corr = y_pred_test_bos_anx_corrected - y_test_bos_anx.cpu().detach().numpy()

def plot_predictions(plt, y_actual, y_pred, title, row_index, col_index, num_rows, num_cols):
    # Convert tensors to numpy arrays if needed
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    # Ensure y_actual and y_pred are 1D
    y_actual = y_actual.flatten()
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
plot_predictions(plt, y_test_bos_hc.cpu().numpy(), y_pred_test_bos_hc,
                 'Actual vs. Predicted values for bos_hc',
                 0, 0, 1, 2)
plot_predictions(plt, y_test_bos_hc.cpu().numpy(), y_pred_test_bos_hc_corrected,
                 'Actual vs. Corrected predicted values for bos_hc',
                 0, 1, 1, 2)

plt.figure(figsize=(14, 6))
plot_predictions(plt, y_test_bos_anx.cpu().numpy(), y_pred_test_bos_anx,
                 'Actual vs. Predicted values for bos_an',
                 0, 0, 1, 2)
plot_predictions(plt, y_test_bos_anx.cpu().numpy(), y_pred_test_bos_anx_corrected,
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
    true_labels = true_labels.flatten()

    # Check the shapes
    # print(f'Predictions shape: {predictions.shape}, True labels shape: {true_labels.shape}')

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

print_metrics("Validation-Set", test_y.cpu().detach().numpy(), y_pred_val)
print_metrics_corr("Validation-Set-CORR", y_pred_val_corrected, y_val)

print_metrics("BOS-HC", y_test_bos_hc.cpu().detach().numpy(), y_pred_test_bos_hc)
print_metrics_corr("BOS-HC-CORR", y_pred_test_bos_hc_corrected, y_test_bos_hc.cpu().detach().numpy())

print_metrics("BOS-AN", y_test_bos_anx.cpu().detach().numpy(), y_pred_test_bos_anx)
print_metrics_corr("BOS-AN-CORR", y_pred_test_bos_anx_corrected, y_test_bos_anx.cpu().detach().numpy())

































